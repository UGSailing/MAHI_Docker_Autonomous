"""
padplanning_buoy.py
Teardrop-parcour rond één boei:
  1. Inkomende boog  – Hermite-spline van start_pos naar de rechterzijde van de boei
  2. Halve cirkel    – 180° CCW om de boei (rechts → achterkant → links)
  3. Terugkerende boog – Hermite-spline van de linkerzijde terug naar start_pos

"Rechts" en "links" zijn gedefinieerd t.o.v. de as start_pos → boei:
  • rechts = 90° met de klok mee (stuurboord)
  • links  = 90° tegen de klok in (bakboord)

De halve cirkel gaat CCW: de boot rijdt eerst weg van start (achterkant),
dan terug naar de linkerzijde.

Waypoint-afstand: identiek aan padplanning_slalom — via INTERPOLATE_USING_DISTANCE
en WAYPOINT_DISTANCE (of de vaste telpunten N_ARC_PTS / N_SLALOM_PTS).

Snelheidsramping: gebaseerd op RAMP_ACCELERATION (m/s²) i.p.v. een vast
aantal waypoints — de ramp-lengte schaalt mee met het snelheidsverschil:
  • inkomende boog  → FAST  (ramp van 0 → FAST aan start, FAST → SLOW aan einde)
  • halve cirkel    → SLOW  (constante snelheid, geen ramp)
  • terugkerende boog → FAST (ramp van SLOW → FAST aan start, FAST → 0 aan einde)

De functie-handtekening is compatibel met padplanning_wrapper:
  padplanning_buoy(buoys, boat_pos, state, slalom_offset,
                   start_pos=..., start_heading_deg=...)
waarbij `buoys` een lijst met één (lon, lat)-tupel is.
"""

import math
import .communication.get_mqtt as get_mqtt
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    N_SLALOM_PTS,
    N_ARC_PTS,
    FAST_SPEED,
    SLOW_SPEED,
    WAYPOINT_DISTANCE,
    INTERPOLATE_USING_DISTANCE,
    RAMP_ACCELERATION,
)

Position = Tuple[float, float]   # (longitude, latitude)


# ─────────────────────────────────────────────────────────────────────────────
#  Speed-ramp helper  (versnelling-gebaseerd i.p.v. vast aantal waypoints)
# ─────────────────────────────────────────────────────────────────────────────
#
# In plaats van een vast aantal ramp-waypoints (N_RAMP_WAYPOINTS) wordt de
# ramp-lengte nu afgeleid uit RAMP_ACCELERATION (m/s²): hoe groter het
# snelheidsverschil, hoe langer (in meters/waypoints) de ramp moet zijn.
#
# Kinematica voor constante versnelling a, van v0 naar v1:
#   afstand  d = |v1² - v0²| / (2a)
# Het aantal waypoints dat nodig is om die afstand te overbruggen hangt af
# van de lokale waypoint-spacing (meters/waypoint) van het segment waarin
# de ramp valt.

def _ramp_distance(v0: float, v1: float) -> float:
    """Afstand (m) nodig om van v0 naar v1 te versnellen/vertragen met
    constante RAMP_ACCELERATION."""
    if RAMP_ACCELERATION <= 0:
        return 0.0
    return abs(v1 * v1 - v0 * v0) / (2.0 * RAMP_ACCELERATION)


def _ramp_n_waypoints(v0: float, v1: float, spacing: float) -> int:
    """Aantal tussenliggende ramp-waypoints voor een overgang v0 → v1,
    gegeven de waypoint-spacing (m) van het betreffende segment."""
    if RAMP_ACCELERATION <= 0 or spacing <= 0:
        return 0
    d = _ramp_distance(v0, v1)
    return max(0, round(d / spacing))


def _ramp(from_speed: float, to_speed: float, spacing: float) -> List[float]:
    """
    Return een lijst van tussenliggende snelheden tussen from_speed en
    to_speed (exclusief beide eindpunten), met:
      • een lengte (aantal waypoints) die overeenkomt met de afstand die
        nodig is om te versnellen/vertragen met RAMP_ACCELERATION
      • snelheden die de juiste kinematische v(s)-curve volgen voor
        constante versnelling, NIET lineair geïnterpoleerd in waypoint-index.

    Voor constante versnelling a geldt  v(s) = sqrt(v0² ± 2·a·s),
    waarbij s de afgelegde afstand is vanaf het begin van de ramp.
    Lineair interpoleren in waypoint-index (i.p.v. in s via deze formule)
    zou een vertekend profiel geven: te traag versnellen aan het begin,
    te snel aan het einde van de ramp.
    """
    n_ramp = _ramp_n_waypoints(from_speed, to_speed, spacing)
    if n_ramp <= 0:
        return []

    accelerating = to_speed > from_speed
    sign = 1.0 if accelerating else -1.0

    speeds = []
    for i in range(n_ramp):
        s = (i + 1) * spacing                       # afgelegde afstand vanaf ramp-start
        v_sq = from_speed * from_speed + sign * 2.0 * RAMP_ACCELERATION * s
        v = math.sqrt(max(0.0, v_sq))
        # Clip tegen het doel zodat afrondingen in n_ramp niet voorbij to_speed schieten
        v = min(v, to_speed) if accelerating else max(v, to_speed)
        speeds.append(v)
    return speeds


def _tag_with_ramp(
    waypoints: List[np.ndarray],
    body_speed: float,
    prev_speed: Optional[float],   # speed of the last waypoint before this segment
    next_speed: Optional[float],   # speed of the first waypoint after this segment
    spacing: float,                # m/waypoint within this segment
) -> List[Tuple[np.ndarray, float]]:
    """
    Tag a list of waypoints with speeds, adding ramp-up at the start and
    ramp-down at the end when the adjacent segment has a different speed.

    The ramp consumes the first / last N waypoints of the segment, where N
    is derived from RAMP_ACCELERATION and this segment's spacing.
    If the segment is too short to fit both ramps, the ramp is clipped.
    """
    n = len(waypoints)
    speeds = [body_speed] * n

    # Ramp up (start of segment)
    if prev_speed is not None and prev_speed != body_speed:
        ramp_up = _ramp(prev_speed, body_speed, spacing)   # ascending toward body_speed
        for j, s in enumerate(ramp_up):
            if j < n:
                speeds[j] = s

    # Ramp down (end of segment)
    if next_speed is not None and next_speed != body_speed:
        ramp_dn = _ramp(body_speed, next_speed, spacing)   # descending toward next_speed
        for j, s in enumerate(reversed(ramp_dn)):
            idx = n - 1 - j
            if idx >= 0:
                # Only overwrite if this cell wasn't already claimed by ramp-up
                if speeds[idx] == body_speed:
                    speeds[idx] = s

    return list(zip(waypoints, speeds))


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper  (druppelt door naar padplanning_buoy; vervangt padplanning_wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def padplanning_wrapper(buoy_positions, marge, state='START',
                        start_position=None, start_heading_deg=None):
    """
    Identieke interface als de originele padplanning_wrapper, maar voor één boei.

    Parameters
    ----------
    buoy_positions    : lijst met één puntenwolk: [[(lat, lon), ...]]
    marge             : veiligheidsmarge bovenop de grootste boeistraal (m)
    state             : 'START' | 'DETECT_1' | ...
    start_position    : (lat, lon) startpositie boot (verplicht)
    start_heading_deg : initiële koers bij start_position (verplicht)
    """
    if start_position is None:
        raise ValueError("start_position is required and must be a (lat, lon) tuple.")
    if start_heading_deg is None:
        raise ValueError("start_heading_deg is required.")

    R_EARTH = 6_371_000.0

    buoys_centers  = []
    buoys_max_dist = []

    for buoy_pos in buoy_positions:
        total_lat = sum(c[0] for c in buoy_pos)
        total_lon = sum(c[1] for c in buoy_pos)
        n = len(buoy_pos)
        center_lat = total_lat / n
        center_lon = total_lon / n
        buoys_centers.append((center_lon, center_lat))   # (lon, lat) intern

        lat_rad = math.radians(center_lat)
        max_d = 0.0
        for lat, lon in buoy_pos:
            dn = math.radians(lat - center_lat) * R_EARTH
            de = math.radians(lon - center_lon) * R_EARTH * math.cos(lat_rad)
            max_d = max(max_d, math.hypot(de, dn))
        buoys_max_dist.append(max_d)

    boat_position = get_mqtt.get_boat_position()
    boat_pos = (boat_position['longitude'], boat_position['latitude'])

    slalom_offset = max(buoys_max_dist) + marge

    start_pos_lonlat = (start_position[1], start_position[0])   # → (lon, lat)

    waypoints = padplanning_buoy(
        buoys_centers,
        boat_pos,
        state,
        slalom_offset,
        start_pos=start_pos_lonlat,
        start_heading_deg=start_heading_deg,
    )

    # (lon, lat) → (lat, lon), behoud speed
    return [((lat_wp, lon_wp), speed) for (lon_wp, lat_wp), speed in waypoints]


# ─────────────────────────────────────────────────────────────────────────────
#  Core planning function
# ─────────────────────────────────────────────────────────────────────────────

def padplanning_buoy(
    buoys: List[Position],
    boat_pos: Position,
    state: str,
    slalom_offset: float = 12.0,
    n_arc_pts: int = N_ARC_PTS,
    n_slalom_pts: int = N_SLALOM_PTS,
    waypoint_distance: float = WAYPOINT_DISTANCE,
    start_pos: Optional[Position] = None,
    start_heading_deg: float = 0.0,
) -> List[Tuple[Position, float]]:
    """
    Berekent het teardrop-parcour rond één boei.

    Padopbouw
    ─────────
      1. Inkomende boog  – Hermite-spline van start_pos (koers start_heading_deg)
                           naar punt R (marge rechts van de as start→boei),
                           aankomend met tangent langs de cirkel (CCW-richting)
      2. Halve cirkel    – 180° CCW om de boei van R via de achterkant naar L
      3. Terugkerende boog – Hermite-spline van punt L (tangent langs de cirkel)
                             terug naar start_pos met omgekeerde aankomstkoers

    Parameters
    ----------
    buoys             : [(lon, lat)]  – lijst met één boei-positie
    boat_pos          : (lon, lat) huidige positie boot (coördinaten-referentie)
    state             : gereserveerd voor toekomstige uitbreiding
    slalom_offset     : afstand boei → R / L  EN straal van de halve cirkel (m)
    n_arc_pts         : waypoints voor de halve cirkel (INTERPOLATE_USING_DISTANCE=False)
    n_slalom_pts      : waypoints voor elke boog     (idem)
    waypoint_distance : gewenste afstand tussen waypoints in m
                        (INTERPOLATE_USING_DISTANCE=True)
    start_pos         : (lon, lat) startpositie aanrijroute (verplicht)
    start_heading_deg : initiële koers bij start_pos (graden, Noord = 0)

    Returns
    -------
    [((lon, lat), speed), ...]
    """

    if start_pos is None:
        raise ValueError("start_pos is required.")

    R_EARTH = 6_371_000.0
    ref_lon, ref_lat = boat_pos
    cos_lat       = math.cos(math.radians(ref_lat))
    m_per_deg_lon = cos_lat * math.pi / 180.0 * R_EARTH
    m_per_deg_lat =           math.pi / 180.0 * R_EARTH

    def to_xy(lon: float, lat: float) -> np.ndarray:
        return np.array([
            (lon - ref_lon) * m_per_deg_lon,
            (lat - ref_lat) * m_per_deg_lat,
        ])

    def to_lonlat(p: np.ndarray) -> Position:
        return (
            ref_lon + p[0] / m_per_deg_lon,
            ref_lat + p[1] / m_per_deg_lat,
        )

    # ── Lokale coördinaten ────────────────────────────────────────────────────
    buoy_xy   = to_xy(*buoys[0])
    start_xy  = to_xy(*start_pos)

    # As start → boei
    axis_vec  = buoy_xy - start_xy
    axis_d    = np.linalg.norm(axis_vec)
    if axis_d < 1e-3:
        raise ValueError("start_pos en buoy liggen op dezelfde positie.")
    axis_u    = axis_vec / axis_d

    # Loodrechte eenheidsvectoren
    # perp_right = 90° CW van axis_u  (stuurboord)
    # perp_left  = 90° CCW van axis_u (bakboord)
    perp_right = np.array([ axis_u[1], -axis_u[0]])
    perp_left  = np.array([-axis_u[1],  axis_u[0]])

    # ── Sleutelposities ───────────────────────────────────────────────────────
    # R = rechts van de boei  (instappunt halve cirkel)
    # L = links  van de boei  (uitstappunt halve cirkel)
    R = buoy_xy + slalom_offset * perp_right
    L = buoy_xy + slalom_offset * perp_left

    # ── Waypoint-aantallen bepalen ────────────────────────────────────────────
    half_circ  = math.pi * slalom_offset     # omtrek halve cirkel
    approach_d = np.linalg.norm(R - start_xy)

    if INTERPOLATE_USING_DISTANCE:
        n_half   = max(2, round(half_circ    / waypoint_distance))
        n_in     = max(2, round(approach_d   / waypoint_distance))
        n_out    = n_in   # retour heeft zelfde afstand
    else:
        n_half = n_arc_pts
        n_in   = n_slalom_pts
        n_out  = n_slalom_pts

    # ── Halve cirkel: CCW van R (rechts) via achterkant naar L (links) ────────
    # Hoek van R t.o.v. het middelpunt van de cirkel (= buoy_xy):
    #   R ligt op +perp_right van buoy, dus hoek = atan2(perp_right)
    # CCW-sweep van 180° brengt ons van R naar L.
    angle_R = math.atan2(perp_right[1], perp_right[0])
    # CCW = positieve sweep
    half_circle_pts = [
        buoy_xy + slalom_offset * np.array([
            math.cos(angle_R + math.pi * i / n_half),
            math.sin(angle_R + math.pi * i / n_half),
        ])
        for i in range(n_half + 1)
    ]

    # Sanity-check: het eindpunt moet ≈ L zijn
    # (angle_R + π ligt in de richting van perp_left ✓)

    # ── Tangent-richtingen aan R en L ─────────────────────────────────────────
    # Bij een CCW-cirkel staat de tangent loodrecht op de straal, in CCW-richting.
    # Straal naar R  = +perp_right  →  CCW-tangent = +axis_u  (vooruit langs de as)
    # Straal naar L  = +perp_left   →  CCW-tangent = -axis_u  (terug langs de as)
    tangent_at_R = axis_u           # richting waaruit de boot bij R aankomt
    tangent_at_L = -axis_u          # richting waaruit de boot bij L vertrekt

    # ── Hermite-spline helper ─────────────────────────────────────────────────
    def hermite_arc(P0: np.ndarray, T0_uv: np.ndarray,
                    P1: np.ndarray, T1_uv: np.ndarray,
                    n: int) -> List[np.ndarray]:
        """
        Kubische Hermite-spline van P0 naar P1.
        T0_uv / T1_uv zijn eenheidsvectoren; ze worden geschaald op de
        koordlengte zodat de bochtvorm proportioneel blijft (× 2.0 geeft
        een beetje extra 'doorzet' zodat de koers al vroeg is vastgelegd).
        """
        chord = np.linalg.norm(P1 - P0)
        T0 = T0_uv * chord * 2.0
        T1 = T1_uv * chord * 2.0

        pts = []
        for i in range(n + 1):
            t  = i / n
            t2 = t * t
            t3 = t2 * t
            h00 =  2*t3 - 3*t2 + 1
            h10 =    t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 =    t3 -   t2
            pts.append(h00*P0 + h10*T0 + h01*P1 + h11*T1)
        return pts

    # ── Koers van de boot bij start_pos ──────────────────────────────────────
    hdg_rad    = math.radians(start_heading_deg)
    heading_uv = np.array([math.sin(hdg_rad), math.cos(hdg_rad)])

    # ── Segment 1: inkomende boog  start_pos → R ─────────────────────────────
    # Vertrekt met heading_uv, arriveert met tangent_at_R (= axis_u)
    in_arc_pts = hermite_arc(start_xy, heading_uv,
                             R,        tangent_at_R,
                             n_in)

    # ── Segment 3: terugkerende boog  L → start_pos ──────────────────────────
    # Vertrekt vanuit L met tangent_at_L (= -axis_u),
    # arriveert bij start_pos met -heading_uv (omgekeerde aankomstkoers)
    out_arc_pts = hermite_arc(L,        tangent_at_L,
                              start_xy, -heading_uv,
                              n_out)

    # ── Segmenten samenvoegen (geen dubbele grenspunten) ─────────────────────
    F, S = FAST_SPEED, SLOW_SPEED

    # Spacing (m/waypoint) per segment — nodig om RAMP_ACCELERATION om te
    # zetten naar een waypoint-aantal binnen elk segment.
    in_arc_len  = sum(np.linalg.norm(in_arc_pts[i + 1] - in_arc_pts[i])
                      for i in range(len(in_arc_pts) - 1))
    half_len    = sum(np.linalg.norm(half_circle_pts[i + 1] - half_circle_pts[i])
                      for i in range(len(half_circle_pts) - 1))
    out_arc_len = sum(np.linalg.norm(out_arc_pts[i + 1] - out_arc_pts[i])
                      for i in range(len(out_arc_pts) - 1))

    in_arc_spacing  = in_arc_len  / max(1, n_in)
    half_spacing    = half_len    / max(1, n_half)
    out_arc_spacing = out_arc_len / max(1, n_out)

    # (puntenlijst, snelheid, spacing)
    raw_segments: List[Tuple[List[np.ndarray], float, float]] = [
        (in_arc_pts,      F, in_arc_spacing),
        (half_circle_pts, S, half_spacing),
        (out_arc_pts,     F, out_arc_spacing),
    ]

    # Elk segment behalve het eerste: verwijder het eerste punt (grenspunt)
    segments: List[Tuple[List[np.ndarray], float, float]] = []
    for si, (pts_seg, spd, spacing) in enumerate(raw_segments):
        trimmed = pts_seg if si == 0 else pts_seg[1:]
        if trimmed:
            segments.append((trimmed, spd, spacing))

    # ── Snelheidsramping ──────────────────────────────────────────────────────
    # Aan de uiterste uiteinden (vóór de eerste boog / na de laatste boog) is er
    # geen aangrenzend segment. Daar wordt 0.0 (stilstand) als impliciete
    # grenswaarde gebruikt, zodat de boot bij start_pos vanuit stilstand
    # optrekt en weer tot stilstand afremt, i.p.v. abrupt op FAST_SPEED te
    # beginnen/eindigen of te blijven "kruisen" op SLOW_SPEED.
    #
    # De ramp-lengte (in waypoints) wordt per segment afgeleid uit
    # RAMP_ACCELERATION en de spacing van dat segment — een groter
    # snelheidsverschil (bv. 0 → FAST) geeft dus een langere ramp dan een
    # kleiner verschil (bv. SLOW → FAST).
    all_waypoints: List[Tuple[np.ndarray, float]] = []
    n_segs = len(segments)
    for si, (pts_seg, body_spd, spacing) in enumerate(segments):
        if body_spd == F:
            prev_spd = segments[si - 1][1] if si > 0          else 0.0
            next_spd = segments[si + 1][1] if si < n_segs - 1 else 0.0
            tagged = _tag_with_ramp(pts_seg, body_spd, prev_spd, next_spd, spacing)
        else:
            tagged = [(wp, body_spd) for wp in pts_seg]
        all_waypoints.extend(tagged)

    # ── Coördinaten terug naar (lon, lat) ────────────────────────────────────
    return [(to_lonlat(wp), speed) for wp, speed in all_waypoints]


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math as _m
    import types, sys

    # Stubs
    _mqtt = types.ModuleType("get_mqtt")
    _mqtt.get_boat_position = lambda: {"longitude": 4.35, "latitude": 51.92}
    sys.modules.setdefault("get_mqtt", _mqtt)

    _cfg = types.ModuleType("config")
    _cfg.N_SLALOM_PTS            = 12
    _cfg.N_ARC_PTS               = 12
    _cfg.FAST_SPEED              = 4.0
    _cfg.SLOW_SPEED              = 2.0
    _cfg.WAYPOINT_DISTANCE       = .2
    _cfg.INTERPOLATE_USING_DISTANCE = True
    _cfg.RAMP_ACCELERATION       = 0.2   # m/s²
    sys.modules.setdefault("config", _cfg)

    R = 6_371_000.0
    lat0, lon0 = 51.92, 4.35
    mpd_lon = _m.cos(_m.radians(lat0)) * _m.pi / 180 * R
    mpd_lat = _m.pi / 180 * R

    def off(dx, dy):
        return (lon0 + dx / mpd_lon, lat0 + dy / mpd_lat)

    buoy  = off(50, 0)    # boei 50 m ten oosten van de boot
    bpos  = off(0, 0)     # bootpositie (referentie)
    spos  = off(-30, 0)   # startpositie 30 m ten westen

    pad = padplanning_buoy(
        [buoy], bpos, "START",
        slalom_offset=12.0,
        start_pos=spos,
        start_heading_deg=90.0,   # rijdt naar het oosten
    )

    print(f"Waypoints: {len(pad)}")
    speeds = list(dict.fromkeys(round(s, 1) for _, s in pad))
    print(f"Unieke snelheden: {speeds}")
    print(f"Eerste WP: {pad[0]}")
    print(f"Laatste WP: {pad[-1]}")