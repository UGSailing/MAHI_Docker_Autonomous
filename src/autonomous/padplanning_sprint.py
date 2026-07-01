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

Waypoint-afstand: schaalbaar met snelheid via WAYPOINT_DT (seconden per
waypoint-interval). De afstand tussen twee opeenvolgende waypoints is dus
  d = v * WAYPOINT_DT
zodat waypoints dichter bij elkaar liggen bij lage snelheid en verder van
elkaar bij hoge snelheid. Dit garandeert dat lineaire interpolatie van
snelheid over waypoints overeenkomt met lineaire interpolatie over tijd.

Snelheidsramping: gebaseerd op RAMP_ACCELERATION (m/s²). Met een vaste
tijdstap WAYPOINT_DT stijgt/daalt de snelheid per waypoint met het
constante bedrag:
  Δv = RAMP_ACCELERATION * WAYPOINT_DT
De ramp is dus een rekenkundige rij van snelheden, en de positie-spacing
van elk ramp-waypoint is v_i * WAYPOINT_DT — eenvoudiger dan de vroegere
sqrt(v² ± 2as)-formule.
"""

import math
from .communication import get_mqtt
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    N_SLALOM_PTS,
    N_ARC_PTS,
    FAST_SPEED,
    SLOW_SPEED,
    WAYPOINT_DT,
    RAMP_ACCELERATION,
)

Position = Tuple[float, float]   # (longitude, latitude)


# ─────────────────────────────────────────────────────────────────────────────
#  Speed-ramp helper  (tijd-domein, vaste tijdstap)
# ─────────────────────────────────────────────────────────────────────────────
#
# Met een vaste tijdstap WAYPOINT_DT en constante versnelling a geldt:
#   v_{i+1} = v_i ± a * WAYPOINT_DT
#
# De ramp bevat alle tussenliggende snelheden (exclusief from_speed en
# to_speed zelf). Elk ramp-waypoint wordt later geplaatst op afstand
# v_i * WAYPOINT_DT van zijn voorganger.

def _ramp(from_speed: float, to_speed: float) -> List[float]:
    """
    Geeft een lijst van tussenliggende snelheden voor de overgang
    from_speed → to_speed bij constante RAMP_ACCELERATION en tijdstap
    WAYPOINT_DT (beide eindpunten zijn uitgesloten).

    De snelheid verandert per stap met:
        dv = RAMP_ACCELERATION * WAYPOINT_DT

    Het aantal stappen is:
        n = round(|to_speed - from_speed| / dv)
    """
    if RAMP_ACCELERATION <= 0 or WAYPOINT_DT <= 0:
        return []

    dv = RAMP_ACCELERATION * WAYPOINT_DT
    n_ramp = max(0, round(abs(to_speed - from_speed) / dv))
    if n_ramp <= 0:
        return []

    sign = 1.0 if to_speed > from_speed else -1.0
    speeds = []
    for i in range(n_ramp):
        v = from_speed + sign * dv * (i + 1)
        # Clip zodat afrondingen in n_ramp niet voorbij to_speed schieten
        v = min(v, to_speed) if sign > 0 else max(v, to_speed)
        speeds.append(v)

    # Verwijder het laatste element als dat exact to_speed is
    # (to_speed is het eerste punt van het volgende segment, niet van de ramp)
    if speeds and abs(speeds[-1] - to_speed) < 1e-9:
        speeds.pop()

    return speeds


def _tag_with_ramp(
    waypoints: List[np.ndarray],
    body_speed: float,
    prev_speed: Optional[float],   # snelheid van het laatste WP vóór dit segment
    next_speed: Optional[float],   # snelheid van het eerste WP ná dit segment
) -> List[Tuple[np.ndarray, float]]:
    """
    Koppelt snelheden aan een lijst waypoints, met ramp-up aan het begin
    en ramp-down aan het einde wanneer het aangrenzende segment een andere
    snelheid heeft.

    De ramp verbruikt de eerste / laatste N waypoints van het segment,
    waarbij N = len(_ramp(...)). Als het segment te kort is voor beide
    rampen wordt de ramp geclipped.
    """
    n = len(waypoints)
    speeds = [body_speed] * n

    # Ramp up (begin van het segment)
    if prev_speed is not None and prev_speed != body_speed:
        ramp_up = _ramp(prev_speed, body_speed)
        for j, s in enumerate(ramp_up):
            if j < n:
                speeds[j] = s

    # Ramp down (einde van het segment)
    if next_speed is not None and next_speed != body_speed:
        ramp_dn = _ramp(body_speed, next_speed)
        for j, s in enumerate(reversed(ramp_dn)):
            idx = n - 1 - j
            if idx >= 0 and speeds[idx] == body_speed:
                speeds[idx] = s

    return list(zip(waypoints, speeds))


# ─────────────────────────────────────────────────────────────────────────────
#  Waypoint-plaatsing op basis van tijd
# ─────────────────────────────────────────────────────────────────────────────

def _place_waypoints_by_time(
    curve_pts: List[np.ndarray],
    speeds: List[float],
) -> List[np.ndarray]:
    """
    Herplaatst waypoints langs een gegeven curve zodanig dat de afstand
    tussen opeenvolgende waypoints i en i+1 gelijk is aan:
        d_i = speeds[i] * WAYPOINT_DT

    De invoer `curve_pts` is een dichte puntenwolk die de curve definieert
    (bijv. de Hermite-spline of de halve cirkel met veel punten).
    `speeds` heeft dezelfde lengte als het gewenste uitvoer-waypoint-aantal.

    Algoritme:
      • Bouw een cumulatieve booglengte-tabel van curve_pts.
      • Loop over de gewenste waypoints; bereken per stap de doelafstand
        speeds[i] * WAYPOINT_DT en zoek het bijbehorende punt op de curve
        via lineaire interpolatie in de booglengte-tabel.
    """
    # Cumulatieve booglengte van de ruwe curve
    arc = [0.0]
    for k in range(len(curve_pts) - 1):
        arc.append(arc[-1] + float(np.linalg.norm(curve_pts[k + 1] - curve_pts[k])))
    total_len = arc[-1]

    def point_at_s(s: float) -> np.ndarray:
        """Lineair geïnterpoleerd punt op booglengte s."""
        s = max(0.0, min(s, total_len))
        # Binair zoeken naar het segment
        lo, hi = 0, len(arc) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if arc[mid + 1] < s:
                lo = mid + 1
            else:
                hi = mid
        seg_len = arc[lo + 1] - arc[lo]
        if seg_len < 1e-12:
            return curve_pts[lo].copy()
        t = (s - arc[lo]) / seg_len
        return curve_pts[lo] + t * (curve_pts[lo + 1] - curve_pts[lo])

    placed = []
    s_cursor = 0.0
    for i, v in enumerate(speeds):
        placed.append(point_at_s(s_cursor))
        if i < len(speeds) - 1:
            step = v * WAYPOINT_DT
            s_cursor += step

    return placed


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
    start_pos: Optional[Position] = None,
    start_heading_deg: float = 0.0,
) -> List[Tuple[Position, float]]:
    """
    Berekent het teardrop-parcour rond één boei.

    Waypoints worden geplaatst op afstand v * WAYPOINT_DT van hun voorganger,
    zodat de tijdstap tussen alle waypoints constant is (= WAYPOINT_DT seconden).
    Snelheidsverandering over waypoints is daarmee equivalent aan
    snelheidsverandering over tijd.

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
    n_arc_pts         : waypoints voor de halve cirkel (fallback als WAYPOINT_DT <= 0)
    n_slalom_pts      : waypoints voor elke boog     (fallback als WAYPOINT_DT <= 0)
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
    perp_right = np.array([ axis_u[1], -axis_u[0]])   # 90° CW  (stuurboord)
    perp_left  = np.array([-axis_u[1],  axis_u[0]])   # 90° CCW (bakboord)

    # ── Sleutelposities ───────────────────────────────────────────────────────
    R = buoy_xy + slalom_offset * perp_right   # instappunt halve cirkel
    L = buoy_xy + slalom_offset * perp_left    # uitstappunt halve cirkel

    # ── Hermite-spline helper ─────────────────────────────────────────────────
    def hermite_arc(P0: np.ndarray, T0_uv: np.ndarray,
                    P1: np.ndarray, T1_uv: np.ndarray,
                    n: int) -> List[np.ndarray]:
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

    # Tangent-richtingen aan R en L (van de CCW-cirkel)
    tangent_at_R = axis_u    # CCW-tangent bij R: vooruit langs de as
    tangent_at_L = -axis_u   # CCW-tangent bij L: terug langs de as

    # ── Dichte referentiecurves (voor booglengte-lookup) ──────────────────────
    # We gebruiken een hoog aantal punten zodat _place_waypoints_by_time
    # nauwkeurig kan interpoleren, ongeacht de uiteindelijke waypoint-dichtheid.
    N_DENSE = 500

    in_arc_dense   = hermite_arc(start_xy, heading_uv,
                                 R,        tangent_at_R,
                                 N_DENSE)

    angle_R        = math.atan2(perp_right[1], perp_right[0])
    half_dense     = [
        buoy_xy + slalom_offset * np.array([
            math.cos(angle_R + math.pi * k / N_DENSE),
            math.sin(angle_R + math.pi * k / N_DENSE),
        ])
        for k in range(N_DENSE + 1)
    ]

    out_arc_dense  = hermite_arc(L,        tangent_at_L,
                                 start_xy, -heading_uv,
                                 N_DENSE)

    # ── Snelheden bepalen per segment ─────────────────────────────────────────
    F, S = FAST_SPEED, SLOW_SPEED

    # Schat het aantal waypoints per segment op basis van WAYPOINT_DT.
    # De dichte curves zijn al beschikbaar voor booglengte-berekening.
    def arc_length(pts: List[np.ndarray]) -> float:
        return sum(float(np.linalg.norm(pts[k + 1] - pts[k]))
                   for k in range(len(pts) - 1))

    in_len   = arc_length(in_arc_dense)
    half_len = arc_length(half_dense)
    out_len  = arc_length(out_arc_dense)

    if WAYPOINT_DT > 0:
        # Schat het aantal waypoints: gebruik gemiddelde snelheid per segment
        # (F voor in/out, S voor half), inclusief de ramp aan beide uiteinden.
        # We berekenen eerst hoeveel ramp-stappen er zijn, dan de rest op F/S.
        dv = RAMP_ACCELERATION * WAYPOINT_DT

        # Ramp 0 → F (begin inkomende boog)
        n_ramp_0f = max(0, round(F / dv))
        # Ramp F → S (einde inkomende boog / begin halve cirkel)
        n_ramp_fs = max(0, round(abs(F - S) / dv))
        # Ramp S → F (einde halve cirkel / begin terugkerende boog)
        n_ramp_sf = n_ramp_fs
        # Ramp F → 0 (einde terugkerende boog)
        n_ramp_f0 = n_ramp_0f

        # Tijdsduur van elk segment berekend via integratie (v * dt per stap)
        # Benadering: de ramp neemt v_gem * n_ramp * dt meters in beslag.
        # Hier gebruiken we de tijd (n waypoints) als benadering voor het aantal.

        # Inkomende boog: ramp 0→F + cruise F + ramp F→S
        #   Afstand van ramp 0→F: som v_i * dt voor v_i = dv, 2dv, ..., F
        def ramp_dist(v0, v1):
            """Totale afstand van ramp v0 → v1 (excl. eindpunten)."""
            speeds = _ramp(v0, v1)
            return sum(v * WAYPOINT_DT for v in speeds)

        # In-boog: afstand = ramp(0→F) + cruise + ramp(F→S)
        in_ramp_start_d = ramp_dist(0.0, F)
        in_ramp_end_d   = ramp_dist(F, S)
        in_cruise_d     = max(0.0, in_len - in_ramp_start_d - in_ramp_end_d)
        n_in_ramp_start = len(_ramp(0.0, F))
        n_in_ramp_end   = len(_ramp(F, S))
        n_in_cruise     = max(0, round(in_cruise_d / (F * WAYPOINT_DT)))
        n_in            = n_in_ramp_start + n_in_cruise + n_in_ramp_end + 1  # +1 voor startpunt

        # Halve cirkel: constant S
        half_ramp_start_d = ramp_dist(S, S)   # = 0 (geen ramp binnen cirkel)
        n_half = max(2, round(half_len / (S * WAYPOINT_DT)))

        # Out-boog: ramp(S→F) + cruise + ramp(F→0)
        out_ramp_start_d = ramp_dist(S, F)
        out_ramp_end_d   = ramp_dist(F, 0.0)
        out_cruise_d     = max(0.0, out_len - out_ramp_start_d - out_ramp_end_d)
        n_out_ramp_start = len(_ramp(S, F))
        n_out_ramp_end   = len(_ramp(F, 0.0))
        n_out_cruise     = max(0, round(out_cruise_d / (F * WAYPOINT_DT)))
        n_out            = n_out_ramp_start + n_out_cruise + n_out_ramp_end + 1
    else:
        # Fallback naar vaste aantallen (WAYPOINT_DT uitgeschakeld)
        n_in   = n_slalom_pts
        n_half = n_arc_pts
        n_out  = n_slalom_pts

    # ── Snelheidsreeksen per segment ──────────────────────────────────────────
    # Segment 1: inkomende boog  — 0 → F → S
    # Segment 2: halve cirkel    — S (constant)
    # Segment 3: terugkerende boog — S → F → 0

    # Inkomende boog
    in_speeds_body  = [F] * n_in
    in_tagged = _tag_with_ramp(
        [None] * n_in,   # placeholders; posities volgen later
        body_speed=F,
        prev_speed=0.0,
        next_speed=S,
    )
    in_speeds = [s for _, s in in_tagged]

    # Halve cirkel
    half_speeds = [S] * n_half

    # Terugkerende boog
    out_tagged = _tag_with_ramp(
        [None] * n_out,
        body_speed=F,
        prev_speed=S,
        next_speed=0.0,
    )
    out_speeds = [s for _, s in out_tagged]

    # ── Waypoints plaatsen op basis van tijd ──────────────────────────────────
    # Verwijder het grenspunt aan het begin van half en out (geen duplicaat)
    in_pts   = _place_waypoints_by_time(in_arc_dense,  in_speeds)
    half_pts = _place_waypoints_by_time(half_dense,    half_speeds)
    out_pts  = _place_waypoints_by_time(out_arc_dense, out_speeds)

    # Samenvoegen zonder grenspunten te dupliceren
    all_pts    = in_pts    + half_pts[1:] + out_pts[1:]
    all_speeds = in_speeds + half_speeds[1:] + out_speeds[1:]

    # ── Coördinaten terug naar (lon, lat) ────────────────────────────────────
    return [(to_lonlat(wp), speed) for wp, speed in zip(all_pts, all_speeds)]


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math as _m
    import types, sys

    # Stubs
    _comm = types.ModuleType("communication")
    sys.modules["communication"] = _comm
    _mqtt_mod = types.ModuleType("communication.get_mqtt")
    _mqtt_mod.get_boat_position = lambda: {"longitude": 4.35, "latitude": 51.92}
    sys.modules["communication.get_mqtt"] = _mqtt_mod
    # Make importable as `get_mqtt` too (wrapper uses it directly)
    sys.modules["get_mqtt"] = _mqtt_mod

    _cfg = types.ModuleType("config")
    _cfg.N_SLALOM_PTS      = 12
    _cfg.N_ARC_PTS         = 12
    _cfg.FAST_SPEED        = 4.0    # m/s
    _cfg.SLOW_SPEED        = 2.0    # m/s
    _cfg.WAYPOINT_DT       = 0.5    # s  (elke 0.5 s een waypoint)
    _cfg.RAMP_ACCELERATION = 0.5    # m/s²  →  dv = 0.5 * 0.5 = 0.25 m/s per stap
    sys.modules["config"] = _cfg

    # Re-import config constants into module namespace for the smoke-test
    import importlib, padplanning_sprint as _self
    for attr in ("FAST_SPEED", "SLOW_SPEED", "WAYPOINT_DT", "RAMP_ACCELERATION",
                 "N_SLALOM_PTS", "N_ARC_PTS"):
        setattr(_self, attr, getattr(_cfg, attr))
    # Also patch get_mqtt used inside wrapper
    import communication.get_mqtt as _gm
    _gm.get_boat_position = _mqtt_mod.get_boat_position

    R = 6_371_000.0
    lat0, lon0 = 51.92, 4.35
    mpd_lon = _m.cos(_m.radians(lat0)) * _m.pi / 180 * R
    mpd_lat = _m.pi / 180 * R

    def off(dx, dy):
        return (lon0 + dx / mpd_lon, lat0 + dy / mpd_lat)

    buoy  = off(50, 0)    # boei 50 m ten oosten
    bpos  = off(0,  0)    # bootpositie (referentie)
    spos  = off(-30, 0)   # startpositie 30 m ten westen

    pad = padplanning_buoy(
        [buoy], bpos, "START",
        slalom_offset=12.0,
        start_pos=spos,
        start_heading_deg=90.0,
    )

    print(f"Waypoints: {len(pad)}")
    print(f"Eerste WP snelheid : {pad[0][1]:.3f} m/s")
    print(f"Laatste WP snelheid: {pad[-1][1]:.3f} m/s")

    speeds = [s for _, s in pad]
    print(f"Min snelheid: {min(speeds):.3f} m/s")
    print(f"Max snelheid: {max(speeds):.3f} m/s")

    # Controleer dat de afstand tussen opeenvolgende waypoints ≈ v * WAYPOINT_DT
    print("\nEerste 10 waypoints (verwacht afstand ≈ v * WAYPOINT_DT):")
    R_EARTH = 6_371_000.0
    for k in range(min(10, len(pad) - 1)):
        (lon_a, lat_a), v_a = pad[k]
        (lon_b, lat_b), _   = pad[k + 1]
        dx = (lon_b - lon_a) * mpd_lon
        dy = (lat_b - lat_a) * mpd_lat
        dist = _m.hypot(dx, dy)
        expected = v_a * _cfg.WAYPOINT_DT
        print(f"  WP{k:3d}: v={v_a:.3f}  dist={dist:.4f}m  verwacht={expected:.4f}m  "
              f"fout={abs(dist-expected):.4f}m")