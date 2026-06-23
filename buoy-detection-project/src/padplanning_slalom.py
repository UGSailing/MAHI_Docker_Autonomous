"""
padplanning_slalom.py
Slalom-parcour: zigzag naar de verste boei, 180° bocht eromheen, zigzag terug.
"""

import math
import get_mqtt
from typing import List, Tuple

import numpy as np

Position = Tuple[float, float]   # (longitude, latitude)

def padplanning_wrapper(buoy_positions, x, state):
    """
    Wrapper to correctly translate inputs/outputs between the old architecture
    and the new padplanning_slalom function.
    """
    buoys_centers = []
    buoys_max_dist = []
    
    # Earth radius constant matching the rest of the script
    R_EARTH = 6371000.0

    for buoy_pos in buoy_positions:
        # 1. Calculate the average position (center)
        total_lat = sum(coord[0] for coord in buoy_pos)
        total_lon = sum(coord[1] for coord in buoy_pos)
        num_points = len(buoy_pos)
        
        center_lat = total_lat / num_points
        center_lon = total_lon / num_points
        
        # Sander's slalom code expects (lon, lat) tuples
        buoys_centers.append((center_lon, center_lat))
        
        # 2. Find the maximum distance from this center in METERS
        lat_rad = math.radians(center_lat)
        max_d_meters = 0.0
        
        for lat, lon in buoy_pos:
            # Calculate delta in radians
            delta_lat = math.radians(lat - center_lat)
            delta_lon = math.radians(lon - center_lon)
            
            # Convert to local North/East meter offsets
            delta_north = delta_lat * R_EARTH
            delta_east = delta_lon * R_EARTH * math.cos(lat_rad)
            
            # Distance in meters
            distance_meters = math.hypot(delta_east, delta_north)
            
            if distance_meters > max_d_meters:
                max_d_meters = distance_meters
                
        buoys_max_dist.append(max_d_meters)

    # Fetch current vehicle telemetry
    lat, lon, heading, *_ = get_mqtt.get_boat_position()
    boat_pos = (lon, lat)  # (lon, lat) for the slalom script
    
    # Safely compute offset entirely in meters
    slalom_offset = max(buoys_max_dist) + x

    # Generate the slalom path waypoints: list of (lon, lat)
    slalom_waypoints = padplanning_slalom(
        buoys_centers, heading, boat_pos, state, slalom_offset
    )
    
    # 3. Translate back to original format: list of (lat, lon, speed)
    # Flipped coordinates and hardcoded constant speed = 1
    final_waypoints = []
    for lon_wp, lat_wp in slalom_waypoints:
        final_waypoints.append(((lat_wp, lon_wp), 2))

    return final_waypoints


def padplanning_slalom(
    buoys: List[Position],
    heading_deg: float,
    boat_pos: Position,
    state: str,
    slalom_offset: float = 12.0,   # dwarse afstand (m) — ook straal van 180° bocht
    n_arc_pts: int = 20,           # waypoints voor de 180° bocht om B2
    n_slalom_pts: int = 12,        # waypoints per slalom-been (S-curve)
) -> List[Position]:
    """
    Berekent het slalom-parcour:
      1. Slalom heen  – S-curve langs B1 en B2 (elk een andere kant)
      2. 180° bocht   – om B2 via de kant weg van B1 (halve cirkel)
      3. Slalom terug – S-curve terug langs B2 en B1 (omgekeerde kanten)
      4. Afrit         – voorbij B1 terug naar het startgebied

    Parameters
    ----------
    buoys         : [(lon1, lat1), (lon2, lat2)]
    heading_deg   : kompaskoers boot (0=N, 90=O, kloksgewijs)
    boat_pos      : (lon, lat) van de boot op het moment van berekening
    state         : 'START' | 'DETECT_1' | 'DETECT_2'
    slalom_offset : dwarse slalom-breedte in meter; ook de straal van de 180° bocht
    n_arc_pts     : resolutie van de halve cirkel (n+1 waypoints)
    n_slalom_pts  : resolutie per slalom-been (n+1 waypoints)

    Returns
    -------
    [(lon, lat), ...] – het volledige resterende parcour, startend bij het
    beste instappunt gegeven de huidige positie en koers.
    """
    R_EARTH = 6_371_000.0
    ref_lon, ref_lat = boat_pos
    cos_lat = math.cos(math.radians(ref_lat))
    m_per_deg_lon = cos_lat * math.pi / 180.0 * R_EARTH
    m_per_deg_lat = math.pi / 180.0 * R_EARTH

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

    # ── Lokale coördinaten (boot = oorsprong) ─────────────────────────────────
    # De invoervolgorde bepaalt de identiteit: buoys[0] = boei 1 (dichtbij, eerst
    # gedetecteerd), buoys[1] = boei 2 (ver, gedetecteerd bij DETECT_2).  Zo blijft
    # het manoeuvre identiek over START → DETECT_1 → DETECT_2 en klapt het niet om
    # als de momenteel dichtste boei wisselt tijdens de vaart.
    boat = np.zeros(2)
    pts = [to_xy(*b) for b in buoys]
    b1, b2 = pts[0], pts[1]

    axis = b2 - b1
    d = np.linalg.norm(axis)
    if d < 1e-3:
        raise ValueError("Boeien liggen op dezelfde positie.")
    axis_u = axis / d
    perp_u = np.array([-axis_u[1], axis_u[0]])   # 90° CCW van as

    # ── Slalom-richting: welke kant gaat de boot als eerste langs B1? ─────────
    # sign = +1: perp-kant eerst (noord als as=oost), -1: andere kant
    side = float(np.dot(boat - b1, perp_u))
    sign = 1 if side >= 0 else -1

    # ── S-curve helper ────────────────────────────────────────────────────────
    def s_curve(p_from: np.ndarray, p_to: np.ndarray,
                off_start: float, off_end: float, n: int) -> List[np.ndarray]:
        """
        Cosinus-gevormde S-curve van p_from naar p_to.
        Laterale offset loopt soepel van off_start naar off_end.
        """
        out = []
        for i in range(n + 1):
            t = i / n
            along = p_from + t * (p_to - p_from)
            off = off_start + (off_end - off_start) * (1.0 - math.cos(math.pi * t)) / 2.0
            out.append(along + off * perp_u)
        return out

    # ── Slalom HEEN: B1 (sign-kant) → B2 (tegenovergestelde kant) ────────────
    slalom_fwd = s_curve(b1, b2,
                         off_start =  sign * slalom_offset,
                         off_end   = -sign * slalom_offset,
                         n         = n_slalom_pts)

    # ── 180° bocht om B2 (via axis-kant, weg van B1) ─────────────────────────
    arc_start_angle = math.atan2(-sign * perp_u[1], -sign * perp_u[0])
    arc_dir = sign   # CCW voor sign=+1, CW voor sign=-1
    arc_pts = [
        b2 + slalom_offset * np.array([
            math.cos(arc_start_angle + arc_dir * math.pi * i / n_arc_pts),
            math.sin(arc_start_angle + arc_dir * math.pi * i / n_arc_pts),
        ])
        for i in range(n_arc_pts + 1)
    ]
    # arc_pts[0]  = slalom_fwd[-1]  (aansluiting, wordt niet gedupliceerd)
    # arc_pts[-1] = b2 + sign*offset*perp_u

    # ── Slalom TERUG: B2 (sign-kant) → B1 (tegenovergestelde kant) ───────────
    slalom_ret = s_curve(b2, b1,
                         off_start =  sign * slalom_offset,
                         off_end   = -sign * slalom_offset,
                         n         = n_slalom_pts)
    # slalom_ret[0]  = arc_pts[-1]  (aansluiting, wordt niet gedupliceerd)

    # ── Afrit: korte rechte uitloop voorbij B1 ───────────────────────────────
    # Vaste lengte (onafhankelijk van waar de boot nu staat) en fijn bemonsterd,
    # zodat de waypoint-dichtheid gelijk blijft aan de rest van het pad.
    exit_len = 1.5 * slalom_offset
    p_gate = slalom_ret[-1]                       # laatste slalom-punt bij B1
    p_finish = b1 - exit_len * axis_u
    n_exit = 4
    afrit = [p_gate + (p_finish - p_gate) * (j / n_exit)
             for j in range(1, n_exit + 1)]

    # ── Volledig pad samenstellen (geen dubbele overgangspunten) ──────────────
    waypoints: List[np.ndarray] = (
        slalom_fwd           # [0 .. n_slalom_pts]
        + arc_pts[1:]        # [1 .. n_arc_pts]     (arc_pts[0] == slalom_fwd[-1])
        + slalom_ret[1:]     # [1 .. n_slalom_pts]  (slalom_ret[0] == arc_pts[-1])
        + afrit              # [1 .. n_exit]        (afrit[0]-startpunt == slalom_ret[-1])
    )

    # ── Beste instappunt (afstand + koersuitlijning + voortgangsbias) ─────────
    # De voortgangsbias (0.5 m per waypoint) zorgt dat de boot het parcours in
    # volgorde aflegt: zonder die term kan het scorer-minimum op het afrit-been
    # vallen, dat eindigt vlak bij het startgebied.
    def score(i: int, wp: np.ndarray) -> float:
        delta = wp - boat
        dist = float(np.linalg.norm(delta))
        if dist < 1e-3:
            return float(i) * 0.5
        bearing = math.degrees(math.atan2(delta[0], delta[1]))
        diff = abs(((bearing - heading_deg + 180.0) % 360.0) - 180.0)
        return dist + 2.0 * diff + 0.5 * i

    best_i = min(range(len(waypoints)), key=lambda i: score(i, waypoints[i]))
    return [to_lonlat(wp) for wp in waypoints[best_i:]]


# ── Snelle test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    R = 6_371_000.0
    lat0, lon0 = 51.92, 4.35
    import math as _m
    mpd_lon = _m.cos(_m.radians(lat0)) * _m.pi / 180 * R
    mpd_lat = _m.pi / 180 * R
    def off(dx, dy): return (lon0 + dx / mpd_lon, lat0 + dy / mpd_lat)

    ba, bb = off(-25, 0), off(25, 0)
    for state, bpos, hdg in [
        ("START",    off(-68, 4), 88),
        ("DETECT_1", off(-48, 4), 88),
        ("DETECT_2", off(  5,-14), 55),
    ]:
        pad = padplanning_slalom([ba, bb], hdg, bpos, state)
        print(f"{state}: {len(pad)} waypoints")
