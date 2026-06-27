"""
padplanning_slalom.py
Slalom-parcour: zigzag naar de verste boei, 180° bocht eromheen, zigzag terug.

Changes vs. previous version
─────────────────────────────
1. Waypoint spacing can be driven by WAYPOINT_DISTANCE (meters) instead of a
   fixed point count.  Controlled by the INTERPOLATE_USING_DISTANCE bool in
   config.py.  Both approaches remain available.

2. Full approach/exit geometry around B1:
   • oprit  – quarter-circle (90°) around B1 from the entry point to slalom start
   • approach – arc from start_position to the oprit entry, bent to match the
                 boat's initial heading
   • afrit  – quarter-circle (90°) around B1 from slalom end to exit point
              (replaces the old straight exit line)

3. Speed ramping: N_RAMP_WAYPOINTS linearly-interpolated intermediate speeds
   are inserted at every slow↔fast transition.
"""

import math
import get_mqtt
from typing import List, Optional, Tuple

import numpy as np

from config import (
    N_SLALOM_PTS,
    N_ARC_PTS,
    FAST_SPEED,
    SLOW_SPEED,
    WAYPOINT_DISTANCE,
    INTERPOLATE_USING_DISTANCE,
    N_RAMP_WAYPOINTS,
)

Position = Tuple[float, float]   # (longitude, latitude)


# ─────────────────────────────────────────────────────────────────────────────
#  Speed-ramp helper
# ─────────────────────────────────────────────────────────────────────────────

def _ramp(from_speed: float, to_speed: float) -> List[float]:
    """
    Return a list of N_RAMP_WAYPOINTS intermediate speeds between from_speed
    and to_speed (exclusive of both endpoints).

    N_RAMP_WAYPOINTS == 0  →  empty list  (hard switch, original behaviour)
    N_RAMP_WAYPOINTS == 1  →  [average]
    N_RAMP_WAYPOINTS == n  →  n linearly interpolated values
    """
    if N_RAMP_WAYPOINTS <= 0:
        return []
    return [
        from_speed + (to_speed - from_speed) * (i + 1) / (N_RAMP_WAYPOINTS + 1)
        for i in range(N_RAMP_WAYPOINTS)
    ]


def _tag_with_ramp(
    waypoints: List[np.ndarray],
    body_speed: float,
    prev_speed: Optional[float],   # speed of the last waypoint before this segment
    next_speed: Optional[float],   # speed of the first waypoint after this segment
) -> List[Tuple[np.ndarray, float]]:
    """
    Tag a list of waypoints with speeds, adding ramp-up at the start and
    ramp-down at the end when the adjacent segment has a different speed.

    The ramp consumes the first / last N_RAMP_WAYPOINTS points of the segment.
    If the segment is too short to fit both ramps, the ramp is clipped.
    """
    n = len(waypoints)
    speeds = [body_speed] * n

    # Ramp up (start of segment)
    if prev_speed is not None and prev_speed != body_speed:
        ramp_up = _ramp(prev_speed, body_speed)   # ascending toward body_speed
        for j, s in enumerate(ramp_up):
            if j < n:
                speeds[j] = s

    # Ramp down (end of segment)
    if next_speed is not None and next_speed != body_speed:
        ramp_dn = _ramp(body_speed, next_speed)   # descending toward next_speed
        for j, s in enumerate(reversed(ramp_dn)):
            idx = n - 1 - j
            if idx >= 0:
                # Only overwrite if this cell wasn't already claimed by ramp-up
                if speeds[idx] == body_speed:
                    speeds[idx] = s

    return list(zip(waypoints, speeds))


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def padplanning_wrapper(buoy_positions, marge, state='START', start_position=None):
    """
    Wrapper to correctly translate inputs/outputs between the old architecture
    and padplanning_slalom.

    Parameters
    ----------
    buoy_positions  : list of buoy point-clouds, each a list of (lat, lon) pairs
    marge           : safety margin added to the largest buoy radius (meters)
    state           : 'START' | 'DETECT_1' | 'DETECT_2' | 'DETECT_3'
    start_position  : (lat, lon) of the boat's starting position, or None to use
                      the current MQTT position as a fallback
    """
    buoys_centers = []
    buoys_max_dist = []

    R_EARTH = 6_371_000.0
    

    for buoy_pos in buoy_positions:
        total_lat = sum(coord[0] for coord in buoy_pos)
        total_lon = sum(coord[1] for coord in buoy_pos)
        num_points = len(buoy_pos)

        center_lat = total_lat / num_points
        center_lon = total_lon / num_points
        buoys_centers.append((center_lon, center_lat))   # (lon, lat) for slalom

        lat_rad = math.radians(center_lat)
        max_d_meters = 0.0
        for lat, lon in buoy_pos:
            delta_lat = math.radians(lat - center_lat)
            delta_lon = math.radians(lon - center_lon)
            delta_north = delta_lat * R_EARTH
            delta_east  = delta_lon * R_EARTH * math.cos(lat_rad)
            distance_meters = math.hypot(delta_east, delta_north)
            if distance_meters > max_d_meters:
                max_d_meters = distance_meters
        buoys_max_dist.append(max_d_meters)

    # Current telemetry
    boat_position = get_mqtt.get_boat_position()
    boat_pos = (boat_position['longitude'], boat_position['latitude'])
    heading  = boat_position['heading']

    slalom_offset = max(buoys_max_dist) + marge

    # Resolve start_position: convert (lat, lon) → (lon, lat)
    if start_position is not None:
        start_pos_lonlat = (start_position[1], start_position[0])
    else:
        start_pos_lonlat = boat_pos   # fall back to current position

    slalom_waypoints = padplanning_slalom(
        buoys_centers,
        heading,
        boat_pos,
        state,
        slalom_offset,
        start_pos=start_pos_lonlat,
        start_heading_deg=heading,
    )

    # Translate back: (lon, lat) → (lat, lon), keep speed
    return [((lat_wp, lon_wp), speed) for (lon_wp, lat_wp), speed in slalom_waypoints]


# ─────────────────────────────────────────────────────────────────────────────
#  Core planning function
# ─────────────────────────────────────────────────────────────────────────────

def padplanning_slalom(
    buoys: List[Position],
    heading_deg: float,
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
    Berekent het volledige slalom-parcour inclusief aanrijroute en afrit.

    Volledige padopbouw
    ───────────────────
      0. Approach  – boog van start_pos naar oprit-ingang (hoge snelheid)
      1. Oprit     – 90° kwartcirkel om B1 van ingang naar slalom-start (laag)
      2. Slalom heen  – S-curve langs B1 en B2 (hoog)
      3. 180° bocht   – om B2 (laag)
      4. Slalom terug – S-curve terug langs B2 en B1 (hoog)
      5. Afrit     – 90° kwartcirkel om B1 van slalom-einde naar uitgang (laag)

    Parameters
    ----------
    buoys             : [(lon1, lat1), (lon2, lat2)]
    heading_deg       : kompaskoers boot (0=N, 90=O, kloksgewijs)
    boat_pos          : (lon, lat) huidige positie boot
    state             : 'START' | 'DETECT_1' | 'DETECT_2'
    slalom_offset     : dwarse breedte én bochtenstraal (m)
    n_arc_pts         : waypoints voor de 180° bocht (gebruikt als
                        INTERPOLATE_USING_DISTANCE=False)
    n_slalom_pts      : waypoints per slalom-been (idem)
    waypoint_distance : gewenste afstand tussen waypoints in m (gebruikt als
                        INTERPOLATE_USING_DISTANCE=True)
    start_pos         : (lon, lat) startpositie voor de aanrijroute;
                        None → boot-positie wordt gebruikt
    start_heading_deg : initiële koers van de boot bij start_pos (graden)

    Returns
    -------
    [((lon, lat), speed), ...]
    """

    R_EARTH = 6_371_000.0
    ref_lon, ref_lat = boat_pos
    cos_lat    = math.cos(math.radians(ref_lat))
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

    # ── Local coordinates (boat = origin) ────────────────────────────────────
    boat  = np.zeros(2)
    pts   = [to_xy(*b) for b in buoys]
    b1, b2 = pts[0], pts[1]

    axis   = b2 - b1
    d      = np.linalg.norm(axis)
    if d < 1e-3:
        raise ValueError("Boeien liggen op dezelfde positie.")
    axis_u = axis / d
    perp_u = np.array([-axis_u[1], axis_u[0]])   # 90° CCW from axis

    sign = 1   # which side of the axis the boat approaches B1 first

    # ── Resolve waypoint counts ───────────────────────────────────────────────
    if INTERPOLATE_USING_DISTANCE:
        # arc length of the 180° half-circle around B2
        half_circ = math.pi * slalom_offset
        n_arc_pts    = max(2, round(half_circ    / waypoint_distance))

        # slalom leg length ≈ distance between buoys
        n_slalom_pts = max(2, round(d            / waypoint_distance))

        # quarter-circle (oprit / afrit) around B1
        qtr_circ = 0.5 * math.pi * slalom_offset
        n_quarter_pts = max(2, round(qtr_circ / waypoint_distance))
    else:
        # Use the fixed counts from config / function arguments
        n_quarter_pts = max(2, n_arc_pts // 2)

    # ── Key geometry points ───────────────────────────────────────────────────
    # A = slalom start  (first point of slalom_fwd, on +sign side of B1)
    # C = slalom end    (last  point of slalom_ret, on -sign side of B1)
    # E = entry/exit    (behind B1, away from B2)

    A = b1 + sign        * slalom_offset * perp_u
    C = b1 - sign        * slalom_offset * perp_u
    E = b1 -              slalom_offset  * axis_u   # quarter-circle meeting point

    # ── S-curve helper ────────────────────────────────────────────────────────
    def s_curve(p_from: np.ndarray, p_to: np.ndarray,
                off_start: float, off_end: float, n: int) -> List[np.ndarray]:
        """Cosinus-shaped S-curve from p_from to p_to with lateral offsets."""
        out = []
        for i in range(n + 1):
            t   = i / n
            along = p_from + t * (p_to - p_from)
            off   = off_start + (off_end - off_start) * (1.0 - math.cos(math.pi * t)) / 2.0
            out.append(along + off * perp_u)
        return out

    # ── Quarter-circle arc helper ─────────────────────────────────────────────
    def quarter_arc(center: np.ndarray,
                    start_angle_rad: float,
                    sweep_rad: float,
                    radius: float,
                    n: int,
                    include_start: bool = True) -> List[np.ndarray]:
        """
        Discretised arc around `center` starting at `start_angle_rad`,
        sweeping `sweep_rad` radians (positive = CCW).
        Returns n+1 points (or n points if include_start=False).
        """
        out = []
        i0 = 0 if include_start else 1
        for i in range(i0, n + 1):
            theta = start_angle_rad + sweep_rad * i / n
            out.append(center + radius * np.array([math.cos(theta), math.sin(theta)]))
        return out

    # ── Oprit: 90° arc from E → A around B1 ──────────────────────────────────
    # E is at angle  atan2(-axis_u) relative to B1
    # A is at angle  atan2(sign * perp_u) relative to B1
    angle_E = math.atan2(-axis_u[1], -axis_u[0])
    angle_A = math.atan2((sign * perp_u)[1], (sign * perp_u)[0])

    # Determine sweep direction: shortest path from angle_E to angle_A
    sweep_oprit = angle_A - angle_E
    # Normalise to (-π, π]
    sweep_oprit = (sweep_oprit + math.pi) % (2 * math.pi) - math.pi

    oprit_pts = quarter_arc(
        center           = b1,
        start_angle_rad  = angle_E,
        sweep_rad        = sweep_oprit,
        radius           = slalom_offset,
        n                = n_quarter_pts,
        include_start    = True,
    )

    # ── Approach: Hermite arc from start_pos to E ────────────────────────────
    # The curve is tangent to the boat's heading at start_xy and tangent to
    # -axis_u (arriving into E from behind B1) at E.  We use a cubic Hermite
    # spline: p(t) = h00·P0 + h10·T0 + h01·P1 + h11·T1, where the tangent
    # magnitudes are scaled to approach_d so the bend is proportional to the
    # distance (standard Catmull-Rom / Hermite convention).
    if start_pos is not None:
        start_xy = to_xy(*start_pos)
    else:
        start_xy = boat.copy()

    # Compass heading → local XY unit vector (x=East, y=North)
    hdg_rad    = math.radians(start_heading_deg)
    heading_uv = np.array([math.sin(hdg_rad), math.cos(hdg_rad)])

    # Arrival tangent at E: perpendicular to the radius (E - B1) = -axis_u,
    # so tangent = sign * perp_u (direction in which the oprit arc departs from E)
    arrival_uv = sign * perp_u

    approach_vec = E - start_xy
    approach_d   = np.linalg.norm(approach_vec)

    if approach_d < 1e-3:
        approach_pts = []
    else:
        if INTERPOLATE_USING_DISTANCE:
            n_approach = max(2, round(approach_d / waypoint_distance))
        else:
            n_approach = max(2, round(approach_d / (d / n_slalom_pts)))

        # Both tangents scaled larger than approach_d so the curve commits to
        # each end direction early and blends smoothly rather than bending sharply
        # near the endpoints when the heading differs from the direct line.
        T0 = heading_uv * approach_d * 2.0
        T1 = arrival_uv * approach_d * 2.0

        P0, P1 = start_xy, E

        approach_pts = []
        for i in range(n_approach + 1):
            t  = i / n_approach
            t2 = t * t
            t3 = t2 * t
            # Cubic Hermite basis functions
            h00 =  2*t3 - 3*t2 + 1
            h10 =    t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 =    t3 -   t2
            p = h00*P0 + h10*T0 + h01*P1 + h11*T1
            approach_pts.append(p)

    # ── Slalom HEEN: B1 (sign-kant) → B2 (tegenovergestelde kant) ────────────
    slalom_fwd = s_curve(b1, b2,
                         off_start =  sign * slalom_offset,
                         off_end   = -sign * slalom_offset,
                         n         = n_slalom_pts)

    # ── 180° bocht om B2 ─────────────────────────────────────────────────────
    arc_start_angle = math.atan2(-sign * perp_u[1], -sign * perp_u[0])
    arc_dir  = sign
    arc_pts  = [
        b2 + slalom_offset * np.array([
            math.cos(arc_start_angle + arc_dir * math.pi * i / n_arc_pts),
            math.sin(arc_start_angle + arc_dir * math.pi * i / n_arc_pts),
        ])
        for i in range(n_arc_pts + 1)
    ]

    # ── Slalom TERUG: B2 → B1 ────────────────────────────────────────────────
    slalom_ret = s_curve(b2, b1,
                         off_start =  sign * slalom_offset,
                         off_end   = -sign * slalom_offset,
                         n         = n_slalom_pts)

    # ── Afrit: 90° arc from C → E around B1 ──────────────────────────────────
    angle_C = math.atan2((- sign * perp_u)[1], (-sign * perp_u)[0])

    sweep_afrit = angle_E - angle_C
    sweep_afrit = (sweep_afrit + math.pi) % (2 * math.pi) - math.pi

    afrit_pts = quarter_arc(
        center          = b1,
        start_angle_rad = angle_C,
        sweep_rad       = sweep_afrit,
        radius          = slalom_offset,
        n               = n_quarter_pts,
        include_start   = True,    # dedup handled by uniform [1:] strip in assembly
    )

    # ── Assemble segments with speed ramping ──────────────────────────────────
    #
    # Segment speeds (body speed):
    #   approach   → FAST
    #   oprit      → SLOW
    #   slalom_fwd → FAST
    #   arc (180°) → SLOW
    #   slalom_ret → FAST
    #   afrit      → SLOW
    #
    # Joining rules (to avoid duplicate waypoints at segment boundaries):
    #   • approach ends at E; oprit starts at E          → oprit[0] is duplicate
    #   • oprit ends at A; slalom_fwd starts at A        → slalom_fwd[0] is duplicate
    #   • slalom_fwd ends at slalom_fwd[-1];
    #     arc_pts[0] == slalom_fwd[-1]                   → use arc_pts[1:]
    #   • arc_pts[-1] == slalom_ret[0]                   → use slalom_ret[1:]
    #   • slalom_ret[-1] == C; afrit[0] == C             → stripped by uniform rule
    #
    # We normalise this by always stripping the first point of every segment
    # after the first (uniform [1:] strip), so all joins are handled identically.

    F, S = FAST_SPEED, SLOW_SPEED

    # Raw segments in order, each as (points, body_speed).
    # Points are the geometrically correct full lists; we strip duplicates below.
    raw_segments: List[Tuple[List[np.ndarray], float]] = []
    if approach_pts:
        raw_segments.append((approach_pts, F))
    raw_segments += [
        (oprit_pts,      S),
        (slalom_fwd,     F),
        (arc_pts,        S),
        (slalom_ret,     F),
        (afrit_pts,      S),
    ]

    # Strip the first point of every segment after the first to remove the
    # duplicate at each geometric join.
    segments: List[Tuple[List[np.ndarray], float]] = []
    for si, (pts_seg, spd) in enumerate(raw_segments):
        trimmed = pts_seg if si == 0 else pts_seg[1:]
        if trimmed:          # skip degenerate empty segments
            segments.append((trimmed, spd))

    # Build tagged waypoints with ramps.
    # Ramps are applied ONLY on FAST (slalom) segments: they ramp down toward
    # the upcoming slow turn and ramp up from the previous slow turn.
    # All SLOW segments (approach excluded — it stays FAST throughout) keep a
    # flat speed so there is no zigzag at the slow↔fast boundaries.
    all_waypoints: List[Tuple[np.ndarray, float]] = []
    n_segs = len(segments)
    for si, (pts_seg, body_spd) in enumerate(segments):
        if body_spd == F:
            # FAST segment: look at neighbouring speeds for ramp direction
            prev_spd = segments[si - 1][1] if si > 0          else None
            next_spd = segments[si + 1][1] if si < n_segs - 1 else None
            tagged = _tag_with_ramp(pts_seg, body_spd, prev_spd, next_spd)
        else:
            # SLOW segment (oprit, arc, afrit): flat speed, no ramp
            tagged = [(wp, body_spd) for wp in pts_seg]
        all_waypoints.extend(tagged)

    # ── Convert to (lon, lat) ─────────────────────────────────────────────────
    return [(to_lonlat(wp), speed) for wp, speed in all_waypoints]


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math as _m

    R = 6_371_000.0
    lat0, lon0 = 51.92, 4.35
    mpd_lon = _m.cos(_m.radians(lat0)) * _m.pi / 180 * R
    mpd_lat = _m.pi / 180 * R

    def off(dx, dy):
        return (lon0 + dx / mpd_lon, lat0 + dy / mpd_lat)

    ba, bb = off(-25, 0), off(25, 0)

    for state, bpos, hdg, spos in [
        ("START",    off(-68,  4), 88, off(-100,  4)),
        ("DETECT_1", off(-48,  4), 88, off(-100,  4)),
        ("DETECT_2", off(  5, -14), 55, off(-100,  4)),
    ]:
        pad = padplanning_slalom(
            [ba, bb], hdg, bpos, state,
            start_pos=spos, start_heading_deg=hdg
        )
        print(f"{state}: {len(pad)} waypoints, "
              f"speeds: {list(dict.fromkeys(round(s,1) for _,s in pad))}")