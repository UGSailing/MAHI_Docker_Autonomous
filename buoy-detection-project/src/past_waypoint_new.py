import math

from config import APRIORI_BUOYLIST
from config import TILT


buoy0_lat, buoy0_lon = APRIORI_BUOYLIST[0][0]
buoy1_lat, buoy1_lon = APRIORI_BUOYLIST[1][0]
buoys = [(buoy0_lat, buoy0_lon), (buoy1_lat, buoy1_lon)]


def _to_local_en(ref_lat: float, ref_lon: float, lat: float, lon: float):
    """
    Project (lat, lon) into a local East-North frame (metres) centred at (ref_lat, ref_lon).
    Small-angle approximation; accurate to <1 mm within a few km.
    Returns (east, north) in metres.
    """
    R = 6_378_137.0
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    north = R * dlat
    east  = R * math.cos(math.radians(ref_lat)) * dlon
    return east, north


def _haversine(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Straight-line distance in metres between two lat/lon points."""
    R = 6_378_137.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def is_past_waypoint(
    prev_waypoint,
    next_waypoint,
    boat_pos,
    tilt: float = TILT,
) -> bool:
    """
    Returns True when the boat has passed next_waypoint.

    The transition line is tilted so that the boat must travel further
    on the buoy side before the waypoint is considered crossed — reducing
    the risk of cutting too close to a buoy.

    In path-aligned coordinates (y = along path, x = right of path):
        pass when  y_local > tilt_signed * x_local

    The buoy closest to next_waypoint determines the tilt direction:
        - buoy to the LEFT  of the path → tilt_signed = +|tilt|
          (boat must go further before crossing on the left)
        - buoy to the RIGHT of the path → tilt_signed = -|tilt|
          (boat must go further before crossing on the right)

    Args:
        prev_waypoint : ((lat, lon), speed)
        next_waypoint : ((lat, lon), speed)
        boat_pos      : dict with keys 'latitude', 'longitude'
        tilt          : magnitude of the tilt (default from config)
                        0.0 = original perpendicular behaviour
    """
    prev_ll = prev_waypoint[0]   # (lat, lon)
    next_ll = next_waypoint[0]   # (lat, lon)

    # ------------------------------------------------------------------
    # 1. Build the path-aligned coordinate frame (metres, origin = next_waypoint)
    # ------------------------------------------------------------------
    pe, pn = _to_local_en(prev_ll[0], prev_ll[1], next_ll[0], next_ll[1])

    path_len = math.hypot(pe, pn)
    if path_len < 1e-6:
        # prev and next are the same point; consider it passed
        return True

    # Unit tangent: points forward along the path (prev → next)
    te, tn = pe / path_len, pn / path_len

    # Unit right-normal: 90° clockwise from tangent → points RIGHT of path
    # Rotation -90°: (e, n) → (n, -e)
    re, rn = tn, -te

    # ------------------------------------------------------------------
    # 2. Determine tilt direction from the nearest buoy
    # ------------------------------------------------------------------
    next_lat, next_lon = next_ll
    closest_buoy = min(
        buoys,
        key=lambda b: _haversine(next_lat, b[0], next_lon, b[1])
    )

    # Project the closest buoy into the path frame (origin = next_waypoint)
    be_buoy, bn_buoy = _to_local_en(next_lat, next_lon, closest_buoy[0], closest_buoy[1])
    buoy_x = re * be_buoy + rn * bn_buoy   # positive = buoy is RIGHT of path

    if buoy_x > 0:
        tilt_signed = -abs(tilt)   # buoy right → tilt left
    else:
        tilt_signed = +abs(tilt)   # buoy left (or ahead) → tilt right

    # ------------------------------------------------------------------
    # 3. Project boat position into path frame (origin = next_waypoint)
    # ------------------------------------------------------------------
    be, bn = _to_local_en(
        next_ll[0], next_ll[1],
        boat_pos['latitude'], boat_pos['longitude'],
    )
    y_local = te * be + tn * bn    # along-path  (positive = ahead of waypoint)
    x_local = re * be + rn * bn    # cross-path  (positive = right of path)

    # ------------------------------------------------------------------
    # 4. Tilted transition line: y > tilt_signed * x
    # ------------------------------------------------------------------
    return y_local > tilt_signed * x_local