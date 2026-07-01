import math

from .config import APRIORI_BUOYLIST, TILT
from .communication import post_mqtt


buoys = []

for buoy_group in APRIORI_BUOYLIST:
    if buoy_group and buoy_group[0]:
        lat_lon = buoy_group[0]
        if len(lat_lon) >= 2:
            buoy_lat, buoy_lon = lat_lon[:2]
            buoys.append((buoy_lat, buoy_lon))


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


def _to_latlon(ref_lat: float, ref_lon: float, east: float, north: float):
    """
    Inverse of _to_local_en: convert a local East-North offset (metres)
    back to (lat, lon). Same small-angle approximation.
    """
    R = 6_378_137.0
    lat = ref_lat + math.degrees(north / R)
    lon = ref_lon + math.degrees(east / (R * math.cos(math.radians(ref_lat))))
    return lat, lon


def _haversine(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Straight-line distance in metres between two lat/lon points."""
    R = 6_378_137.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _publish_crossline(
    next_ll: tuple[float, float],
    te: float, tn: float,
    re: float, rn: float,
    tilt_signed: float,
    line_half_width_m: float = 20.0,
) -> None:
    """
    Publish the tilted cross line to /navigation/crossline.

    The transition condition is:  y_local > tilt_signed * x_local
    Rearranged, the line itself is:  y_local = tilt_signed * x_local

    A direction vector lying along this line (in local EN coords) is:
        d = right_normal + tilt_signed * tangent
          = (re, rn) + tilt_signed * (te, tn)

    We step ±line_half_width_m along d from next_waypoint to get two
    points that visually span the cross line on a map.

    Publishes: [(lat1, lon1), (lat2, lon2)]
    """
    next_lat, next_lon = next_ll

    # Direction vector along the cross line (not normalised, but that's fine
    # for scaling by line_half_width_m since we normalise explicitly below)
    de = re + tilt_signed * te
    dn = rn + tilt_signed * tn
    d_len = math.hypot(de, dn)
    if d_len < 1e-9:
        # Degenerate (tilt makes the line collapse); fall back to pure right-normal
        de, dn = re, rn
        d_len = 1.0
    de /= d_len
    dn /= d_len

    # Two points ±line_half_width_m along the cross line from next_waypoint
    p1 = _to_latlon(next_lat, next_lon,  de * line_half_width_m,  dn * line_half_width_m)
    p2 = _to_latlon(next_lat, next_lon, -de * line_half_width_m, -dn * line_half_width_m)

    post_mqtt.publish_crossline(p1, p2)


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
        tilt_signed = +abs(tilt)   # buoy right → tilt left
    else:
        tilt_signed = -abs(tilt)   # buoy left (or ahead) → tilt right

    # ------------------------------------------------------------------
    # 3. Publish the cross line for visualisation / debugging
    # ------------------------------------------------------------------
    # _publish_crossline(next_ll, te, tn, re, rn, tilt_signed)

    # ------------------------------------------------------------------
    # 4. Project boat position into path frame (origin = next_waypoint)
    # ------------------------------------------------------------------
    be, bn = _to_local_en(
        next_ll[0], next_ll[1],
        boat_pos['latitude'], boat_pos['longitude'],
    )
    y_local = te * be + tn * bn    # along-path  (positive = ahead of waypoint)
    x_local = re * be + rn * bn    # cross-path  (positive = right of path)

    # ------------------------------------------------------------------
    # 5. Tilted transition line: y > tilt_signed * x
    # ------------------------------------------------------------------
    return y_local > tilt_signed * x_local