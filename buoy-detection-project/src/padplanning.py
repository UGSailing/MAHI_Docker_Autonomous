"""
Compute 6 points forming a figure-eight pattern around two buoys A and B,
where each buoy may have a LIST of possible positions (lat, lon) instead of
a single known position. The path is planned to clear all possible
positions of both buoys by at least x meters.

Pure-Python, no external dependencies. Flat-earth (equirectangular)
approximation -- accurate to well under a meter at distances up to a few km,
more than sufficient for ~100 m scale offsets.

--------------------------------------------------------------------------
Local coordinate system (right-handed):
  - Origin = average of A's possible positions.
  - x-axis runs from avg(A) to avg(B) (bearing = bearing_ab).
  - y-axis is perpendicular, +y = 90 deg counterclockwise from +x
    (i.e. to the LEFT of the A->B direction of travel).

Every possible position of A and B is projected into this local frame.
--------------------------------------------------------------------------

General rule per buoy (collapses to the single-position formula when a
buoy only has one possible position):

  "top"     point = (x of the possible-position with max y,  max_y + x)
  "bottom"  point = (x of the possible-position with min y,  min_y - x)
  "outward" point = the possible-position most extreme AWAY from the other
                     buoy, pushed out by x:
                       A: (min_x - x, y of that position)   [away = -x side]
                       B: (max_x + x, y of that position)   [away = +x side]

Path order (figure-eight):
  1. A_top
  2. B_bottom
  3. B_outward
  4. B_top
  5. A_bottom
  6. A_outward
"""

import math

R_EARTH = 6371000.0  # mean Earth radius in meters, fine for this scale


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2)
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Initial bearing (degrees, compass: 0=N, 90=E, clockwise) from point 1 to 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360) % 360


def latlon_to_local(lat0, lon0, bearing_ab_deg, lat, lon):
    """Project (lat, lon) into the local (x, y) frame centered at (lat0, lon0)."""
    lat0_rad = math.radians(lat0)
    north = math.radians(lat - lat0) * R_EARTH
    east = math.radians(lon - lon0) * R_EARTH * math.cos(lat0_rad)

    bearing_ab = math.radians(bearing_ab_deg)
    bearing_y = math.radians(bearing_ab_deg - 90.0)  # left of travel

    x_local = north * math.cos(bearing_ab) + east * math.sin(bearing_ab)
    y_local = north * math.cos(bearing_y) + east * math.sin(bearing_y)
    return x_local, y_local


def local_to_latlon(lat0, lon0, bearing_ab_deg, x_local, y_local, speed):
    """Convert a local (x_local, y_local) offset in meters back into lat/lon."""
    bearing_ab = math.radians(bearing_ab_deg)
    bearing_y = math.radians(bearing_ab_deg - 90.0)  # left of travel

    north = x_local * math.cos(bearing_ab) + y_local * math.cos(bearing_y)
    east = x_local * math.sin(bearing_ab) + y_local * math.sin(bearing_y)

    lat0_rad = math.radians(lat0)
    dlat = north / R_EARTH
    dlon = east / (R_EARTH * math.cos(lat0_rad))

    return (lat0 + math.degrees(dlat), lon0 + math.degrees(dlon)), speed


def average_position(positions):
    """Simple arithmetic mean of a list of (lat, lon) tuples."""
    n = len(positions)
    avg_lat = sum(p[0] for p in positions) / n
    avg_lon = sum(p[1] for p in positions) / n
    return avg_lat, avg_lon


def compute_six_points(positions_a, positions_b, x):
    """
    positions_a, positions_b: lists of (lat, lon) tuples -- the possible
        positions for buoy A and buoy B respectively (length >= 1 each).
    x: clearance distance in meters.

    Returns: (list of 6 (lat, lon) points in order, distance avg(A)->avg(B),
              bearing avg(A)->avg(B))
    """
    avg_a = average_position(positions_a)
    avg_b = average_position(positions_b)

    d = haversine_distance(avg_a[0], avg_a[1], avg_b[0], avg_b[1])
    bearing_ab = initial_bearing(avg_a[0], avg_a[1], avg_b[0], avg_b[1])

    # Project every possible position into the local frame (origin = avg_a)
    a_local = [latlon_to_local(avg_a[0], avg_a[1], bearing_ab, lat, lon)
               for lat, lon in positions_a]
    b_local = [latlon_to_local(avg_a[0], avg_a[1], bearing_ab, lat, lon)
               for lat, lon in positions_b]

    # --- Extremes for A ---
    ax_top, ay_top = max(a_local, key=lambda p: p[1])       # max y
    ax_bot, ay_bot = min(a_local, key=lambda p: p[1])       # min y
    ax_out, ay_out = min(a_local, key=lambda p: p[0])       # min x (away from B)

    # --- Extremes for B ---
    bx_top, by_top = max(b_local, key=lambda p: p[1])       # max y
    bx_bot, by_bot = min(b_local, key=lambda p: p[1])       # min y
    bx_out, by_out = max(b_local, key=lambda p: p[0])       # max x (away from A)

    local_points = [
        (ax_top, ay_top + x, 2),   # 1. A top
        (bx_bot, by_bot - x, 2),   # 2. B bottom
        (bx_out + x, by_out, 1),   # 3. B outward
        (bx_top, by_top + x, 1),   # 4. B top
        (ax_bot, ay_bot - x, 2),   # 5. A bottom
        (ax_out - x, ay_out, 1),   # 6. A outward
    ]

    results = [local_to_latlon(avg_a[0], avg_a[1], bearing_ab, xl, yl, speed)
               for xl, yl, speed in local_points]

    return results, d, bearing_ab


def padplanning_8(buoy_positions,marge, state):
    positions_a, positions_b = buoy_positions

    points, d, bearing = compute_six_points(positions_a, positions_b, marge)



    if state == 'START':
        return points[1:]
    elif state == 'DETECT_1':
        return points
    else: # state == 'DETECT_2':
        return points[1:]
