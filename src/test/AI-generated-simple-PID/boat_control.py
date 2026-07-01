"""
boat_control.py
Figure-8 autonomous navigation around two buoys.
Architecture: lookahead guidance (Pure Pursuit) → PD heading controller.
Output: normalized rudder [-1, 1] and constant throttle.
"""

import math

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
BUOY_A = (51.200000, 4.400000)   # (lat, lon) of first buoy
BUOY_B = (51.200450, 4.400000)   # (lat, lon) of second buoy  (~50 m north)

WAYPOINTS_PER_LOBE = 8           # resolution of each lobe
LOOKAHEAD_M        = 8.0         # lookahead circle radius in metres
MAX_RUDDER_DEG     = 30.0        # physical rudder limit in degrees
THROTTLE           = 0.5         # constant normalised throttle [0, 1]

# PD gains  (error in degrees, output normalised to [-1, 1])
KP = 0.015
KD = 0.030

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
EARTH_R = 6_371_000.0  # metres

def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    """Convert (lat, lon) to local ENU (x east, y north) in metres."""
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    x = dlon * EARTH_R * math.cos(math.radians(origin_lat))
    y = dlat * EARTH_R
    return x, y

def bearing_deg(from_xy, to_xy):
    """Compass bearing in degrees [0, 360) from one XY point to another."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    angle = math.degrees(math.atan2(dx, dy))   # atan2(east, north) = bearing
    return angle % 360.0

def wrap_180(angle):
    """Wrap an angle to (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# ---------------------------------------------------------------------------
# Path generation: figure-8 around two buoys
# ---------------------------------------------------------------------------
def _lobe_waypoints(centre, radius, n, start_angle_deg, clockwise):
    """Return n XY points on a circle (one lobe of the figure-8)."""
    pts = []
    step = (360.0 / n) * (1 if clockwise else -1)
    for i in range(n):
        a = math.radians(start_angle_deg + i * step)
        pts.append((centre[0] + radius * math.sin(a),
                    centre[1] + radius * math.cos(a)))
    return pts

def build_path(origin_lat, origin_lon):
    """
    Build a figure-8 polyline in local XY.
    Lobe A goes clockwise, lobe B counter-clockwise so they join smoothly.
    Returns list of (x, y) tuples.
    """
    ax, ay = latlon_to_xy(BUOY_A[0], BUOY_A[1], origin_lat, origin_lon)
    bx, by = latlon_to_xy(BUOY_B[0], BUOY_B[1], origin_lat, origin_lon)

    # Lobe radius = half the inter-buoy distance so lobes just meet in the middle
    lobe_r = distance((ax, ay), (bx, by)) / 2.0

    mid_x = (ax + bx) / 2.0
    mid_y = (ay + by) / 2.0

    # Lobe A: clockwise, start heading south from midpoint (180°)
    lobe_a = _lobe_waypoints((ax, ay), lobe_r, WAYPOINTS_PER_LOBE, 180, clockwise=True)
    # Lobe B: counter-clockwise, start heading north from midpoint (0°)
    lobe_b = _lobe_waypoints((bx, by), lobe_r, WAYPOINTS_PER_LOBE, 0,   clockwise=False)

    return lobe_a + lobe_b   # closed loop: end of B lands back at start of A

# ---------------------------------------------------------------------------
# Lookahead guidance
# ---------------------------------------------------------------------------
def _segment_circle_intersect(p1, p2, centre, radius):
    """
    Return the intersection point on segment p1→p2 that is closest to p2
    (i.e. the 'forward' one) and lies on the circle, or None.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - centre[0]
    fy = p1[1] - centre[1]

    a = dx*dx + dy*dy
    if a < 1e-12:
        return None
    b = 2.0 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - radius*radius
    disc = b*b - 4*a*c
    if disc < 0:
        return None

    sq = math.sqrt(disc)
    best = None
    for sign in (1, -1):
        t = (-b + sign * sq) / (2.0 * a)
        if 0.0 <= t <= 1.0:
            pt = (p1[0] + t*dx, p1[1] + t*dy)
            if best is None or t > best[1]:
                best = (pt, t)
    return best[0] if best else None

def lookahead_target(boat_xy, path, lookahead):
    """
    Walk the path from the nearest segment forward and return the first
    intersection with the lookahead circle.  Falls back to the nearest
    point if no intersection is found.
    """
    n = len(path)

    # Find nearest segment start index
    nearest_idx = min(range(n), key=lambda i: distance(boat_xy, path[i]))

    # Search forward through segments (wrap around, full loop)
    for offset in range(n):
        i  = (nearest_idx + offset) % n
        j  = (i + 1) % n
        pt = _segment_circle_intersect(path[i], path[j], boat_xy, lookahead)
        if pt:
            return pt

    # Fallback: nearest point on path
    return path[nearest_idx]

# ---------------------------------------------------------------------------
# PD heading controller
# ---------------------------------------------------------------------------
class HeadingController:
    def __init__(self):
        self._prev_error = 0.0

    def compute(self, desired_heading_deg, current_heading_deg, dt):
        """
        Returns normalised rudder command in [-1, 1].
        Positive = steer right.
        """
        error = wrap_180(desired_heading_deg - current_heading_deg)
        d_error = wrap_180(error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        raw = KP * error + KD * d_error
        return max(-1.0, min(1.0, raw))   # clamp

# ---------------------------------------------------------------------------
# Top-level step function
# ---------------------------------------------------------------------------
def create_controller():
    """Return a fresh HeadingController.  Call once before the loop."""
    return HeadingController()

def navigation_step(controller, boat_lat, boat_lon, boat_heading_deg, dt, path):
    """
    Single control cycle.

    Parameters
    ----------
    controller      : HeadingController instance
    boat_lat/lon    : current GNSS position
    boat_heading_deg: current compass heading [0, 360)
    dt              : seconds since last call
    path            : XY waypoint list from build_path()

    Returns
    -------
    rudder_norm     : normalised rudder [-1, 1]  (positive = right)
    throttle        : constant THROTTLE value
    desired_heading : for logging / debugging
    """
    origin_lat, origin_lon = BUOY_A   # arbitrary fixed origin for XY projection

    boat_xy = latlon_to_xy(boat_lat, boat_lon, origin_lat, origin_lon)
    target  = lookahead_target(boat_xy, path, LOOKAHEAD_M)
    desired = bearing_deg(boat_xy, target)

    rudder_norm = controller.compute(desired, boat_heading_deg, dt)

    # Scale to physical degrees for reference (not used internally)
    rudder_deg = rudder_norm * MAX_RUDDER_DEG

    return rudder_norm, THROTTLE, desired
