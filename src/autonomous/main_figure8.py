"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras in a
background thread, then runs the mission loop (detect → plan → sail).
"""

import threading
import time
import random

from . import camera
from . import communication.post_mqtt as post_mqtt
from . import communication.get_mqtt as get_mqtt
from .padplanning_slalom import padplanning_wrapper
from .autopilot import set_waypoint, start_navigation, stop_navigation
from .past_waypoint_new import is_past_waypoint
from .config import INDEX_LOOK_AHEAD, MARGE, STATE_TRANS_DIST, APRIORI_BUOYLIST, TEST_PADAANPASSING_ZONDER_BOEIEN, BUOY_MATCH_DISTANCE


# ---------------------------------------------------------------------------
# Haversine helper (distance in metres between two lat/lon points)
# ---------------------------------------------------------------------------

import math



def haversine(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    R = 6_378_137.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def is_past_waypoint_old(prev_waypoint, next_waypoint, boat_pos):
    """
    Returns False if boat_pos is on the prev_waypoint-side of the perpendicular line through next_waypoint,
    Returns True if boat_pos is past next_waypoint (on the far side).

    for prev and next waypoint this is ((lat,lon),speed), boat_pos (lat,lon, ...)
    """
    prev_waypoint = prev_waypoint[0]
    next_waypoint = next_waypoint[0]

    dx = next_waypoint[0] - prev_waypoint[0]
    dy = next_waypoint[1] - prev_waypoint[1]

    if dx == 0 and dy == 0:
        return True

    bpx = boat_pos['latitude'] - next_waypoint[0]
    bpy = boat_pos['longitude'] - next_waypoint[1]

    dot = dx * bpx + dy * bpy

    return dot > 0



def calculate_best_i(i,waypoints):
    j = max(i - 2, 1)
    boat_pos = get_mqtt.get_boat_position()
    if boat_pos is not None:
        while is_past_waypoint(waypoints[j-1],waypoints[j],boat_pos):
            j += 1
        return j
    else:
        return i

def replan(buoy_positions, i, waypoints, start_position, start_heading_deg):
    with camera.buoy_list_lock:
        waypoints = padplanning_wrapper(
            buoy_positions,
            marge=MARGE,
            start_position=start_position,
            start_heading_deg=start_heading_deg,
        )
    post_mqtt.publish_path(waypoints)
    i = calculate_best_i(i, waypoints)
    prev_waypoint = waypoints[i - 1]
    next_waypoint = waypoints[i]
    set_waypoint(waypoints[i + INDEX_LOOK_AHEAD])
    return waypoints, i, prev_waypoint, next_waypoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    # ------------------------------------------------------------------
    # 1. Seed the shared buoy list with a-priori positions BEFORE
    #    starting the camera threads, so the worker thread immediately
    #    has anchor points to match detections against.
    # ------------------------------------------------------------------
    with camera.buoy_list_lock:
        camera.buoy_list.clear()
        camera.buoy_list.extend(APRIORI_BUOYLIST)


    camera.open_log()

    # ------------------------------------------------------------------
    # 2. Start the camera pipeline in a background thread so it doesn't
    #    block the mission loop below.  run_cameras() joins its threads
    #    internally, so we wrap it in a daemon thread here.
    # ------------------------------------------------------------------
    cam_thread = threading.Thread(target=camera.run_cameras, daemon=True)
    cam_thread.start()

    # Give the RTSP streams and YOLO model time to warm up.
    time.sleep(10)
    
    initial_boat_position = get_mqtt.get_boat_position()
    initial_start_position = (initial_boat_position['latitude'], initial_boat_position['longitude'])
    initial_start_heading = initial_boat_position['heading'] or 0

    # ------------------------------------------------------------------
    # 3. Take a thread-safe snapshot of buoy_positions for path planning.
    #    buoy_list is mutated in place by the worker thread, so we hold
    #    the lock only long enough to grab a reference — the list object
    #    itself is shared, so later reads of buoy_positions[i] will see
    #    updated detections as long as we re-read under the lock.
    # ------------------------------------------------------------------
    with camera.buoy_list_lock:
        buoy_positions = camera.buoy_list   # shared reference, not a copy

    # ------------------------------------------------------------------
    # 4. Initial path plan and sail.
    # ------------------------------------------------------------------
    waypoints = padplanning_wrapper(buoy_positions, marge=MARGE, state='START', start_position=initial_start_position, start_heading_deg=initial_start_heading) # padplanning_wrapper(buoy_positions, marge=MARGE, state='START')
    # sail_path(waypoints)
    start_navigation()
    post_mqtt.publish_path(waypoints)

    with camera.buoy_list_lock:
        buoy0_lat, buoy0_lon = buoy_positions[0][0]
        buoy1_lat, buoy1_lon = buoy_positions[1][0]


    R_EARTH = 6_371_000.0
    m_per_deg_lat = math.pi / 180.0 * R_EARTH
    m_per_deg_lon = math.cos(math.radians(buoy1_lat)) * math.pi / 180.0 * R_EARTH
    if TEST_PADAANPASSING_ZONDER_BOEIEN:
        buoy_positions[0] += [(
            buoy0_lat + 4 * BUOY_MATCH_DISTANCE * (random.random()-.5) / m_per_deg_lat,
            buoy0_lon + 4 * BUOY_MATCH_DISTANCE * (random.random()-.5) / m_per_deg_lon
        )]


    i = 1
    prev_waypoint = waypoints[i-1]
    next_waypoint = waypoints[i]
    sail_waypoint = waypoints[i + INDEX_LOOK_AHEAD]
    set_waypoint(sail_waypoint)


    near_B0 = False
    near_B1 = False




    while True:

        boat_pos = get_mqtt.get_boat_position()
        if boat_pos is None:
            time.sleep(0.5)
            continue

        dist_B0 = haversine(boat_pos["latitude"], buoy0_lat,
                            boat_pos["longitude"], buoy0_lon)
        dist_B1 = haversine(boat_pos["latitude"], buoy1_lat,
                            boat_pos["longitude"], buoy1_lon)

        if not near_B0 and dist_B0 < STATE_TRANS_DIST:
            near_B0, near_B1 = True, False           # the true/false setting is correct like this (when we come near the second buoy we know we're away from the first one and vice versa), dont change that @Robin's Claude
            waypoints, i, prev_waypoint, next_waypoint = replan(
                buoy_positions, i, waypoints, initial_start_position, initial_start_heading
            )
            if TEST_PADAANPASSING_ZONDER_BOEIEN:
                buoy_positions[1] += [(
                    buoy1_lat + 4 * BUOY_MATCH_DISTANCE * (random.random()-.5) / m_per_deg_lat,
                    buoy1_lon + 4 * BUOY_MATCH_DISTANCE * (random.random()-.5) / m_per_deg_lon
                )]

        elif not near_B1 and dist_B1 < STATE_TRANS_DIST:
            near_B0, near_B1 = False, True
            waypoints, i, prev_waypoint, next_waypoint = replan(
                buoy_positions, i, waypoints, initial_start_position, initial_start_heading
            )
            if TEST_PADAANPASSING_ZONDER_BOEIEN:
                buoy_positions[0] = [(buoy0_lat,buoy0_lon)]


        if is_past_waypoint(prev_waypoint,next_waypoint,boat_pos):
            i += 1
            if i >= len(waypoints)-INDEX_LOOK_AHEAD:
                break
            prev_waypoint = next_waypoint
            next_waypoint = waypoints[i]
            sail_waypoint = waypoints[i + INDEX_LOOK_AHEAD]
            set_waypoint(sail_waypoint)

    stop_navigation()


if __name__ == "__main__":
    run()