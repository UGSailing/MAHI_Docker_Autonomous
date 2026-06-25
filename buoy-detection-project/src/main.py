"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras in a
background thread, then runs the mission loop (detect → plan → sail).
"""

import threading
import time

import camera
import get_mqtt
import post_mqtt
from padplanning import padplanning_8
from padplanning_slalom import padplanning_wrapper
# from autopilot import sail_path
from autopilot import set_waypoint, start_navigation, stop_navigation
from config import INDEX_LOOK_AHEAD
from config import MARGE
from config import STATE_TRANS_DIST


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

def is_past_waypoint(prev_waypoint, next_waypoint, boat_pos):
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
        

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Seed the shared buoy list with a-priori positions BEFORE
    #    starting the camera threads, so the worker thread immediately
    #    has anchor points to match detections against.
    # ------------------------------------------------------------------
    with camera.buoy_list_lock:
        camera.buoy_list.clear()
        camera.buoy_list.extend([
            [(51.14339735730541, 2.7468965058450774)],   # buoy 0 — replace with real a-priori GPS coords
            [(51.14344451713093, 2.7475114395335662)],   # buoy 1 — replace with real a-priori GPS coords
        ])       

        

    # ------------------------------------------------------------------
    # 2. Start the camera pipeline in a background thread so it doesn't
    #    block the mission loop below.  run_cameras() joins its threads
    #    internally, so we wrap it in a daemon thread here.
    # ------------------------------------------------------------------
    cam_thread = threading.Thread(target=camera.run_cameras, daemon=True)
    cam_thread.start()

    # Give the RTSP streams and YOLO model time to warm up.
    time.sleep(10)

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
    waypoints = padplanning_wrapper(buoy_positions, marge=MARGE, state='START') # padplanning_wrapper(buoy_positions, marge=MARGE, state='START')
    # sail_path(waypoints)
    start_navigation()
    post_mqtt.publish_path(waypoints)

    # ------------------------------------------------------------------
    # 5. Wait until the boat is within 7 m of buoy 0's a-priori position.
    # ------------------------------------------------------------------
    with camera.buoy_list_lock:
        buoy0_lat, buoy0_lon = buoy_positions[0][0]
        buoy1_lat, buoy1_lon = buoy_positions[1][0]
    
    i = 1
    prev_waypoint = waypoints[i-1]
    next_waypoint = waypoints[i]
    set_waypoint(next_waypoint + INDEX_LOOK_AHEAD)
    while True:
        boat_pos = get_mqtt.get_boat_position()
        if boat_pos is None:
            time.sleep(0.5)
            continue

        dist = haversine(
            boat_pos["latitude"], buoy0_lat,
            boat_pos["longitude"], buoy0_lon,
        )        
        if dist < STATE_TRANS_DIST:
            print("distance 1 smaller")
            break

        if is_past_waypoint(prev_waypoint,next_waypoint,boat_pos):
            print("i")
            print(i)
            i += 1
            prev_waypoint = next_waypoint
            next_waypoint = waypoints[i]
            set_waypoint(next_waypoint + INDEX_LOOK_AHEAD)
            
    print("DETECT_1")

    with camera.buoy_list_lock:
        waypoints = padplanning_wrapper(buoy_positions, marge=MARGE, state='DETECT_1') # padplanning_wrapper(buoy_positions, marge=MARGE, state='DETECT_1')
    # sail_path(waypoints)
    post_mqtt.publish_path(waypoints)

    # nieuwe i berekenen
    i = calculate_best_i(i,waypoints)

    prev_waypoint = waypoints[i-1]
    next_waypoint = waypoints[i]
    set_waypoint(next_waypoint + INDEX_LOOK_AHEAD)

    # ------------------------------------------------------------------
    # 6. Wait until the boat is within 7 m of buoy 1's a-priori position.
    # ------------------------------------------------------------------
    while True:
        boat_pos = get_mqtt.get_boat_position()
        if boat_pos is None:
            time.sleep(0.5)
            continue
        dist = haversine(
            boat_pos["latitude"], buoy1_lat,
            boat_pos["longitude"], buoy1_lon,
        )
        print("BOEI 2")
        print(dist)
        if dist < STATE_TRANS_DIST:
            break

        if is_past_waypoint(prev_waypoint,next_waypoint,boat_pos):
            i += 1
            prev_waypoint = next_waypoint
            next_waypoint = waypoints[i]
            set_waypoint(waypoints[i] + INDEX_LOOK_AHEAD)


    print("DETECT_2")

    with camera.buoy_list_lock:
        waypoints = padplanning_wrapper(buoy_positions, marge=MARGE, state='DETECT_2')#padplanning_wrapper(buoy_positions, marge=MARGE, state='DETECT_2')
    # sail_path(waypoints) # TODO uncomment
    post_mqtt.publish_path(waypoints)

    # nieuwe i berekenen
    i = calculate_best_i(i,waypoints)

    prev_waypoint = waypoints[i-1]
    next_waypoint = waypoints[i]
    set_waypoint(next_waypoint + INDEX_LOOK_AHEAD)

    while True:
        boat_pos = get_mqtt.get_boat_position()
        if boat_pos is None:
            time.sleep(0.5)
            continue
        print("i")
        print(i)
        if is_past_waypoint(prev_waypoint,next_waypoint,boat_pos):
            i += 1
            if i < len(waypoints)-INDEX_LOOK_AHEAD:
                prev_waypoint = next_waypoint
                next_waypoint = waypoints[i]
                set_waypoint(next_waypoint + INDEX_LOOK_AHEAD)
            else:
                stop_navigation()

if __name__ == "__main__":
    main()