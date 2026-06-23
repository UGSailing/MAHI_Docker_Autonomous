"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import camera
import time
from padplanning import padplanning_8
from camera import get_current_frame
from padplanning_thomas import process_pair
import get_mqtt
from autopilot import sail_path
from padplanning_slalom import padplanning_wrapper

def main() -> None:
    camera.run_cameras()


    time.sleep(10)
    buoy_positions = [[(50,3)],[(50,3)]]
    #waypoints = padplanning_8(buoy_positions,x=2)
    waypoints = padplanning_wrapper(buoy_positions, x=2, state='START')
    sail_path(waypoints)
    while True:
        


        left_frame, right_frame = get_current_frame('left',False), get_current_frame('right',False)
        boat_pos = get_mqtt.get_boat_position()
        updated, buoy_positions = process_pair(boat_pos, buoy_positions, left_frame, right_frame, distance_allowed = 2)
        if updated:
            waypoints = padplanning_8(buoy_positions,x=2)
            sail_path(waypoints)



if __name__ == "__main__":
    main()