"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import camera
from post_mqtt import *


def main() -> None:
    publish_detection_coordinates([{"buoy": "1", "latitude": 51.08237244887396, "longitude": 2.588432716735549}, {"buoy": "2", "latitude": 51.08243266810585, "longitude": 2.5884415111765375}])
    
    
if __name__ == "__main__":
    main()