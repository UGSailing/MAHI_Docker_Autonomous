"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import camera
from post_mqtt import *


def main() -> None:
    publish_detection_coordinates([{"buoy": "1", "latitude": 51.08231926089576, "longitude": 2.5878133539137997}])
    
    
if __name__ == "__main__":
    main()