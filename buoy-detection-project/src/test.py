"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import get_mqtt
import os
import paho.mqtt.client as mqtt

def main() -> None:
    while True:
        time.sleep(1)
        publish_temperature(get_mqtt.get_mahi_temperature())

    
    
if __name__ == "__main__":
    main()
