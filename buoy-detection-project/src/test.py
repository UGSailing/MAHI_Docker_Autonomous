"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import get_mqtt
import os
import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "172.17.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")  # optional
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")  # optional
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "set-mqtt-publisher")


def main() -> None:
    # client = mqtt.Client(client_id="stream-preview", clean_session=True)
    # client.max_queued_messages_set(1)
    # client.reconnect_delay_set(min_delay=1, max_delay=5)
    # client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    # client.loop_start()

    print("STARTED")
    while True:
        time.sleep(2)
        temp = get_mqtt.get_mahi_temperature()
        print(temp)

    
    
if __name__ == "__main__":
    main()
