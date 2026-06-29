"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import get_mqtt
import paho.mqtt.client as mqtt


def main() -> None:
    client = mqtt.Client(client_id="stream-preview", clean_session=True)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    # Keep the outgoing queue at 1 so stale frames are never sent.
    client.max_queued_messages_set(1)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()

    temp = get_mqtt.get_mahi_temperature()
    print(temp)

    
    
if __name__ == "__main__":
    main()
