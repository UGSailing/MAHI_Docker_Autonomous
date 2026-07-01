import base64
import json
import threading
import time

import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_TX = "can/ugent/tx"
MQTT_TOPIC_RX = "can/ugent/rx"

TARGET_CAN_ID = 289
CAN_ID = 99 # CAN_ID%16=node_id, and the node id of the MAHI Sense is 3

state = 0b00000000
state_lock = threading.Lock()


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC_TX)


def on_message(client, userdata, msg):
    global state
    try:
        payload = json.loads(msg.payload.decode())
        if payload.get("can_id") != TARGET_CAN_ID:
            return

        raw_bytes = base64.b64decode(payload["data"])
        last_byte = raw_bytes[-1]
        print(f"Last byte: 0b{last_byte:08b}")

        if last_byte == 0b00000001:
            with state_lock:
                if state == 0b00000000:
                    state = 0b00000001
                    print("Condition met, state updated from 0b00000000 to 0b00000001")

    except Exception as e:
        print(f"Error processing message: {e}")


def publish_loop(client):
    while True:
        with state_lock:
            current_state = state
        data_bytes = bytes([0, 0, 0, 0, 0, 0, 0, current_state])
        payload = {
            "can_id": CAN_ID,
            "data": base64.b64encode(data_bytes).decode(),
        }
        client.publish(MQTT_TOPIC_RX, json.dumps(payload))
        print(f"Published: can_id={CAN_ID}, state byte = 0b{current_state:08b}")
        time.sleep(1)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

publish_loop(client)
