import base64
import json
import threading
import time

import paho.mqtt.client as mqtt

import check_camera
import check_gnss
import execute_race

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_TX = "can/ugent/tx"
MQTT_TOPIC_RX = "can/ugent/rx"

ECU_CAN_ID = 0x211  # 529 — autonomous state + selected mission
MAHI_STATE_CAN_ID = 0x63  # 99 — MAHI autonomous state
MAHI_ERROR_FLAGS_CAN_ID = 0x13  # 19 — autonomous systems error flags

MISSION_NONE = 0xFF

# MAHI autonomous states (README_ECU.md)
STATE_WAITING_AMS = 0
STATE_CHECK_SYSTEMS = 1
STATE_SYSTEMS_OK = 2
STATE_RUNNING_MISSION = 3
STATE_FINISHED = 4
STATE_ERROR = 0xFF

# Error-flag bits (1 = system OK)
FLAG_GNSS = 1 << 0
FLAG_CAMERA = 1 << 1
FLAG_OTHER_SOFTWARE = 1 << 2
FLAGS_ALL_OK = FLAG_GNSS | FLAG_CAMERA | FLAG_OTHER_SOFTWARE

state = STATE_WAITING_AMS
error_flags = 0
ecu_state = 0
ecu_mission = MISSION_NONE
selected_mission = MISSION_NONE

state_lock = threading.Lock()
checks_started = False
race_started = False


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC_TX)


def on_message(client, userdata, msg):
    global ecu_state, ecu_mission
    try:
        payload = json.loads(msg.payload.decode())
        if payload.get("can_id") != ECU_CAN_ID:
            return

        raw_bytes = base64.b64decode(payload["data"])
        with state_lock:
            ecu_state = raw_bytes[-1]
            ecu_mission = raw_bytes[-2] if len(raw_bytes) >= 2 else MISSION_NONE
        print(
            f"ECU message: state={ecu_state}, mission={ecu_mission} "
            f"(0b{ecu_state:08b}, 0b{ecu_mission:08b})"
        )
    except Exception as e:
        print(f"Error processing ECU message: {e}")


def run_system_checks():
    global state, error_flags, checks_started

    gnss_ok = check_gnss.check()
    camera_ok = check_camera.check()
    # Docker shell / other software — not yet a separate module
    other_ok = True

    flags = 0
    if gnss_ok:
        flags |= FLAG_GNSS
    if camera_ok:
        flags |= FLAG_CAMERA
    if other_ok:
        flags |= FLAG_OTHER_SOFTWARE

    with state_lock:
        error_flags = flags
        if flags == FLAGS_ALL_OK:
            state = STATE_SYSTEMS_OK
            print(f"All systems OK, state updated to {STATE_SYSTEMS_OK}")
        else:
            state = STATE_ERROR
            print(f"System check failed, state updated to {STATE_ERROR}, flags=0b{flags:03b}")
        checks_started = True


def run_mission(mission_id: int):
    global state, error_flags

    success = execute_race.run(mission_id)
    with state_lock:
        if success:
            state = STATE_FINISHED
            print(f"Mission finished, state updated to {STATE_FINISHED}")
        else:
            state = STATE_ERROR
            error_flags &= ~FLAG_OTHER_SOFTWARE
            print(f"Mission failed, state updated to {STATE_ERROR}")


def state_machine_loop():
    global state, checks_started, race_started, selected_mission, error_flags

    while True:
        with state_lock:
            current_state = state
            current_ecu_state = ecu_state
            current_ecu_mission = ecu_mission
            current_checks_started = checks_started
            current_race_started = race_started

        if (
            current_state == STATE_WAITING_AMS
            and current_ecu_state == 1
            and not current_checks_started
        ):
            with state_lock:
                state = STATE_CHECK_SYSTEMS
                checks_started = True
            print(f"AMS enabled, state updated to {STATE_CHECK_SYSTEMS}")
            threading.Thread(target=run_system_checks, daemon=True).start()

        elif (
            current_state == STATE_SYSTEMS_OK
            and current_ecu_state == 3
            and current_ecu_mission != MISSION_NONE
            and not current_race_started
        ):
            with state_lock:
                selected_mission = current_ecu_mission
                state = STATE_RUNNING_MISSION
                race_started = True
            print(
                f"Mission selected (id={current_ecu_mission}), "
                f"state updated to {STATE_RUNNING_MISSION}"
            )
            threading.Thread(
                target=run_mission, args=(current_ecu_mission,), daemon=True
            ).start()

        elif current_state == STATE_FINISHED and current_ecu_state == 0:
            with state_lock:
                state = STATE_WAITING_AMS
                error_flags = 0
                checks_started = False
                race_started = False
                selected_mission = MISSION_NONE
            print(f"ECU reset, state updated to {STATE_WAITING_AMS}")

        elif current_state == STATE_ERROR and current_ecu_state == 0:
            with state_lock:
                state = STATE_WAITING_AMS
                error_flags = 0
                checks_started = False
                race_started = False
                selected_mission = MISSION_NONE
            print(f"ECU reset after error, state updated to {STATE_WAITING_AMS}")

        time.sleep(0.1)


def publish_can_message(client, can_id: int, data_bytes: bytes):
    payload = {
        "can_id": can_id,
        "data": base64.b64encode(data_bytes).decode(),
    }
    client.publish(MQTT_TOPIC_RX, json.dumps(payload))


def publish_loop(client):
    while True:
        with state_lock:
            current_state = state
            current_flags = error_flags

        state_bytes = bytes([0, 0, 0, 0, 0, 0, 0, current_state])
        publish_can_message(client, MAHI_STATE_CAN_ID, state_bytes)
        print(f"Published: can_id={MAHI_STATE_CAN_ID}, state={current_state}")

        if current_state in (STATE_CHECK_SYSTEMS, STATE_SYSTEMS_OK, STATE_ERROR):
            flag_bytes = bytes([0, 0, 0, 0, 0, 0, 0, current_flags])
            publish_can_message(client, MAHI_ERROR_FLAGS_CAN_ID, flag_bytes)
            print(
                f"Published: can_id={MAHI_ERROR_FLAGS_CAN_ID}, "
                f"error_flags=0b{current_flags:03b}"
            )

        time.sleep(1)


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    threading.Thread(target=state_machine_loop, daemon=True).start()
    publish_loop(client)


if __name__ == "__main__":
    main()
