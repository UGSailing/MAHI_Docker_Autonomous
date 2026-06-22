"""
MAHI API - sail_path()
Publishes a persistent route plan to the MAHI autopilot over MQTT.

Based on MAHI API manual (rev 0.3, 23/09/2025).
Requires: paho-mqtt
"""

import json
import uuid
import time
import paho.mqtt.client as mqtt


def sail_path(
    sense_id: str,
    pilot_id: str,
    waypoints: list[tuple[tuple[float, float], float]],
    client: mqtt.Client,
    execute: bool = True,
    waypoint_type: str = "W",
) -> str:
    """
    Build and publish a MAHI persistent route plan from a list of (coordinate, speed) tuples.

    Parameters
    ----------
    sense_id : str
        The MAHI Sense device ID (e.g. "sense-3C6D66019257").
    pilot_id : str
        The MAHI pilot ID (e.g. "mahi-1234").
    waypoints : list of ((lat, lon), speed_knots)
        Ordered list of waypoints. Each entry is a tuple of:
          - (latitude: float, longitude: float)
          - speed in knots: float
        Example: [((51.0728, 4.3654), 1.0), ((51.0744, 4.3659), 2.5)]
    client : mqtt.Client
        A connected paho-mqtt client instance.
    execute : bool
        If True (default), also sets the autopilot to PathTracking mode
        so the plan starts executing immediately after upload.
    waypoint_type : str
        Default waypoint type for all points: "W" (normal) or "L" (loiter).

    Returns
    -------
    str
        The plan string that was published, for inspection/logging.
    """
    if not waypoints:
        raise ValueError("waypoints list must not be empty.")

    if waypoint_type not in {"W", "L"}:
        raise ValueError("waypoint_type must be 'W' or 'L'.")

    plan = _build_plan(waypoints, waypoint_type)
    _publish_plan(sense_id, pilot_id, plan, client)

    if execute:
        _set_path_tracking(client)

    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_plan(
    waypoints: list[tuple[tuple[float, float], float]],
    waypoint_type: str,
) -> str:
    """Assemble the MAHI plan text format with \r\n line endings."""
    plan_token = uuid.uuid4().hex
    lines = [f"START {plan_token}"]

    for (lat, lon), speed_knots in waypoints:
        # active=0: in persistent mode the autopilot handles waypoint advancement
        lines.append(f"{waypoint_type}, {lat:.7f}, {lon:.7f}, {speed_knots:.3f}, 0")

    lines.append("END")
    return "\r\n".join(lines) + "\r\n"


def _publish_plan(sense_id: str, pilot_id: str, plan: str, client: mqtt.Client) -> None:
    """Publish to: sense-ID/autopilot/pilot-ID/route"""
    topic = f"{sense_id}/autopilot/{pilot_id}/route"
    result = client.publish(topic, plan)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"Failed to publish plan to '{topic}' (rc={result.rc}).")


def _set_path_tracking(client: mqtt.Client) -> None:
    """Publish to: external/mode_request (no prefix)"""
    mode_msg = json.dumps({
        "autopilot_mode": "PathTracking",
        "autopilot_heading": 0.0,
        "persistent": True,
    })
    result = client.publish("external/mode_request", mode_msg)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"Failed to publish mode_request (rc={result.rc}).")


# ---------------------------------------------------------------------------
# Convenience: abort any active plan
# ---------------------------------------------------------------------------

def abort_plan(sense_id: str, client: mqtt.Client) -> None:
    """Abort a running plan by switching to Manual and clearing the route."""
    mode_msg = json.dumps({
        "autopilot_mode": "Manual",
        "autopilot_heading": 0.0,
        "persistent": False,
    })
    client.publish("external/mode_request", mode_msg)

    empty_plan = f"START {uuid.uuid4().hex}\r\nEND\r\n"
    client.publish(f"{sense_id}/autopilot/external/route", empty_plan)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SENSE_ID = "sense-3C6D66019257"
    PILOT_ID = "mahi-1234"
    BROKER   = "172.17.0.1"
    PORT     = 1883

    route = [
        ((51.0728671, 4.3654559), 1.0),
        ((51.0744104, 4.3659370), 1.0),
        ((51.0754940, 4.3660841), 1.5),
        ((51.0763674, 4.3662915), 2.0),
    ]

    in_command = False

    def on_message(client, userdata, msg):
        global in_command
        if msg.topic == "onboard/state":
            state = json.loads(msg.payload)
            print("Vessel state:", state)
            if state.get("state") == "External" and not in_command:
                in_command = True
                print("Control confirmed. Uploading plan...")
                plan_text = sail_path(SENSE_ID, PILOT_ID, route, client, execute=True)
                print("Published plan:\n", plan_text)

    mqttc = mqtt.Client()
    mqttc.on_message = on_message
    mqttc.connect(BROKER, PORT)
    mqttc.subscribe("onboard/state")
    mqttc.loop_start()

    # Take control
    take_control_msg = json.dumps({"UUID": uuid.uuid4().hex, "action": "take_cmd"})
    mqttc.publish("external/command/button", take_control_msg)
    print("Waiting for control confirmation...")

    timeout = 10
    start = time.time()
    while not in_command and time.time() - start < timeout:
        time.sleep(0.1)

    if not in_command:
        print("ERROR: Did not receive External state within timeout. Is the vessel ready?")

    time.sleep(2)
    mqttc.loop_stop()
    mqttc.disconnect()