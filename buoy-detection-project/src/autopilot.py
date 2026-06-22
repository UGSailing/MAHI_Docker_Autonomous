"""
MAHI API - sail_path()
Publishes an instant route plan to the MAHI autopilot over MQTT.

Based on MAHI API manual (rev 0.3, 23/09/2025).
Requires: paho-mqtt
"""

import json
import uuid
import time
import paho.mqtt.client as mqtt


def sail_path(
    sense_id: str,
    waypoints: list[tuple[tuple[float, float], float]],
    client: mqtt.Client,
    execute: bool = True,
    waypoint_type: str = "W",
) -> str:
    """
    Build and publish a MAHI route plan from a list of (coordinate, speed) tuples.

    Parameters
    ----------
    sense_id : str
        The MAHI Sense device ID (e.g. "sense-ABC123").
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
        Default waypoint type for all points: "W" (normal), "L" (loiter),
        or "D" is not supported via this helper (use build_plan() directly
        for DP waypoints which require an extra heading parameter).

    Returns
    -------
    str
        The plan string that was published, for inspection/logging.

    Raises
    ------
    ValueError
        If waypoints is empty or waypoint_type is unsupported.
    RuntimeError
        If the MQTT publish fails (rc != 0).

    Notes
    -----
    - The plan is uploaded as a *persistent* plan
      (topic: sense-ID/autopilot/external/route-persistent).
    - Each waypoint's 'active' flag is set to 0; in persistent mode the
      autopilot manages advancement automatically.
    - W waypoints are ticked off automatically when the vessel comes within
      the configured tick-off distance. L and D waypoints require explicit
      continuation triggers or C-line configuration.
    - Call this function periodically if you need a live "navigate to here"
      behaviour; for that use case, publish to the non-persistent route topic
      (sense-ID/autopilot/external/route) and mark exactly one waypoint active=1.
    """
    if not waypoints:
        raise ValueError("waypoints list must not be empty.")

    supported_types = {"W", "L"}
    if waypoint_type not in supported_types:
        raise ValueError(
            f"waypoint_type must be one of {supported_types}. "
            "For DP (D) waypoints build the plan string manually."
        )

    plan = _build_plan(waypoints, waypoint_type)
    _publish_plan(sense_id, plan, client)

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
    """
    Assemble the MAHI plan text format.

    Format per waypoint line (W / L):
        <Type>, <Lat>, <Lon>, <Speed>, <active>
    active is always 0 for persistent plans (autopilot manages advancement).
    """
    # A fresh random token is required every time the plan is updated
    plan_token = uuid.uuid4().hex

    lines = [f"START {plan_token}"]

    for (lat, lon), speed_knots in waypoints:
        # active=0: in persistent mode the autopilot handles waypoint advancement
        lines.append(f"{waypoint_type}, {lat:.7f}, {lon:.7f}, {speed_knots:.3f}, 0")

    lines.append("END")

    # MAHI expects \r\n line endings
    return "\r\n".join(lines) + "\r\n"


def _publish_plan(sense_id: str, plan: str, client: mqtt.Client) -> None:
    """
    Publish the plan to the persistent route topic.

    Topic: sense-ID/autopilot/external/route-persistent
    """
    topic = f"{sense_id}/autopilot/external/route-persistent"
    result = client.publish(topic, plan)

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(
            f"Failed to publish plan to '{topic}' (rc={result.rc})."
        )


def _set_path_tracking(client: mqtt.Client) -> None:
    """
    Switch the autopilot to PathTracking mode with persistent=True so that
    the uploaded plan starts executing immediately.

    Topic: external/mode_request
    """
    mode_msg = json.dumps({
        "autopilot_mode": "PathTracking",
        "autopilot_heading": 0.0,   # heading is managed by the route in this mode
        "persistent": True,
    })
    result = client.publish("external/mode_request", mode_msg)

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(
            f"Failed to publish mode_request (rc={result.rc})."
        )


# ---------------------------------------------------------------------------
# Convenience: abort any active plan
# ---------------------------------------------------------------------------

def abort_plan(sense_id: str, client: mqtt.Client) -> None:
    """
    Abort a running plan by switching the autopilot to Manual mode and
    uploading an empty (START … END) instant plan, which cancels the
    current persistent plan.

    Parameters
    ----------
    sense_id : str
        The MAHI Sense device ID.
    client : mqtt.Client
        A connected paho-mqtt client instance.
    """
    # 1. Switch out of PathTracking
    mode_msg = json.dumps({
        "autopilot_mode": "Manual",
        "autopilot_heading": 0.0,
        "persistent": False,
    })
    client.publish("external/mode_request", mode_msg)

    # 2. Publish an empty instant plan to clear the previous one
    empty_plan = f"START {uuid.uuid4().hex}\r\nEND\r\n"
    client.publish(f"{sense_id}/autopilot/external/route", empty_plan)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SENSE_ID = "sense-3C6D66019257"
    BROKER   = "172.17.0.1"   # replace with actual broker IP
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
                plan_text = sail_path(SENSE_ID, route, client, execute=True)
                print("Published plan:\n", plan_text)

    mqttc = mqtt.Client()
    mqttc.on_message = on_message
    mqttc.connect(BROKER, PORT)
    mqttc.subscribe("onboard/state")
    mqttc.loop_start()

    # Request control
    take_control_msg = json.dumps({"UUID": uuid.uuid4().hex, "action": "take_cmd"})
    mqttc.publish("external/command/button", take_control_msg)
    print("Waiting for control confirmation...")

    # Wait until in_command is set by the callback, or time out
    timeout = 10
    start = time.time()
    while not in_command and time.time() - start < timeout:
        time.sleep(0.1)

    if not in_command:
        print("ERROR: Did not receive External state within timeout. Is the vessel ready?")

    time.sleep(2)  # let the plan execute a moment before disconnecting
    mqttc.loop_stop()
    mqttc.disconnect()