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
    - Control is taken automatically before uploading the plan. The function
      waits 2 seconds for the vessel to switch to "External" state.
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

    # Must take control before any commands will be accepted (API manual §3.1).
    _take_control(client)

    plan = _build_plan(waypoints, waypoint_type)
    _publish_plan(sense_id, plan, client)

    if execute:
        _set_path_tracking(client)

    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _take_control(client: mqtt.Client) -> None:
    """
    Take external control of the vessel (API manual §3.1).

    Publishes a take_cmd action to 'external/command/button' and waits
    2 seconds for the vessel state to switch to "External". If the vessel
    is already under external control this is a no-op from the autopilot's
    perspective, but the publish is harmless.
    """
    msg = json.dumps({"UUID": uuid.uuid4().hex, "action": "take_cmd"})
    result = client.publish("external/command/button", msg)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(
            f"Failed to publish take_cmd (rc={result.rc})."
        )
    # Allow time for the vessel state to transition to "External".
    time.sleep(2)


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
    # A fresh random token is required every time the plan is updated.
    plan_token = uuid.uuid4().hex

    lines = [f"START {plan_token}"]

    for (lat, lon), speed_knots in waypoints:
        # active=0: in persistent mode the autopilot handles waypoint
        # advancement automatically; setting active=1 has no effect per
        # the API manual (§3.10) and may confuse the autopilot.
        lines.append(f"{waypoint_type}, {lat:.7f}, {lon:.7f}, {speed_knots:.3f}, 0")

    lines.append("END")

    # MAHI expects \r\n line endings.
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
    the uploaded plan starts executing immediately (API manual §3.4 / §3.7).

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
    current persistent plan (API manual §3.8).

    Parameters
    ----------
    sense_id : str
        The MAHI Sense device ID.
    client : mqtt.Client
        A connected paho-mqtt client instance.
    """
    # 1. Switch out of PathTracking.
    mode_msg = json.dumps({
        "autopilot_mode": "Manual",
        "autopilot_heading": 0.0,
        "persistent": False,
    })
    client.publish("external/mode_request", mode_msg)

    # 2. Publish an empty instant plan to clear the previous persistent plan.
    empty_plan = f"START {uuid.uuid4().hex}\r\nEND\r\n"
    client.publish(f"{sense_id}/autopilot/external/route", empty_plan)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SENSE_ID = "sense-3C6D66019257"
    BROKER   = "172.17.0.1"
    PORT     = 1883

    # List of ((lat, lon), speed_knots) tuples.
    route = [
        ((51.14436890252032, 2.7471946822090847), 1000),
    ]

    mqttc = mqtt.Client()
    mqttc.connect(BROKER, PORT)
    mqttc.loop_start()

    # sail_path() handles take_cmd internally, so no need to call it here.
    plan_text = sail_path(SENSE_ID, route, mqttc, execute=True)
    print("Published plan:\n", plan_text)

    mqttc.loop_stop()
    mqttc.disconnect()
