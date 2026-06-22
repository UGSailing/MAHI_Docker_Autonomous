"""
MAHI API - sail_path()
Based on MAHI API manual (rev 0.3, 23/09/2025).
Requires: paho-mqtt
"""

import json
import uuid
import time
import paho.mqtt.client as mqtt


def sail_path(waypoints: list[tuple[tuple[float, float], float]], active_index: int = 0) -> None:
    """
    Publish an instant route plan to the MAHI autopilot.
    Exactly one waypoint must be marked active (active_index).

    Parameters
    ----------
    waypoints : list of ((lat, lon), speed_knots)
        Example: [((51.0728, 4.3654), 1.0), ((51.0744, 4.3659), 2.5)]
    active_index : int
        Index of the waypoint to mark as active (default: 0).
    """
    client = mqtt.Client()
    client.connect("172.17.0.1", 1883)
    client.loop_start()

    lines = [f"START {uuid.uuid4().hex}"]
    for i, ((lat, lon), speed) in enumerate(waypoints):
        active = 1 if i == active_index else 0
        lines.append(f"W, {lat:.7f}, {lon:.7f}, {speed:.3f}, {active}")
    lines.append("END")
    plan = "\r\n".join(lines) + "\r\n"

    client.publish("sense-3C6D66019257/autopilot/mahi-1234/route", plan)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    while True:
        sail_path([
            ((51.14338696558262, 2.747221672346525), 4.0),
            ((51.14344030487403, 2.74765152071387), 4.0)
        ], active_index=0)
        time.sleep(0.5)