"""
MAHI API - sail_path()
Based on MAHI API manual (rev 0.3, 23/09/2025).
Requires: paho-mqtt
"""

import json
import uuid
import time
import paho.mqtt.client as mqtt


def sail_path(waypoints: list[tuple[tuple[float, float], float]]) -> None:
    """
    Connect to the MAHI broker, take control, and sail a route.

    Parameters
    ----------
    waypoints : list of ((lat, lon), speed_knots)
        Example: [((51.0728, 4.3654), 1.0), ((51.0744, 4.3659), 2.5)]
    """
    client = mqtt.Client()
    client.connect("172.17.0.1", 1883)
    client.loop_start()

    # Step 1: take control
    client.publish("external/command/button", json.dumps({
        "UUID": "",
        "action": "take_cmd"
    }))
    time.sleep(2)  # wait for control to be granted

    # Step 2: build and publish the route
    lines = [f"START {uuid.uuid4().hex}"]
    for (lat, lon), speed in waypoints:
        lines.append(f"W, {lat:.7f}, {lon:.7f}, {speed:.3f}, 1")
    lines.append("END")
    plan = "\r\n".join(lines) + "\r\n"

    client.publish("sense-3C6D66019257/autopilot/mahi-1234/route", plan)

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    sail_path([
        ((51.14437193985988, 2.7471670611359977), 1.0),
    ])   