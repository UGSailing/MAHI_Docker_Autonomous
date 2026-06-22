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
    client.publish("sense-3C6D66019257/autopilot/external/command/button", json.dumps({
        "UUID": uuid.uuid4().hex,
        "action": "take_cmd"
    }))
    time.sleep(2)  # wait for control to be granted

    # # Step 2: build and publish the route
    # lines = [f"START {uuid.uuid4().hex}"]
    # for (lat, lon), speed in waypoints:
    #     lines.append(f"W, {lat:.7f}, {lon:.7f}, {speed:.3f}, 0")
    # lines.append("END")
    # plan = "\r\n".join(lines) + "\r\n"

    # client.publish("sense-3C6D66019257/autopilot/mahi-1234/route", plan)

    # # Step 3: set autopilot to PathTracking
    # client.publish("external/mode_request", json.dumps({
    #     "autopilot_mode": "PathTracking",
    #     "autopilot_heading": 0.0,
    #     "persistent": True,
    # }))

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    sail_path([
        ((51.0728671, 4.3654559), 1.0),
        ((51.0744104, 4.3659370), 1.0),
        ((51.0754940, 4.3660841), 1.5),
        ((51.0763674, 4.3662915), 2.0),
    ])