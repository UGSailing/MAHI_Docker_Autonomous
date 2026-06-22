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
    client = mqtt.Client()
    client.connect("172.17.0.1", 1883)
    client.loop_start()

    lines = [f"START {uuid.uuid4().hex}"]
    for (lat, lon), speed in waypoints:
        lines.append(f"W, {lat:.7f}, {lon:.7f}, {speed:.3f}, 0")
    lines.append("END")
    plan = "\r\n".join(lines) + "\r\n"

    client.publish("sense-3C6D66019257/autopilot/mahi-1234/route", plan)

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    while (1):
        sail_path([
            ((51.14338696558262, 2.747221672346525), 4.0),
            ((51.14344030487403, 2.74765152071387), 4.0)
        ])   
        time.sleep(0.5) 