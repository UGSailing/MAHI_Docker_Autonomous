"""
MAHI API - sail_path()
Based on MAHI API manual (rev 0.3, 23/09/2025).
Requires: paho-mqtt, get_mqtt.py
"""

import json
import uuid
import time
import math
import paho.mqtt.client as mqtt
from get_mqtt import get_boat_position


ROUTE_TOPIC = "sense-3C6D66019257/autopilot/mahi-1234/route"
ARRIVAL_RADIUS_M = 10.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def publish_route(client, waypoints, active_index):
    lines = [f"START {uuid.uuid4().hex}"]
    for i, ((lat, lon), speed) in enumerate(waypoints):
        active = 1 if i == active_index else 0
        lines.append(f"W, {lat:.7f}, {lon:.7f}, {speed:.3f}, {active}")
    lines.append("END")
    client.publish(ROUTE_TOPIC, "\r\n".join(lines) + "\r\n")


def sail_path(waypoints: list[tuple[tuple[float, float], float]]) -> None:
    client = mqtt.Client()
    client.connect("172.17.0.1", 1883)
    client.loop_start()

    # Take control
    client.publish("external/command/button", json.dumps({
        "UUID": uuid.uuid4().hex,
        "action": "take_cmd"
    }))
    time.sleep(2)

    for active_index, ((target_lat, target_lon), _) in enumerate(waypoints):
        publish_route(client, waypoints, active_index)
        print(f"Heading to waypoint {active_index}: ({target_lat}, {target_lon})")

        while True:
            time.sleep(0.5)
            pos = get_boat_position()
            if pos is not None:
                dist = haversine(pos["latitude"], pos["longitude"], target_lat, target_lon)
                if dist < ARRIVAL_RADIUS_M:
                    print(f"Reached waypoint {active_index} (dist={dist:.1f}m)")
                    break

    print("All waypoints reached.")
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    pos = get_boat_position()
    print(pos)
    sail_path([
        ((51.14338696558262, 2.747221672346525), 4.0),
        ((51.14344030487403, 2.74765152071387), 4.0)
    ])