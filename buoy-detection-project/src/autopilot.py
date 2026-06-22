"""
MAHI API - sail_path()
Based on MAHI API manual (rev 0.3, 23/09/2025).
Requires: paho-mqtt
"""

import json
import uuid
import time
import math
import paho.mqtt.client as mqtt


ROUTE_TOPIC = "sense-3C6D66019257/autopilot/mahi-1234/route"
GNSS_TOPIC = "sense-3C6D66019257/gnss/Left/pvt"
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
    position = {"lat": None, "lon": None}

    def on_message(client, userdata, msg):
        try:
            fix = json.loads(msg.payload.decode())
            if not fix.get("FixIsValid"):
                return
            lat_lon = (fix.get("Position") or {}).get("LatLon") or {}
            lat = lat_lon.get("Latitude")
            lon = lat_lon.get("Longitude")
            if lat is not None and lon is not None:
                position["lat"] = lat
                position["lon"] = lon
        except (json.JSONDecodeError, KeyError):
            pass

    client = mqtt.Client()
    client.on_message = on_message
    client.connect("172.17.0.1", 1883)
    client.subscribe(GNSS_TOPIC)

    # Take control
    client.publish("external/command/button", json.dumps({
        "UUID": uuid.uuid4().hex,
        "action": "take_cmd"
    }))
    client.loop_start()
    time.sleep(2)

    for active_index, ((target_lat, target_lon), _) in enumerate(waypoints):
        publish_route(client, waypoints, active_index)
        print(f"Heading to waypoint {active_index}: ({target_lat}, {target_lon})")

        while True:
            time.sleep(0.5)
            publish_route(client, waypoints, active_index)
            if position["lat"] is not None:
                dist = haversine(position["lat"], position["lon"], target_lat, target_lon)
                print(f"Distance to waypoint {active_index}: {dist:.1f}m")
                if dist < ARRIVAL_RADIUS_M:
                    print(f"Reached waypoint {active_index}")
                    break
            else:
                print("Waiting for GPS fix...")

    print("All waypoints reached.")
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    sail_path([
        ((51.14338696558262, 2.747221672346525), 4.0),
        ((51.14344030487403, 2.74765152071387), 4.0)
    ])