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


BROKER = "172.17.0.1"
ROUTE_TOPIC = "sense-3C6D66019257/autopilot/mahi-1234/route"
NMEA_TOPIC = "+/nmea/#"
ARRIVAL_RADIUS_M = 5.0  # meters within which a waypoint is considered reached


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Returns distance in meters between two GPS coordinates."""
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
    plan = "\r\n".join(lines) + "\r\n"
    client.publish(ROUTE_TOPIC, plan)


def sail_path(waypoints: list[tuple[tuple[float, float], float]]) -> None:
    """
    Sail through all waypoints in order, advancing to the next one
    once the vessel is within ARRIVAL_RADIUS_M of the current target.

    Parameters
    ----------
    waypoints : list of ((lat, lon), speed_knots)
    """
    active_index = 0
    current_lat = None
    current_lon = None

    def on_message(client, userdata, msg):
        nonlocal current_lat, current_lon
        payload = msg.payload.decode(errors="ignore")
        # Parse GPGGA for position
        if "GPGGA" in payload:
            parts = payload.split(",")
            try:
                raw_lat = float(parts[2])
                raw_lon = float(parts[4])
                # Convert NMEA ddmm.mmmm to decimal degrees
                current_lat = int(raw_lat / 100) + (raw_lat % 100) / 60
                current_lon = int(raw_lon / 100) + (raw_lon % 100) / 60
            except (ValueError, IndexError):
                pass

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER, 1883)
    client.subscribe(NMEA_TOPIC)
    client.loop_start()

    # Take control
    client.publish("external/command/button", json.dumps({
        "UUID": uuid.uuid4().hex,
        "action": "take_cmd"
    }))
    time.sleep(2)

    print(f"Sailing {len(waypoints)} waypoints...")

    while active_index < len(waypoints):
        publish_route(client, waypoints, active_index)

        target_lat, target_lon = waypoints[active_index][0]
        print(f"Heading to waypoint {active_index}: ({target_lat}, {target_lon})")

        # Wait until close enough to advance
        while True:
            time.sleep(0.5)
            if current_lat is not None and current_lon is not None:
                dist = haversine(current_lat, current_lon, target_lat, target_lon)
                if dist < ARRIVAL_RADIUS_M:
                    print(f"Reached waypoint {active_index} (dist={dist:.1f}m)")
                    active_index += 1
                    break

    print("All waypoints reached.")
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    sail_path([
        ((51.14338696558262, 2.747221672346525), 4.0),
        ((51.14344030487403, 2.74765152071387), 4.0)
    ])