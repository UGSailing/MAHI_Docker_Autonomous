"""
post_mqtt.py

Background MQTT publisher used by the detection pipeline. Mirrors the
lazy-singleton-client pattern in get_mqtt.py, but for publishing instead of
subscribing:

    from post_mqtt import publish_video_frame, publish_detection_coordinates

    publish_video_frame("left", jpg_bytes)
    publish_detection_coordinates("left", latitude=51.05, longitude=3.72)

A single MQTT client is created on first use and kept alive in a background
network thread (loop_start()), so publish calls never block waiting to
establish a connection.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional
import base64

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "172.17.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")  # optional
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")  # optional
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "set-mqtt-publisher")

VIDEO_TOPICS = {
    "left": "detections/video/left",
    "right": "detections/video/right",
}

COORDINATE_TOPIC = "detections/coordinates"
PATH_TOPIC = "navigation/path"

_client: Optional[mqtt.Client] = None
_client_lock = threading.Lock()

CROSSLINE_TOPIC = "navigation/crossline"
CAN_RX_TOPIC = "can/ugent/rx"


def _ensure_client_started() -> mqtt.Client:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
        if MQTT_USER:
            client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        # Keep the internal outgoing queue tiny: we never want stale
        # frames/detections sitting around waiting to be sent. If the
        # broker can't keep up we'd rather drop at the application level
        # than have publish latency creep up over time.
        client.max_queued_messages_set(1)
        client.reconnect_delay_set(min_delay=1, max_delay=5)
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        client.loop_start()  # background network thread, non-blocking publish()

        _client = client
        return _client


def _topic_for(topics: dict[str, str], side: str) -> str:
    try:
        return topics[side]
    except KeyError as error:
        raise ValueError(f"Unknown camera side {side!r}, expected 'left' or 'right'") from error

def publish_crossline(point1: tuple[float, float], point2: tuple[float, float]) -> None:
    """Publish the tilted cross line to navigation/crossline.

    Each point is a (latitude, longitude) tuple. The two points define
    the line the boat must cross for the current waypoint to be considered
    passed.

        [
            {"latitude": 50.9148, "longitude": 2.6892},
            {"latitude": 50.9150, "longitude": 2.6895}
        ]

    QoS 1 and retain=True so that a visualisation subscriber that connects
    mid-mission immediately sees the current cross line.
    """
    payload = json.dumps([
        (point1[0], point1[1]),
        (point2[0], point2[1]),
    ])
    client = _ensure_client_started()
    info = client.publish(
        CROSSLINE_TOPIC,
        payload=payload,
        qos=1,
        retain=True,
    )
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed for crossline: rc={info.rc}")



def publish_video_frame(side: str, jpg_bytes: bytes) -> None:
    """Publish an annotated JPEG frame for the given camera side
    ('left'/'right') to detections/video/<side>. Raw JPEG bytes, not
    base64/JSON, to keep payload size and CPU overhead down. QoS 0 so this
    never blocks waiting for a broker ack.
    """
    topic = _topic_for(VIDEO_TOPICS, side)
    client = _ensure_client_started()
    info = client.publish(topic, payload=jpg_bytes, qos=0, retain=False)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed for {side} video: rc={info.rc}")


def publish_detection_coordinates(detections: list[dict]) -> None:
    client = _ensure_client_started()
    info = client.publish(
        COORDINATE_TOPIC,
        payload=json.dumps(detections),
        qos=0,
        retain=False,
    )

    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed: rc={info.rc}")


def publish_point(waypoint) -> None:
    payload = json.dumps(
        {"latitude": waypoint[0][0], "longitude": waypoint[0][1], "speed": waypoint[1]}
    )
    client = _ensure_client_started()
    info = client.publish(
        "navigation/current",
        payload=payload,
        qos=1,
        retain=True,
    )
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed for path: rc={info.rc}")


def publish_path(waypoints: list[tuple[float, float]]) -> None:
    """Publish the planned path to navigation/path.
 
    Each waypoint is a ((latitude, longitude), speed) tuple.  They are
    serialised as a JSON array of objects so subscribers don't need to
    know the tuple order:
 
        [
            {"latitude": 50.9148, "longitude": 2.6892, "speed": 1.5},
            {"latitude": 50.9150, "longitude": 2.6895, "speed": 2.0},
            ...
        ]
 
    QoS 1 and retain=True are used here (unlike video/detection publishes)
    because the path is mission-critical: a subscriber that connects after
    the publish still needs the current path, and we want at-least-once
    delivery to the broker.
    """
    payload = json.dumps(
        [{"latitude": lat, "longitude": lon, "speed": speed}
         for (lat, lon), speed in waypoints]
    )
    client = _ensure_client_started()
    info = client.publish(
        PATH_TOPIC,
        payload=payload,
        qos=1,
        retain=True,
    )
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed for path: rc={info.rc}")


def publish_can_message(can_id: int, data_bytes: bytes) -> None:
    """Publish a CAN frame to can/ugent/rx for transmission onto the bus.

    data_bytes is base64-encoded because raw bytes aren't JSON-safe:

        {"can_id": 1536, "data": "AAECAwQF"}

    QoS 1 so the frame reaches the broker at-least-once, but retain=False:
    unlike the path/crossline publishes, a stale CAN command replayed to a
    late-connecting subscriber could actuate something unintended.
    """
    payload = json.dumps({
        "can_id": can_id,
        "data": base64.b64encode(data_bytes).decode(),
    })
    client = _ensure_client_started()
    info = client.publish(CAN_RX_TOPIC, payload=payload, qos=1, retain=False)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed for CAN message: rc={info.rc}")