"""
set_mqtt.py

Background MQTT publisher used by the detection pipeline. Mirrors the
lazy-singleton-client pattern in get_mqtt.py, but for publishing instead of
subscribing:

    from set_mqtt import publish_video_frame, publish_detection_coordinates

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

_client: Optional[mqtt.Client] = None
_client_lock = threading.Lock()


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