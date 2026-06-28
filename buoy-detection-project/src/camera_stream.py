"""
stream_preview.py

Lightweight raw video streamer — runs as a completely separate process from
the detection pipeline so the OS can schedule it on a different core.

Opens the left and right RTSP streams, decodes frames, and publishes them
as JPEG over MQTT on the same topics the detection pipeline uses
(detections/video/left and detections/video/right).  No YOLO, no GPS, no
locks — just frames as fast as the network allows.

Run alongside main.py:
    python3 stream_preview.py &
    python3 main.py
"""

from __future__ import annotations

import os
import threading
import time
import urllib.parse

import cv2
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# RTSP config — mirrors camera.py env vars so one .env file covers both
# ---------------------------------------------------------------------------

RIGHT_STREAM_HOST     = os.getenv("RIGHT_STREAM_HOST",     "192.168.77.100")
RIGHT_STREAM_USER     = os.getenv("RIGHT_STREAM_USER",     "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")
RIGHT_STREAM_PATH     = os.getenv("RIGHT_STREAM_PATH",     "/axis-media/media.amp?camera=1")

LEFT_STREAM_HOST      = os.getenv("LEFT_STREAM_HOST",      "192.168.77.99")
LEFT_STREAM_USER      = os.getenv("LEFT_STREAM_USER",      "root")
LEFT_STREAM_PASSWORD  = os.getenv("LEFT_STREAM_PASSWORD",  "mahi1234")
LEFT_STREAM_PATH      = os.getenv("LEFT_STREAM_PATH",      "/axis-media/media.amp?camera=1")

# ---------------------------------------------------------------------------
# MQTT config
# ---------------------------------------------------------------------------

MQTT_HOST      = os.getenv("MQTT_HOST",      "172.17.0.1")
MQTT_PORT      = int(os.getenv("MQTT_PORT",  "1883"))
MQTT_USER      = os.getenv("MQTT_USER")
MQTT_PASSWORD  = os.getenv("MQTT_PASSWORD")

VIDEO_TOPICS = {
    "left":  "detections/video/left",
    "right": "detections/video/right",
}

JPEG_QUALITY = int(os.getenv("PREVIEW_JPEG_QUALITY", "40"))
PREVIEW_WIDTH  = int(os.getenv("PREVIEW_WIDTH",  "640"))
PREVIEW_HEIGHT = int(os.getenv("PREVIEW_HEIGHT", "360"))

# ---------------------------------------------------------------------------
# RTSP helpers
# ---------------------------------------------------------------------------

def _build_url(host: str, user: str, password: str, path: str) -> str:
    u = urllib.parse.quote(user,     safe="")
    p = urllib.parse.quote(password, safe="")
    return f"rtsp://{u}:{p}@{host}{path}"


RIGHT_STREAM_URL = _build_url(
    RIGHT_STREAM_HOST, RIGHT_STREAM_USER, RIGHT_STREAM_PASSWORD, RIGHT_STREAM_PATH
)
LEFT_STREAM_URL = _build_url(
    LEFT_STREAM_HOST, LEFT_STREAM_USER, LEFT_STREAM_PASSWORD, LEFT_STREAM_PATH
)


def _open_capture(url: str) -> cv2.VideoCapture:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------------------------
# MQTT client (shared across both streamer threads)
# ---------------------------------------------------------------------------

def _make_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id="stream-preview", clean_session=True)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.max_queued_messages_set(1)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()
    return client


# ---------------------------------------------------------------------------
# Per-camera streamer thread
# ---------------------------------------------------------------------------

def _stream(url: str, side: str, client: mqtt.Client) -> None:
    """Read frames from *url* and publish raw JPEGs on the MQTT video topic."""
    topic = VIDEO_TOPICS[side]
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        cap = _open_capture(url)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open RTSP stream: {url}")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("RTSP read failed")
                resized = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                ok2, jpg = cv2.imencode(".jpg", resized, encode_params)
                if ok2:
                    client.publish(topic, payload=jpg.tobytes(), qos=0, retain=False)
        except Exception as error:  # noqa: BLE001
            print(f"[stream_preview] {side} error: {error} — retrying in 1 s")
        finally:
            cap.release()
        time.sleep(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = _make_mqtt_client()

    threads = [
        threading.Thread(target=_stream, args=(LEFT_STREAM_URL,  "left",  client), daemon=True),
        threading.Thread(target=_stream, args=(RIGHT_STREAM_URL, "right", client), daemon=True),
    ]
    for t in threads:
        t.start()

    print("stream_preview running — publishing raw frames to detections/video/left and detections/video/right")

    # Keep the main thread alive; threads are daemon so Ctrl-C exits cleanly.
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("stream_preview stopped")