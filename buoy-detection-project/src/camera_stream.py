"""
stream_preview.py

Lightweight raw video streamer — runs as a completely separate process from
the detection pipeline so the OS can schedule it on a different core.

Opens the left and right RTSP streams, decodes frames, and publishes them
as JPEG over MQTT on the same topics the detection pipeline uses
(detections/video/left and detections/video/right).  No YOLO, no GPS, no
locks — just frames as fast as the network allows.

A single pairing thread grabs one frame from each camera box and publishes
them together so left and right are always in sync.

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

MQTT_HOST     = os.getenv("MQTT_HOST",     "172.17.0.1")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER     = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

VIDEO_TOPICS = {
    "left":  "detections/video/left",
    "right": "detections/video/right",
}

JPEG_QUALITY   = int(os.getenv("PREVIEW_JPEG_QUALITY", "40"))
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
# Single-slot mailbox (same pattern as camera.py LatestFrameBox)
# Always holds only the newest frame — if the consumer is slow, old frames
# are overwritten rather than queued, so latency can never accumulate.
# ---------------------------------------------------------------------------

class LatestFrameBox:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: cv2.typing.MatLike | None = None

    def put(self, frame: cv2.typing.MatLike) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def get(self) -> cv2.typing.MatLike:
        with self._cond:
            while self._frame is None:
                self._cond.wait()
            frame, self._frame = self._frame, None
            return frame


# ---------------------------------------------------------------------------
# MQTT client
# ---------------------------------------------------------------------------

def _make_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id="stream-preview", clean_session=True)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    # Keep the outgoing queue at 1 so stale frames are never sent.
    client.max_queued_messages_set(1)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()
    return client


# ---------------------------------------------------------------------------
# Per-camera reader thread
# ---------------------------------------------------------------------------

def _reader(url: str, box: LatestFrameBox) -> None:
    """Continuously decode frames from *url* and drop them into *box*."""
    while True:
        cap = _open_capture(url)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open RTSP stream: {url}")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("RTSP read failed")
                box.put(frame)
        except Exception as error:  # noqa: BLE001
            print(f"[stream_preview] reader error ({url}): {error} — retrying in 1 s")
        finally:
            cap.release()
        time.sleep(1)


# ---------------------------------------------------------------------------
# Pairing + publish thread
# ---------------------------------------------------------------------------

def _publisher(
    left_box:  LatestFrameBox,
    right_box: LatestFrameBox,
    client:    mqtt.Client,
) -> None:
    """
    Grab the latest frame from both cameras, encode, and publish together.
    Because both boxes always hold only the newest frame, left and right are
    always temporally matched before anything is sent.

    Encoding is done here (not in the reader threads) so the resize+JPEG
    work happens once per published pair, not once per decoded frame.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    target_size   = (PREVIEW_WIDTH, PREVIEW_HEIGHT)

    while True:
        left_frame  = left_box.get()
        right_frame = right_box.get()

        for side, frame in (("left", left_frame), ("right", right_frame)):
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            ok, jpg = cv2.imencode(".jpg", resized, encode_params)
            if ok:
                client.publish(
                    VIDEO_TOPICS[side],
                    payload=jpg.tobytes(),
                    qos=0,
                    retain=False,
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client    = _make_mqtt_client()
    left_box  = LatestFrameBox()
    right_box = LatestFrameBox()

    threads = [
        threading.Thread(target=_reader,    args=(LEFT_STREAM_URL,  left_box),                daemon=True),
        threading.Thread(target=_reader,    args=(RIGHT_STREAM_URL, right_box),               daemon=True),
        threading.Thread(target=_publisher, args=(left_box, right_box, client),               daemon=True),
    ]
    for t in threads:
        t.start()

    print(
        "stream_preview running — publishing raw frames to "
        "detections/video/left and detections/video/right"
    )

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("stream_preview stopped")