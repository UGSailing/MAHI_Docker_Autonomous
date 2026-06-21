import base64
import json
import os
import queue
import re
import threading
import time
import urllib.request

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

RIGHT_STREAM_URL = os.getenv(
    "RIGHT_STREAM_URL",
    "http://192.168.77.100/axis-cgi/mjpg/video.cgi?camera=1&fps=5",
)
RIGHT_STREAM_USER = os.getenv("RIGHT_STREAM_USER", "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")

LEFT_STREAM_URL = os.getenv(
    "LEFT_STREAM_URL",
    "http://192.168.77.99/axis-cgi/mjpg/video.cgi?camera=1&fps=5",
)
LEFT_STREAM_USER = os.getenv("LEFT_STREAM_USER", "root")
LEFT_STREAM_PASSWORD = os.getenv("LEFT_STREAM_PASSWORD", "mahi1234")

MQTT_HOST = os.getenv("MQTT_HOST", "172.17.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")  # optional
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "buoy-detector")

TOPIC_LEFT = "detections/video/left"
TOPIC_RIGHT = "detections/video/right"

model = YOLO("../models/white_buoy_yolo11s.pt")
model_lock = threading.Lock()


def build_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
    # Keep the internal outgoing queue tiny: we never want stale frames
    # sitting around waiting to be sent. If the broker can't keep up we'd
    # rather drop frames at the application level (see FrameBus below).
    client.max_queued_messages_set(1)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()  # background network thread, non-blocking publish()
    return client


mqtt_client = build_mqtt_client()


def decode_frame(frame_bytes: bytes):
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_array, cv2.IMREAD_COLOR)


def stream_frames(url: str, user: str, password: str):
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, url, user, password)
    auth_handler = urllib.request.HTTPDigestAuthHandler(password_manager)
    opener = urllib.request.build_opener(auth_handler)

    while True:
        response = opener.open(url, timeout=15)
        content_type = response.headers.get("Content-Type", "")
        boundary_match = re.search(r'boundary=(?:"?)([^";]+)', content_type)
        boundary = boundary_match.group(1).encode("ascii") if boundary_match else None
        try:
            while True:
                line = response.readline()
                if not line:
                    break
                if boundary is not None and line.strip() != b"--" + boundary:
                    continue

                content_length = None
                while True:
                    header_line = response.readline()
                    if not header_line or header_line in (b"\r\n", b"\n"):
                        break
                    lower_line = header_line.lower()
                    if lower_line.startswith(b"content-length:"):
                        try:
                            content_length = int(header_line.split(b":", 1)[1].strip())
                        except ValueError:
                            content_length = None

                if content_length is None:
                    continue

                frame_bytes = response.read(content_length)
                if len(frame_bytes) != content_length:
                    break
                response.readline()

                frame = decode_frame(frame_bytes)
                if frame is not None:
                    yield frame
        finally:
            response.close()


class LatestFrameBox:
    """Single-slot mailbox: producer overwrites, consumer always gets the
    newest frame and never builds up a backlog. This is what prevents
    growing latency when inference is slower than the camera's frame rate.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def get(self):
        with self._cond:
            while self._frame is None:
                self._cond.wait()
            frame, self._frame = self._frame, None
            return frame


def reader_thread(url: str, user: str, password: str, box: LatestFrameBox) -> None:
    while True:
        try:
            for frame in stream_frames(url, user, password):
                box.put(frame)
        except Exception as error:  # noqa: BLE001 - keep the reader alive
            print(f"Stream read error ({url}): {error}")
            time.sleep(1)


def worker_thread(box: LatestFrameBox, topic: str, label: str) -> None:
    while True:
        frame = box.get()

        with model_lock:
            result = model(frame)[0]

        annotated = result.plot()
        annotated = cv2.resize(annotated, (640, 360))
        ok, jpg = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 40]
        )
        if not ok:
            continue

        publish_frame(topic, label, jpg.tobytes())


def publish_frame(topic: str, label: str, jpg_bytes: bytes) -> None:
    # Publish raw JPEG bytes (not base64/JSON) -> smaller payload, less CPU,
    # and the browser can turn it straight into a Blob. QoS 0 means
    # publish() never blocks waiting for broker acks, so a slow/flaky
    # connection can't cause frames to pile up and add delay.
    info = mqtt_client.publish(topic, payload=jpg_bytes, qos=0, retain=False)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed for {label}: rc={info.rc}")


def main() -> None:
    right_box = LatestFrameBox()
    left_box = LatestFrameBox()

    threads = [
        threading.Thread(
            target=reader_thread,
            args=(RIGHT_STREAM_URL, RIGHT_STREAM_USER, RIGHT_STREAM_PASSWORD, right_box),
            daemon=True,
        ),
        threading.Thread(
            target=reader_thread,
            args=(LEFT_STREAM_URL, LEFT_STREAM_USER, LEFT_STREAM_PASSWORD, left_box),
            daemon=True,
        ),
        threading.Thread(
            target=worker_thread, args=(right_box, TOPIC_RIGHT, "right"), daemon=True
        ),
        threading.Thread(
            target=worker_thread, args=(left_box, TOPIC_LEFT, "left"), daemon=True
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()