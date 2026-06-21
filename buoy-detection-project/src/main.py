import os
import threading
import time
import urllib.parse

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# RTSP target stream. Axis's default RTSP path is /axis-media/media.amp.
# Adjust resolution/fps/codec via query params if you want (see Axis VAPIX
# RTSP streaming docs), or just set those on the camera's own stream
# profile config instead.
RIGHT_STREAM_HOST = os.getenv("RIGHT_STREAM_HOST", "192.168.77.100")
RIGHT_STREAM_USER = os.getenv("RIGHT_STREAM_USER", "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")
RIGHT_STREAM_PATH = os.getenv("RIGHT_STREAM_PATH", "/axis-media/media.amp?camera=1")

LEFT_STREAM_HOST = os.getenv("LEFT_STREAM_HOST", "192.168.77.99")
LEFT_STREAM_USER = os.getenv("LEFT_STREAM_USER", "root")
LEFT_STREAM_PASSWORD = os.getenv("LEFT_STREAM_PASSWORD", "mahi1234")
LEFT_STREAM_PATH = os.getenv("LEFT_STREAM_PATH", "/axis-media/media.amp?camera=1")


def build_rtsp_url(host: str, user: str, password: str, path: str) -> str:
    # Credentials are URL-encoded in case the password has special chars.
    user_enc = urllib.parse.quote(user, safe="")
    pass_enc = urllib.parse.quote(password, safe="")
    return f"rtsp://{user_enc}:{pass_enc}@{host}{path}"


RIGHT_STREAM_URL = build_rtsp_url(
    RIGHT_STREAM_HOST, RIGHT_STREAM_USER, RIGHT_STREAM_PASSWORD, RIGHT_STREAM_PATH
)
LEFT_STREAM_URL = build_rtsp_url(
    LEFT_STREAM_HOST, LEFT_STREAM_USER, LEFT_STREAM_PASSWORD, LEFT_STREAM_PATH
)

MQTT_HOST = os.getenv("MQTT_HOST", "192.168.0.190")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")  # optional
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")  # optional
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "buoy-detector")

TOPIC_LEFT = "detections/video/left"
TOPIC_RIGHT = "detections/video/right"

model = YOLO("../models/white_buoy_yolo11s.pt")
model_lock = threading.Lock()


def build_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    # Keep the internal outgoing queue tiny: we never want stale frames
    # sitting around waiting to be sent. If the broker can't keep up we'd
    # rather drop frames at the application level (see LatestFrameBox below).
    client.max_queued_messages_set(1)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()  # background network thread, non-blocking publish()
    return client


mqtt_client = build_mqtt_client()


def open_rtsp_capture(url: str) -> cv2.VideoCapture:
    # FFMPEG backend handles RTSP/H.264 natively, decoding frames for us
    # instead of us hand-parsing MJPEG multipart bodies.
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Keep OpenCV's internal buffer to a single frame. Without this, if our
    # consumer (inference) is ever slower than the stream's frame rate,
    # OpenCV happily queues frames internally and reads start returning
    # increasingly stale ones -- the exact "growing latency" problem we're
    # trying to avoid. A buffer size of 1 forces grab()/read() to always
    # hand back the most recent decoded frame.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def stream_frames(url: str):
    capture = open_rtsp_capture(url)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {url}")

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("RTSP read failed (stream likely dropped)")
            yield frame
    finally:
        capture.release()


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


def reader_thread(url: str, box: LatestFrameBox) -> None:
    while True:
        try:
            for frame in stream_frames(url):
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
            args=(RIGHT_STREAM_URL, right_box),
            daemon=True,
        ),
        threading.Thread(
            target=reader_thread,
            args=(LEFT_STREAM_URL, left_box),
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