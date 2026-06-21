import asyncio
import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import threading

import cv2
import numpy as np

from ultralytics import YOLO


RIGHT_STREAM_URL = os.getenv(
    "RIGHT_STREAM_URL",
    "http://192.168.77.100/axis-cgi/mjpg/video.cgi?camera=1&fps=5"
)
RIGHT_STREAM_USER = os.getenv("RIGHT_STREAM_USER", "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")
LEFT_STREAM_URL = os.getenv(
    "LEFT_STREAM_URL",
    "http://192.168.77.99/axis-cgi/mjpg/video.cgi?camera=1&fps=5"
)
LEFT_STREAM_USER = os.getenv("LEFT_STREAM_USER", "root")
LEFT_STREAM_PASSWORD = os.getenv("LEFT_STREAM_PASSWORD", "mahi1234")
PUBLISH_URL = os.getenv("PUBLISH_URL", "http://192.168.0.190:9000/publish")

model = YOLO("../models/white_buoy_yolo11s.pt")
model_lock = threading.Lock()


def decode_frame(frame_bytes: bytes) -> cv2.Mat | None:
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


def stream_and_detect(url: str, user: str, password: str, label: str) -> None:
    for frame in stream_frames(url, user, password):
        process_frame(frame, label)


def process_frame(frame: cv2.Mat, label: str) -> None:
    with model_lock:
        result = model(frame)[0]

    annotated = result.plot()

    annotated = cv2.resize(
        annotated,
        (640, 360)
    )

    _, jpg = cv2.imencode(
        ".jpg",
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 40]
    )

    payload = {
        "label": label,
        "frame": base64.b64encode(jpg).decode()
    }

    request = urllib.request.Request(
        PUBLISH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        print(f"Published {label}")
    except urllib.error.URLError as error:
        print(f"Publish failed for {label}: {error}")


async def main():
    await asyncio.gather(
        asyncio.to_thread(
            stream_and_detect,
            RIGHT_STREAM_URL,
            RIGHT_STREAM_USER,
            RIGHT_STREAM_PASSWORD,
            "right",
        ),
        asyncio.to_thread(
            stream_and_detect,
            LEFT_STREAM_URL,
            LEFT_STREAM_USER,
            LEFT_STREAM_PASSWORD,
            "left",
        ),
    )


asyncio.run(main())