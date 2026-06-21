import asyncio
import base64
import json
import os
import re
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

import math
import cv2
import numpy as np

from ultralytics import YOLO


camera_angle_horizontal = 105.2  # degrees. volgens specificaties 114
tan_horizontal = np.tan(np.radians(camera_angle_horizontal / 2))
ball_radius = .55 / 2  # radius of witte boei in meter

# laat toe om de camera's in de XY plan te roteren:
# geef dan hier dynamisch de offsets van de cams mee - tussen -90 en 90°, positieve offset = naar rechts, negatief naar links! (Simeon gaat blij zijn)
angle_offset_cam_left = 0
angle_offset_cam_right = 0
angle_offset_cam_left_radians = math.radians(angle_offset_cam_left)
angle_offset_cam_right_radians = math.radians(angle_offset_cam_right)

s = .545 # TODO: PAS AAN (nu gewoon random ingesteld) : separation_distance between cameras in meter



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
PUBLISH_URL = os.getenv("PUBLISH_URL", "http://192.168.0.173:9000/publish")

model = YOLO("../models/white_buoy_yolo11s.pt")
model_lock = threading.Lock()



def calculate_2D_coordinates(M_x, depth_in_meter, pixels_width):
    """
    (M_x)=positie middelpunt in pixels, radius in pixels
    returnt x,y coördinaten van de bal, waarbij assenstelsel als volgt is gedefiniëerd:
    POSITIEVE X = STUURBOORD
    x = horizontale as, met 0 in midden van camera
    POSITIEVE Y = VÓÓR DE BOOT
    y = diepte, loodrechte afstand van bal tot camera
    """
    depth = depth_in_meter
    y = depth
    meters_width = 2 * depth * tan_horizontal
    x = (M_x - pixels_width / 2) * meters_width / pixels_width
    return x, y

def calculate_depth(radius, pixels_width):
    k = ball_radius/(2*tan_horizontal)*pixels_width
    if radius > 0:
        depth = k / radius
        return depth



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


class FrameGrabber:
    """
    Leest continu frames van een MJPEG-stream in een eigen thread en houdt
    alleen het laatst ontvangen frame bij (met timestamp), zodat de
    hoofd-loop altijd de meest recente frame van beide camera's kan pakken
    zonder zelf te hoeven blocken op netwerk-I/O.
    """

    def __init__(self, url: str, user: str, password: str, label: str):
        self.url = url
        self.user = user
        self.password = password
        self.label = label

        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        for frame in stream_frames(self.url, self.user, self.password):
            with self._lock:
                self._frame = frame
                self._timestamp = time.monotonic()

    def get_latest(self):
        with self._lock:
            return self._frame, self._timestamp


def detect_buoy(frame: cv2.Mat):
    """
    Draait YOLO op één frame en geeft (annotated_frame, box) terug.
    box is None als er niets gedetecteerd is.
    """
    with model_lock:
        result = model(frame, conf=0.6)[0]

    annotated = result.plot()

    box = None
    if len(result.boxes) > 0:
        box = result.boxes.xyxy[0].tolist()  # [x1, y1, x2, y2]

    return annotated, box


def publish(label: str, annotated: cv2.Mat) -> None:
    annotated = cv2.resize(annotated, (640, 360))
    _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 40])

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

def annotate_coordinates(frame: cv2.Mat, x: float, y: float) -> None:
    text = f"x: {x:.2f} m  y: {y:.2f} m"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
    )


def process_stereo_pair(left_frame: cv2.Mat, right_frame: cv2.Mat) -> None:
    left_annotated, left_box = detect_buoy(left_frame)
    right_annotated, right_box = detect_buoy(right_frame)

    pixels_width = left_frame.shape[1]  # ga uit van gelijke resolutie L/R

    if left_box is not None:
        x1, _, x2, _ = left_box
        radius_left = abs(x2 - x1) / 2
        M_x_left = (x1 + x2) / 2
    else:
        radius_left = M_x_left = None

    if right_box is not None:
        x1, _, x2, _ = right_box
        radius_right = abs(x2 - x1) / 2
        M_x_right = (x1 + x2) / 2
    else:
        radius_right = M_x_right = None


    x = y = None 
    if M_x_left is not None and M_x_right is not None:
        # beide camera's zien de boei -> gebruik stereo-disparity voor de diepte
        imaginary_depth_horizontal = (pixels_width/2)/tan_horizontal
        angle_left = math.atan2((M_x_left - pixels_width/2), imaginary_depth_horizontal) + angle_offset_cam_left_radians
        angle_right = math.atan2((M_x_right - pixels_width/2), imaginary_depth_horizontal) + angle_offset_cam_right_radians

        try:
            denom = (1 / math.tan(math.pi / 2 - angle_left)) + (1 / math.tan(math.pi / 2 + angle_right))
            depth = s / denom if denom != 0 else None
        except ZeroDivisionError:
            depth = None
        # om dit te testen eens de coordinaten berekenen in assenstelsel van linkercamera:
        x, y = calculate_2D_coordinates(M_x_left,depth,pixels_width)
	
        print(f"[stereo] x: {x}, depth: {y}")
    elif M_x_left is not None:
        # alleen linkerbeeld -> val terug op monoculaire schatting via radius
        depth = calculate_depth(radius_left, pixels_width)
        if depth is not None:
            x, y = calculate_2D_coordinates(M_x_left, depth, pixels_width)
            print(f"[mono-left] x,y: {x}, {y}")

    elif M_x_right is not None:
        depth = calculate_depth(radius_right, pixels_width)
        if depth is not None:
            x, y = calculate_2D_coordinates(M_x_right, depth, pixels_width)
            print(f"[mono-right] x,y: {x}, {y}")

    if x is not None and y is not None:
        annotate_coordinates(left_annotated, x, y)
        annotate_coordinates(right_annotated, x, y)

    publish("left", left_annotated)
    publish("right", right_annotated)


def main():
    left_grabber = FrameGrabber(LEFT_STREAM_URL, LEFT_STREAM_USER, LEFT_STREAM_PASSWORD, "left").start()
    right_grabber = FrameGrabber(RIGHT_STREAM_URL, RIGHT_STREAM_USER, RIGHT_STREAM_PASSWORD, "right").start()

    last_processed_left_ts = 0.0
    last_processed_right_ts = 0.0

    while True:
        left_frame, left_ts = left_grabber.get_latest()
        right_frame, right_ts = right_grabber.get_latest()

        if left_frame is None or right_frame is None:
            time.sleep(0.05)
            continue

        # alleen verwerken als er minstens 1 nieuw frame is t.o.v. de vorige ronde
        if left_ts == last_processed_left_ts and right_ts == last_processed_right_ts:
            time.sleep(0.02)
            continue

        last_processed_left_ts = left_ts
        last_processed_right_ts = right_ts

        process_stereo_pair(left_frame, right_frame)


if __name__ == "__main__":
    main()