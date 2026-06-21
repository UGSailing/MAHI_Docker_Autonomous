"""
camera.py

RTSP capture + YOLO detection for the left/right cameras. Each camera runs
a reader thread (pulls frames off RTSP, always keeping only the latest one
so processing delay can't accumulate) and a worker thread (runs YOLO,
computes 3-D buoy coordinates, and publishes the annotated frame plus GPS
coordinates via post_mqtt).

Coordinate system (camera-local, before heading rotation):
    x  – horizontal,  positive = starboard
    y  – depth,       positive = ahead of the boat
    z  – vertical,    positive = above the camera plane
"""

from __future__ import annotations

import math
import os
import threading
import time
import urllib.parse

import cv2
import numpy as np
from ultralytics import YOLO

import post_mqtt
import get_mqtt

# ---------------------------------------------------------------------------
# Stream configuration
# ---------------------------------------------------------------------------

RIGHT_STREAM_HOST = os.getenv("RIGHT_STREAM_HOST", "192.168.77.100")
RIGHT_STREAM_USER = os.getenv("RIGHT_STREAM_USER", "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")
RIGHT_STREAM_PATH = os.getenv("RIGHT_STREAM_PATH", "/axis-media/media.amp?camera=1")

LEFT_STREAM_HOST = os.getenv("LEFT_STREAM_HOST", "192.168.77.99")
LEFT_STREAM_USER = os.getenv("LEFT_STREAM_USER", "root")
LEFT_STREAM_PASSWORD = os.getenv("LEFT_STREAM_PASSWORD", "mahi1234")
LEFT_STREAM_PATH = os.getenv("LEFT_STREAM_PATH", "/axis-media/media.amp?camera=1")

MODEL_PATH = os.getenv("MODEL_PATH", "../models/white_buoy_yolo11s.pt")

# ---------------------------------------------------------------------------
# Camera geometry constants
# ---------------------------------------------------------------------------

# Horizontal / vertical field-of-view (degrees).
CAMERA_FOV_H = float(os.getenv("CAMERA_FOV_H", "105.2"))
CAMERA_FOV_V = float(os.getenv("CAMERA_FOV_V", "62.0"))

_TAN_H = math.tan(math.radians(CAMERA_FOV_H / 2))
_TAN_V = math.tan(math.radians(CAMERA_FOV_V / 2))

# Physical radius of a white buoy (metres).
BUOY_RADIUS = float(os.getenv("BUOY_RADIUS", "0.275"))  # 0.55 m diameter / 2

# Distance between the two cameras (metres, centre-to-centre).
CAMERA_SEPARATION = float(os.getenv("CAMERA_SEPARATION", "0.545"))  # s

# Distance from both cameras to the centre of the boat (metres).
CAMERA_TO_BOAT_CENTRE = float(os.getenv("CAMERA_TO_BOAT_CENTRE", "1.5"))  # r

# Optional yaw offsets so each camera can be mounted slightly off-axis.
# Positive = rotated to starboard, negative = to port.
ANGLE_OFFSET_LEFT = math.radians(float(os.getenv("ANGLE_OFFSET_LEFT", "0")))
ANGLE_OFFSET_RIGHT = math.radians(float(os.getenv("ANGLE_OFFSET_RIGHT", "0")))

# ---------------------------------------------------------------------------
# YOLO model (shared, protected by a lock)
# ---------------------------------------------------------------------------

model = YOLO(MODEL_PATH)
model_lock = threading.Lock()


# ---------------------------------------------------------------------------
# RTSP helpers
# ---------------------------------------------------------------------------

def build_rtsp_url(host: str, user: str, password: str, path: str) -> str:
    """Return a fully-encoded RTSP URL."""
    user_enc = urllib.parse.quote(user, safe="")
    pass_enc = urllib.parse.quote(password, safe="")
    return f"rtsp://{user_enc}:{pass_enc}@{host}{path}"


RIGHT_STREAM_URL = build_rtsp_url(
    RIGHT_STREAM_HOST, RIGHT_STREAM_USER, RIGHT_STREAM_PASSWORD, RIGHT_STREAM_PATH
)
LEFT_STREAM_URL = build_rtsp_url(
    LEFT_STREAM_HOST, LEFT_STREAM_USER, LEFT_STREAM_PASSWORD, LEFT_STREAM_PATH
)


def open_rtsp_capture(url: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Keep OpenCV's internal buffer at one frame so we never serve stale data
    # when inference is slower than the stream frame-rate.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def stream_frames(url: str):
    """Yield decoded frames from an RTSP stream, raising on failure."""
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


# ---------------------------------------------------------------------------
# Single-slot mailbox – prevents latency from accumulating
# ---------------------------------------------------------------------------

class LatestFrameBox:
    """Producer overwrites; consumer always gets the newest frame."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None

    def put(self, frame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def get(self):
        with self._cond:
            while self._frame is None:
                self._cond.wait()
            frame, self._frame = self._frame, None
            return frame


# ---------------------------------------------------------------------------
# Geometry: depth estimation and 3-D localisation
# ---------------------------------------------------------------------------

def _calculate_depth(apparent_radius_px: float, frame_width_px: int) -> float | None:
    """
    Estimate the distance from the camera to the buoy centre (metres).

    Uses the thin-lens / pinhole projection:
        apparent_radius = focal_length * real_radius / depth
        focal_length     = (frame_width / 2) / tan(FOV_H / 2)
    Rearranged: depth = BUOY_RADIUS * focal_length / apparent_radius_px
    """
    if apparent_radius_px <= 0:
        return None
    focal_length_px = (frame_width_px / 2) / _TAN_H
    return BUOY_RADIUS * focal_length_px / apparent_radius_px


def _calculate_3d_coords(
    cx_px: float,
    cy_px: float,
    depth_m: float,
    frame_width_px: int,
    frame_height_px: int,
) -> tuple[float, float, float]:
    """
    Convert a pixel centre + depth into camera-local 3-D coordinates (metres).

    Returns (x, y, z) where:
        x – starboard-positive horizontal offset
        y – depth (distance straight ahead from the camera)
        z – upward-positive vertical offset
    """
    metres_per_px_h = 2 * depth_m * _TAN_H / frame_width_px
    metres_per_px_v = 2 * depth_m * _TAN_V / frame_height_px

    x = (cx_px - frame_width_px / 2) * metres_per_px_h
    z = -(cy_px - frame_height_px / 2) * metres_per_px_v
    return x, depth_m, z


# ---------------------------------------------------------------------------
# Coordinate conversion: camera-local ↔ GPS
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_378_137.0


def _local_xy_to_gps(
    lat: float,
    lon: float,
    heading_deg: float,
    x_m: float,
    y_m: float,
) -> tuple[float, float]:
    """
    Convert a camera-local (x, y) offset into GPS coordinates.

    heading_deg – true heading of the boat (0° = north, clockwise).
    x_m         – starboard offset in metres.
    y_m         – forward offset in metres.
    """
    distance_m = math.hypot(x_m, y_m)
    relative_bearing_rad = math.atan2(x_m, y_m)
    absolute_bearing_rad = math.radians(heading_deg) + relative_bearing_rad

    lat_rad = math.radians(lat)
    delta_north = distance_m * math.cos(absolute_bearing_rad)
    delta_east = distance_m * math.sin(absolute_bearing_rad)

    new_lat = lat + math.degrees(delta_north / _EARTH_RADIUS_M)
    new_lon = lon + math.degrees(delta_east / (_EARTH_RADIUS_M * math.cos(lat_rad)))
    return new_lat, new_lon


# ---------------------------------------------------------------------------
# Per-detection processing
# ---------------------------------------------------------------------------

def _detections_to_gps(
    frame: np.ndarray,
    result,
    side: str,
    boat_lat: float,
    boat_lon: float,
    heading_deg: float,
) -> list[tuple[float, float]]:
    """
    For every bounding box in *result*, compute a GPS position and return
    the list of (lat, lon) pairs.

    *side* is ``"left"`` or ``"right"`` and controls the lateral camera
    offset applied before the heading rotation.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    frame_h, frame_w = frame.shape[:2]
    gps_hits: list[tuple[float, float]] = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        apparent_radius_px = abs(x2 - x1) / 2
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2

        depth = _calculate_depth(apparent_radius_px, frame_w)
        if depth is None:
            continue

        x, y, _z = _calculate_3d_coords(cx_px, cy_px, depth, frame_w, frame_h)

        # Shift x so it is relative to the boat centre rather than the
        # individual camera lens.  Left camera is offset to port (−x),
        # right camera to starboard (+x).
        if side == "left":
            x += CAMERA_SEPARATION / 2
        else:
            x -= CAMERA_SEPARATION / 2

        # Shift y to account for how far forward the cameras sit relative
        # to the declared boat reference point.
        y += CAMERA_TO_BOAT_CENTRE

        buoy_lat, buoy_lon = _local_xy_to_gps(boat_lat, boat_lon, heading_deg, x, y)
        gps_hits.append((buoy_lat, buoy_lon))

    return gps_hits


# ---------------------------------------------------------------------------
# Thread bodies
# ---------------------------------------------------------------------------

def reader_thread(url: str, box: LatestFrameBox) -> None:
    """Continuously read frames from *url* and drop them into *box*."""
    while True:
        try:
            for frame in stream_frames(url):
                box.put(frame)
        except Exception as error:  # noqa: BLE001 – keep the reader alive
            print(f"Stream read error ({url}): {error}")
            time.sleep(1)


def worker_thread(box: LatestFrameBox, side: str) -> None:
    """
    Pull the latest frame, run YOLO, derive GPS coordinates for every
    detected buoy, then publish the annotated JPEG and coordinates.
    """
    while True:
        frame = box.get()

        # Fetch the boat's current position and heading from MQTT.
        boat_lat, boat_lon, heading = get_mqtt.get_boat_position()

        with model_lock:
            result = model(frame)[0]

        # Publish GPS coordinates for every buoy detected in this frame.
        for buoy_lat, buoy_lon in _detections_to_gps(
            frame, result, side, boat_lat, boat_lon, heading
        ):
            post_mqtt.publish_detection_coordinates(side, buoy_lat, buoy_lon)

        # Encode and publish the annotated preview frame.
        annotated = result.plot()
        annotated = cv2.resize(annotated, (640, 360))
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 40])
        if not ok:
            continue
        post_mqtt.publish_video_frame(side, jpg.tobytes())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cameras() -> None:
    right_box = LatestFrameBox()
    left_box = LatestFrameBox()

    threads = [
        threading.Thread(
            target=reader_thread, args=(RIGHT_STREAM_URL, right_box), daemon=True
        ),
        threading.Thread(
            target=reader_thread, args=(LEFT_STREAM_URL, left_box), daemon=True
        ),
        threading.Thread(
            target=worker_thread, args=(right_box, "right"), daemon=True
        ),
        threading.Thread(
            target=worker_thread, args=(left_box, "left"), daemon=True
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()