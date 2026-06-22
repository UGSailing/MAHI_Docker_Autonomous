"""
camera.py

RTSP capture + YOLO detection for the left/right cameras. Each camera runs
a reader thread (pulls frames off RTSP, always keeping only the latest one
so processing delay can't accumulate) and a worker thread (runs YOLO,
computes 3-D buoy coordinates, and publishes the annotated frame plus GPS
coordinates via post_mqtt.

Coordinate system (camera-local, before heading rotation):
    x  – horizontal,  positive = starboard
    y  – depth,       positive = ahead of the boat
    z  – vertical,    positive = above the camera plane

Timing note: the boat position (lat, lon, heading) is snapshotted in the
reader thread at the moment the frame is received from the RTSP stream.
That snapshot travels with the frame through LatestFrameBox so that
coordinate conversion always uses the position that matches when the image
was captured, not the (potentially later) position after YOLO finishes.
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

RIGHT_STREAM_HOST     = os.getenv("RIGHT_STREAM_HOST",     "192.168.77.100")
RIGHT_STREAM_USER     = os.getenv("RIGHT_STREAM_USER",     "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")
RIGHT_STREAM_PATH     = os.getenv("RIGHT_STREAM_PATH",     "/axis-media/media.amp?camera=1")

LEFT_STREAM_HOST      = os.getenv("LEFT_STREAM_HOST",      "192.168.77.99")
LEFT_STREAM_USER      = os.getenv("LEFT_STREAM_USER",      "root")
LEFT_STREAM_PASSWORD  = os.getenv("LEFT_STREAM_PASSWORD",  "mahi1234")
LEFT_STREAM_PATH      = os.getenv("LEFT_STREAM_PATH",      "/axis-media/media.amp?camera=1")

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
BUOY_RADIUS = float(os.getenv("BUOY_RADIUS", "0.275"))   # 0.55 m diameter / 2

# Distance between the two cameras (metres, centre-to-centre).
CAMERA_SEPARATION = float(os.getenv("CAMERA_SEPARATION", "0.545"))   # s

# Distance from both cameras to the boat's reference point (metres).
CAMERA_TO_BOAT_CENTRE = float(os.getenv("CAMERA_TO_BOAT_CENTRE", "1.5"))   # r

# Optional yaw offsets so each camera can be mounted slightly off-axis.
# Positive = rotated to starboard, negative = to port.
ANGLE_OFFSET_LEFT  = math.radians(float(os.getenv("ANGLE_OFFSET_LEFT",  "0")))
ANGLE_OFFSET_RIGHT = math.radians(float(os.getenv("ANGLE_OFFSET_RIGHT", "0")))

# How close (metres, in boat-local XY) a new detection must be to an
# existing known buoy to be considered the same object.
BUOY_MATCH_DISTANCE = float(os.getenv("BUOY_MATCH_DISTANCE", "1.0"))

# ---------------------------------------------------------------------------
# YOLO model (shared across both worker threads, protected by a lock)
# ---------------------------------------------------------------------------

model      = YOLO(MODEL_PATH)
model_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Known-buoy list (shared across both worker threads, protected by a lock)
# ---------------------------------------------------------------------------
# Each entry: (lat, lon).  Updated in place by both workers when a detection
# is close enough to an existing buoy, or appended when it's a new one.
# The list and its lock are module-level so process_pair() can access them
# from either worker thread without extra plumbing.

buoy_list: list[tuple[float, float]] = []
buoy_list_lock = threading.Lock()


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
    # Single-frame buffer: if inference is slower than the camera frame-rate
    # OpenCV would otherwise queue frames internally and hand back stale ones.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def stream_frames(url: str):
    """Yield decoded frames from an RTSP stream, raising on read failure."""
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
# Single-slot mailbox
# ---------------------------------------------------------------------------
# Carries a (frame, boat_position) pair so that the boat position is always
# the one that was current when the frame was grabbed, not when inference
# finishes.  This is the key fix for the timing issue: the reader thread
# snapshots get_mqtt.get_boat_position() immediately after capture.read(),
# before any processing delay can accumulate.

FrameWithPos = tuple[np.ndarray, dict]   # (frame, BoatPosition dict)


class LatestFrameBox:
    """Producer overwrites; consumer always gets the newest (frame, pos) pair."""

    def __init__(self):
        self._cond  = threading.Condition()
        self._item: FrameWithPos | None = None

    def put(self, item: FrameWithPos) -> None:
        with self._cond:
            self._item = item
            self._cond.notify()

    def get(self) -> FrameWithPos:
        with self._cond:
            while self._item is None:
                self._cond.wait()
            item, self._item = self._item, None
            return item


# ---------------------------------------------------------------------------
# Geometry: depth + 3-D localisation
# ---------------------------------------------------------------------------

def _calculate_depth(apparent_radius_px: float, frame_width_px: int) -> float | None:
    """
    Estimate the distance from the camera to the buoy centre (metres).

    Pinhole model:
        focal_length_px = (frame_width_px / 2) / tan(FOV_H / 2)
        depth           = BUOY_RADIUS * focal_length_px / apparent_radius_px
    """
    if apparent_radius_px <= 0:
        return None
    focal_length_px = (frame_width_px / 2) / _TAN_H
    return BUOY_RADIUS * focal_length_px / apparent_radius_px


def _calculate_3d_coords(
    cx_px:          float,
    cy_px:          float,
    depth_m:        float,
    frame_width_px:  int,
    frame_height_px: int,
) -> tuple[float, float, float]:
    """
    Convert a pixel centre + depth into camera-local 3-D coordinates (metres).

    Returns (x, y, z):
        x – starboard-positive horizontal offset from the camera
        y – depth (perpendicular distance from the camera plane)
        z – upward-positive vertical offset from the camera
    """
    metres_width  = 2 * depth_m * _TAN_H
    metres_height = 2 * depth_m * _TAN_V

    x = (cx_px - frame_width_px  / 2) * metres_width  / frame_width_px
    z = -(cy_px - frame_height_px / 2) * metres_height / frame_height_px
    return x, depth_m, z


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_378_137.0


def viewer_offset_lat_lon(
    lat:         float,
    lon:         float,
    heading_deg: float,
    x:           float,
    y:           float,
) -> tuple[float, float]:
    """
    Given a viewer at (lat, lon) facing heading_deg (0° = north, clockwise),
    and an object at offset (x, y) in the viewer's local frame
    (y = metres straight ahead, x = metres to the right / starboard),
    return the (lat, lon) of the object.
    """
    distance_m            = math.hypot(x, y)
    relative_bearing_rad  = math.atan2(x, y)
    bearing_rad           = math.radians(heading_deg) + relative_bearing_rad

    lat_rad      = math.radians(lat)
    delta_north  = distance_m * math.cos(bearing_rad)
    delta_east   = distance_m * math.sin(bearing_rad)

    new_lat = lat + math.degrees(delta_north / _EARTH_RADIUS_M)
    new_lon = lon + math.degrees(delta_east  / (_EARTH_RADIUS_M * math.cos(lat_rad)))
    return new_lat, new_lon


def lat_lon_to_viewer_xy(
    viewer_lat:  float,
    viewer_lon:  float,
    heading_deg: float,
    obj_lat:     float,
    obj_lon:     float,
) -> tuple[float, float]:
    """
    Inverse of viewer_offset_lat_lon.

    Given a viewer at (viewer_lat, viewer_lon) facing heading_deg and an
    object at (obj_lat, obj_lon), return (x, y) in the viewer's local frame:
    y = metres straight ahead, x = metres to starboard.
    """
    lat_rad      = math.radians(viewer_lat)
    delta_lat    = math.radians(obj_lat - viewer_lat)
    delta_lon    = math.radians(obj_lon - viewer_lon)

    delta_north  = delta_lat * _EARTH_RADIUS_M
    delta_east   = delta_lon * _EARTH_RADIUS_M * math.cos(lat_rad)

    distance_m   = math.hypot(delta_east, delta_north)
    bearing_deg  = math.degrees(math.atan2(delta_east, delta_north))

    rel_bearing_rad = math.radians(bearing_deg - heading_deg)
    x = distance_m * math.sin(rel_bearing_rad)
    y = distance_m * math.cos(rel_bearing_rad)
    return x, y


# ---------------------------------------------------------------------------
# Per-camera detection + buoy-list update
# ---------------------------------------------------------------------------

def _process_cam(
    frame:       np.ndarray,
    result,
    side:        str,
    boat_pos:    dict,
    distance_allowed: float = BUOY_MATCH_DISTANCE,
) -> None:
    """
    For every YOLO detection in *result*:
      1. Estimate depth and compute camera-local 3-D coordinates.
      2. Apply camera-to-boat-centre offsets.
      3. Convert to GPS.
      4. If close enough to a known buoy, update that buoy's position;
         otherwise append it as a new buoy.
      5. Publish coordinates via post_mqtt.

    Mutates the module-level *buoy_list* under *buoy_list_lock*.
    """
    # Guard: no position fix or no heading yet.
    if boat_pos is None:
        return
    if result.boxes is None or len(result.boxes) == 0:
        return

    # Safely extract numeric values from the BoatPosition dict.
    latitude  = float(boat_pos["latitude"])
    longitude = float(boat_pos["longitude"])
    heading   = boat_pos["heading"]
    if heading is None:
        return
    heading = float(heading)

    frame_h, frame_w = frame.shape[:2]

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        apparent_radius_px = abs(x2 - x1) / 2
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2

        depth = _calculate_depth(apparent_radius_px, frame_w)
        if depth is None:
            continue

        x, y, _z = _calculate_3d_coords(cx_px, cy_px, depth, frame_w, frame_h)

        # Shift from camera-lens origin to boat centre.
        # Left camera is to port  → add half-separation to move right to centre.
        # Right camera is to starboard → subtract.
        if side == "left":
            x += CAMERA_SEPARATION / 2
        else:
            x -= CAMERA_SEPARATION / 2

        # Cameras are mounted forward of the boat's reference point.
        y += CAMERA_TO_BOAT_CENTRE

        obj_lat, obj_lon = viewer_offset_lat_lon(latitude, longitude, heading, x, y)

        # Update existing buoy or append a new one.
        with buoy_list_lock:
            best_idx  = None
            best_dist = distance_allowed

            for i, (b_lat, b_lon) in enumerate(buoy_list):
                bx, by = lat_lon_to_viewer_xy(latitude, longitude, heading, b_lat, b_lon)
                dist   = math.hypot(x - bx, y - by)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i

            if best_idx is not None:
                buoy_list[best_idx] = (obj_lat, obj_lon)
            else:
                buoy_list.append((obj_lat, obj_lon))

        post_mqtt.publish_detection_coordinates(side, obj_lat, obj_lon)


# ---------------------------------------------------------------------------
# Thread bodies
# ---------------------------------------------------------------------------

def reader_thread(url: str, box: LatestFrameBox) -> None:
    """
    Continuously read frames from *url*.

    The boat position is snapshotted here, immediately after each
    capture.read(), so it reflects the moment the frame was captured rather
    than the moment inference completes.

    Frames are only queued when a valid position fix (including heading) is
    available; frames captured before GPS lock are silently dropped.
    """
    while True:
        try:
            for frame in stream_frames(url):
                boat_pos = get_mqtt.get_boat_position()
                # Only queue when we have a fix and a heading; worker_thread
                # guards too, but skipping here avoids unnecessary inference.
                if boat_pos is not None and boat_pos["heading"] is not None:
                    box.put((frame, boat_pos))
        except Exception as error:   # noqa: BLE001 – keep the reader alive
            print(f"Stream read error ({url}): {error}")
            time.sleep(1)


def worker_thread(box: LatestFrameBox, side: str) -> None:
    """
    Pull the latest (frame, boat_pos) pair, run YOLO, update the buoy list,
    and publish the annotated preview frame.
    """
    while True:
        frame, boat_pos = box.get()

        with model_lock:
            result = model(frame)[0]

        _process_cam(frame, result, side, boat_pos)

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
    left_box  = LatestFrameBox()

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