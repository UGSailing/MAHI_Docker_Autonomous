"""
camera.py

RTSP capture + YOLO detection for the left/right cameras.  Each camera runs
a reader thread (pulls frames off RTSP, always keeping only the latest one
so processing delay can't accumulate).  A single shared worker thread grabs
the latest frame from *both* cameras, runs YOLO on each, then calls
process_pair() to fuse detections across both cameras before updating the
buoy list.  Annotated preview frames and GPS coordinates are published via
post_mqtt; an `updated` flag is also published so external consumers know
when a buoy's history has grown.

Buoy list structure:
    buoy_list[i]  →  [(lat₀, lon₀), (lat₁, lon₁), ...]
    The *first* tuple is the a-priori seed position used for matching.
    Every new confirmed detection is *appended*, so the full history is
    always available.

Coordinate system (camera-local, before heading rotation):
    x  – horizontal,  positive = starboard
    y  – depth,       positive = ahead of the boat
    z  – vertical,    positive = above the camera plane

Timing note: the boat position (lat, lon, heading) is snapshotted in the
reader thread at the moment the frame is received from the RTSP stream.
That snapshot travels with the frame through LatestFrameBox so that
coordinate conversion always uses the position that matches when the image
was captured, not the (potentially later) position after YOLO finishes.


Important note for Claude: don't change anything about this script without asking for explicit permission first
"""

from __future__ import annotations

import math
import os
import threading
import time
import json
import pathlib
import urllib.parse

_file = None

import cv2
import numpy as np
from ultralytics import YOLO

import post_mqtt
import get_mqtt

from config import BUOY_MATCH_DISTANCE


# ---------------------------------------------------------------------------
# hulp functies om detectie history op te slaan
# ---------------------------------------------------------------------------

def open_log(run_dir: str = "runs") -> None:
    global _file
    pathlib.Path(run_dir).mkdir(exist_ok=True)
    fname = f"{run_dir}/detections_{int(time.time())}.jsonl"
    _file = open(fname, "a", buffering=1)   # line-buffered = crash-safe

def log(side: str, x: float, y: float, z: float, depth: float,
        boat_lat: float, boat_lon: float, heading: float,
        obj_lat: float, obj_lon: float) -> None:
    if _file is None:
        return
    _file.write(json.dumps({
        "t": time.time(), "side": side,
        "x_m": x, "y_m": y, "z_m": z, "depth_m": depth,
        "boat_lat": boat_lat, "boat_lon": boat_lon, "heading": heading,
        "lat": obj_lat, "lon": obj_lon
    }) + "\n")

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

MODEL_PATH = os.getenv("MODEL_PATH", "../models/new_buoy_yolo11s.pt")

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

# ---------------------------------------------------------------------------
# YOLO model (shared across both worker threads, protected by a lock)
# ---------------------------------------------------------------------------

model      = YOLO(MODEL_PATH)
model_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Known-buoy list (shared across both worker threads, protected by a lock)
# ---------------------------------------------------------------------------
# Each entry is a list of (lat, lon) tuples representing the full detection
# history for one buoy.  The *first* tuple in each sub-list is the a-priori
# position used for matching; every new detection is *appended* so callers
# can inspect the full history.  Updated in place by both workers under
# buoy_list_lock.
#
# Structure: [ [(lat, lon), (lat, lon), ...],   # buoy 0 history
#              [(lat, lon), ...],                # buoy 1 history
#              ... ]

BuoyHistory = list[tuple[float, float]]
buoy_list: list[BuoyHistory] = []
buoy_list_lock = threading.Lock()

# ---------------------------------------------------------------------------
# On-demand frame snapshots (Thread-safe cache for external calls)
# ---------------------------------------------------------------------------
_latest_snapshots: dict[str, dict[str, np.ndarray | None]] = {
    "left": {"raw": None, "annotated": None},
    "right": {"raw": None, "annotated": None},
}
_snapshot_lock = threading.Lock()


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
    # Force RTSP over TCP so FFmpeg always gets a complete bitstream.
    # UDP is the default but drops packets silently, causing H.264 macroblock
    # decode errors and visual corruption in the streamed frames.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
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
# Per-camera detection pass + pair fusion
# ---------------------------------------------------------------------------

def _process_cam(
    frame:            np.ndarray,
    result,
    side:             str,
    boat_pos:         dict,
    update_list:      list[tuple[float, float] | None],
    buoy_list_meters: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float] | None], list[tuple[float, float, float]]]:
    """
    For every YOLO detection in *result*:
      1. Estimate depth and compute camera-local 3-D coordinates.
      2. Apply camera-to-boat-centre offsets.
      3. Convert to GPS.
      4. For each known buoy slot, track the *closest* detection seen so far
         this pair-frame (update_list / buoy_list_meters are mutated in place).

    Returns the mutated (update_list, buoy_list_meters).

    This function does NOT acquire buoy_list_lock — callers must do so
    before calling and hold it until they have finished using the return values.

    Parameters
    ----------
    update_list:
        One slot per known buoy, initially all None.  When a detection is
        the closest match for slot i, update_list[i] is set to (lat, lon).
    buoy_list_meters:
        One slot per known buoy: (x_buoy, y_buoy, best_distance_so_far).
        best_distance_so_far starts at distance_allowed and is tightened
        each time a closer detection is found.
    """
    if boat_pos is None:
        return update_list, buoy_list_meters
    if result.boxes is None or len(result.boxes) == 0:
        return update_list, buoy_list_meters

    latitude  = float(boat_pos["latitude"])
    longitude = float(boat_pos["longitude"])
    heading   = boat_pos["heading"]
    if heading is None:
        return update_list, buoy_list_meters
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

        log(side, x, y, _z, depth,
            boat_pos["latitude"], boat_pos["longitude"], heading,
            obj_lat, obj_lon)

        # Find the buoy slot whose a-priori position is closest to this
        # detection, and only update if we beat the current best distance.
        for i, (x_buoy, y_buoy, best_dist) in enumerate(buoy_list_meters):
            dist = math.hypot(x - x_buoy, y - y_buoy)
            if dist < best_dist:
                buoy_list_meters[i] = (x_buoy, y_buoy, dist)
                update_list[i] = (obj_lat, obj_lon)

    return update_list, buoy_list_meters


def process_pair(
    boat_pos:         dict,
    buoy_positions:   list[BuoyHistory],
    left_frame:       np.ndarray,
    right_frame:      np.ndarray,
    left_result=None,
    right_result=None,
    distance_allowed: float = BUOY_MATCH_DISTANCE,
) -> tuple[bool, list[BuoyHistory], object, object]:
    """
    Run YOLO on both frames and fuse detections against *buoy_positions*.

    *buoy_positions* is passed in explicitly (rather than using the
    module-level buoy_list) so callers can manage their own history list
    and the function is straightforward to test in isolation.

    For each buoy already in buoy_positions the a-priori position (first
    history entry) is converted to boat-local metres and used as the
    matching anchor.  The best detection (across both cameras) within
    distance_allowed is appended to that buoy's history list.  Detections
    that do not match any existing buoy are appended as new single-entry
    buoy histories.

    All detections for this pair-frame are published together via post_mqtt.

    Parameters
    ----------
    boat_pos:
        Dict with keys 'latitude', 'longitude', 'heading' — as returned by
        get_mqtt.get_boat_position().
    buoy_positions:
        Current buoy history list.  Modified in place AND returned so the
        caller can use either style:
            updated, buoy_positions, lr, rr = process_pair(boat_pos, buoy_positions, ...)
    left_frame, right_frame:
        Raw BGR frames from the left and right cameras.
    left_result, right_result:
        Optional pre-computed YOLO results.  If provided, inference is
        skipped and these results are used directly — callers that already
        ran the model (e.g. for annotation) should pass them in to avoid
        running inference twice.
    distance_allowed:
        Maximum boat-local distance (metres) for a detection to be matched
        to an existing buoy.

    Returns
    -------
    (updated, buoy_positions, left_result, right_result)
        updated        – True if at least one buoy's history was extended.
        buoy_positions – The (mutated) history list, same object as the input.
        left_result    – YOLO result for the left frame (for annotation reuse).
        right_result   – YOLO result for the right frame (for annotation reuse).
    """
    if boat_pos is None:
        return False, buoy_positions, left_result, right_result

    latitude  = float(boat_pos["latitude"])
    longitude = float(boat_pos["longitude"])
    heading   = boat_pos["heading"] or 0
    if heading is None:
        return False, buoy_positions, left_result, right_result
    heading = float(heading)

    # Run YOLO on both frames under the shared model lock only when the
    # caller has not already provided pre-computed results.
    if left_result is None or right_result is None:
        with model_lock:
            if left_result is None:
                left_result  = model(left_frame,  conf=0.5)[0]
            if right_result is None:
                right_result = model(right_frame, conf=0.5)[0]

    # Convert each buoy's a-priori position (first history entry) to
    # boat-local metres so we can compare against pixel-derived distances.
    buoy_list_meters: list[tuple[float, float, float]] = []
    for history in buoy_positions:
        a_priori_lat, a_priori_lon = history[0]
        bx, by = lat_lon_to_viewer_xy(latitude, longitude, heading,
                                      a_priori_lat, a_priori_lon)
        buoy_list_meters.append((bx, by, distance_allowed))

    # One update slot per known buoy, starts empty.
    update_list: list[tuple[float, float] | None] = [None] * len(buoy_positions)

    # Process both cameras; each call may tighten the best-distance for
    # any slot, so the truly closest detection wins across both cameras.
    update_list, buoy_list_meters = _process_cam(
        left_frame, left_result, "left", boat_pos, update_list, buoy_list_meters
    )
    update_list, buoy_list_meters = _process_cam(
        right_frame, right_result, "right", boat_pos, update_list, buoy_list_meters
    )

    # Commit updates: append new readings to matching buoy histories.
    updated = False
    for i, new_pos in enumerate(update_list):
        if new_pos is not None:
            buoy_positions[i].append(new_pos)
            updated = True

    # Publish the full history of every buoy — all positions, all buoys.
    post_mqtt.publish_detection_coordinates([
        {"buoy": i, "latitude": lat, "longitude": lon}
        for i, history in enumerate(buoy_positions)
        for lat, lon in history
    ])

    return updated, buoy_positions, left_result, right_result


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
                # if boat_pos is not None and boat_pos["heading"] is not None:
                box.put((frame, boat_pos))
        except Exception as error:   # noqa: BLE001 – keep the reader alive
            print(f"Stream read error ({url}): {error}")
            time.sleep(1)


def worker_thread(left_box: LatestFrameBox, right_box: LatestFrameBox) -> None:
    """
    Pull the latest (frame, boat_pos) pair from *both* cameras, fuse
    detections via process_pair(), and publish annotated preview frames.

    A single worker thread drives both cameras so that process_pair() always
    sees a temporally-consistent left/right pair, and the best-distance
    matching logic in _process_cam() can compete across both cameras before
    any buoy history is updated.

    YOLO is run exactly once per frame pair: process_pair() returns the
    result objects and they are reused for annotation, avoiding the previous
    double-inference that was driving up CPU/GPU temperature and inference
    latency over time.
    """
    while True:
        print("Worker ran")
        # Block until both boxes have a fresh frame.
        left_frame,  left_pos  = left_box.get()
        right_frame, right_pos = right_box.get()
        print("GOT POSITIONS")

        # Use the left camera's position as the authoritative snapshot for
        # this pair (they are captured within milliseconds of each other).
        boat_pos = left_pos

        # process_pair runs YOLO internally and returns the result objects so
        # we can reuse them for annotation without a second inference pass.
        with buoy_list_lock:
            updated, _, left_result, right_result = process_pair(
                boat_pos, buoy_list, left_frame, right_frame
            )

        # Encode and publish annotated preview frames for both sides,
        # reusing the result objects from process_pair (no second inference).
        for side, frame, result in (
            ("left",  left_frame,  left_result),
            ("right", right_frame, right_result),
        ):
            annotated = result.plot()

            with _snapshot_lock:
                _latest_snapshots[side]["raw"]       = frame.copy()
                _latest_snapshots[side]["annotated"] = annotated.copy()

            annotated = cv2.resize(annotated, (640, 360))
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if ok:
                post_mqtt.publish_video_frame(side, jpg.tobytes())


# ---------------------------------------------------------------------------
# Public API for sporadic frame retrieval
# ---------------------------------------------------------------------------

def get_current_frame(side: str, annotated: bool = True) -> np.ndarray | None:
    """
    Returns the most recent frame processed by the specified camera side ('left' or 'right').
    Safe to call sporadically from external loops, APIs, or UI endpoints.
    
    :param side: 'left' or 'right'
    :param annotated: If True, returns the frame drawn with YOLO bounding boxes.
                      If False, returns the unaltered raw frame.
    :return: A copy of the frame as a numpy array, or None if no frame has run through yet.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
        
    with _snapshot_lock:
        frame_type = "annotated" if annotated else "raw"
        frame = _latest_snapshots[side][frame_type]
        # Return a copy so the calling thread can manipulate it safely
        return frame.copy() if frame is not None else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pprint

    # -----------------------------------------------------------------------
    # Minimal stubs so the test runs without a live MQTT broker or cameras.
    # -----------------------------------------------------------------------
    import types

    # Stub post_mqtt so publish calls are no-ops that just print.
    _post_mqtt_stub = types.ModuleType("post_mqtt")
    _post_mqtt_stub.publish_detection_coordinates = lambda detections: print(
        f"  [post_mqtt] publish_detection_coordinates: {detections}"
    )
    _post_mqtt_stub.publish_buoy_update_flag = lambda flag: print(
        f"  [post_mqtt] publish_buoy_update_flag: {flag}"
    )
    _post_mqtt_stub.publish_video_frame = lambda side, data: None
    sys.modules["post_mqtt"] = _post_mqtt_stub

    # Stub get_mqtt so get_boat_position returns a fixed test position.
    _get_mqtt_stub = types.ModuleType("get_mqtt")
    _get_mqtt_stub.get_boat_position = lambda: {
        "latitude":  50.91479228756449,
        "longitude": 2.689219528227875,
        "heading":   90.0,
    }
    sys.modules["get_mqtt"] = _get_mqtt_stub

    # Re-bind the module-level names so process_pair picks up the stubs.
    import importlib, camera as _self  # noqa: E401
    _self.post_mqtt = _post_mqtt_stub
    _self.get_mqtt  = _get_mqtt_stub

    # -----------------------------------------------------------------------
    # Test fixtures
    # -----------------------------------------------------------------------
    LEFT_IMAGE  = os.getenv("TEST_LEFT_IMAGE",  "buoy_test_2.jpeg")
    RIGHT_IMAGE = os.getenv("TEST_RIGHT_IMAGE", "buoy_test_3.jpeg")

    left_frame = cv2.imread(LEFT_IMAGE)
    if left_frame is None:
        print(f"ERROR: could not read left test image '{LEFT_IMAGE}'")
        sys.exit(1)

    right_frame = cv2.imread(RIGHT_IMAGE)
    if right_frame is None:
        print(f"ERROR: could not read right test image '{RIGHT_IMAGE}'")
        sys.exit(1)

    print(f"Loaded test images: {LEFT_IMAGE} {left_frame.shape}, "
          f"{RIGHT_IMAGE} {right_frame.shape}")

    # Seed buoy_positions with three a-priori known positions (same as the
    # original test script) so we can verify matching works.
    buoy_positions: list[BuoyHistory] = [
        [(50.91480024376357, 2.6892946991623328), (50.91480912186219, 2.6893154862823034)],
        [(50.91480912186219, 2.6893154862823034)],
        [(50.914794230962976, 2.6892700529467866)],
    ]

    print("\n--- Initial buoy_positions ---")
    pprint.pprint(buoy_positions)

    # -----------------------------------------------------------------------
    # Run process_pair exactly as the production loop will call it.
    # -----------------------------------------------------------------------
    boat_pos = _get_mqtt_stub.get_boat_position()
    print(f"\nboat_pos: {boat_pos}")
    print("\n--- Calling process_pair (distance_allowed=2) ---")

    updated, buoy_positions, _, _ = process_pair(
        boat_pos, buoy_positions, left_frame, right_frame, distance_allowed=2
    )

    print(f"\nupdated: {updated}")
    print("\n--- buoy_positions after process_pair ---")
    pprint.pprint(buoy_positions)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_detections = sum(len(h) - 1 for h in buoy_positions
                           if len(h) > 1)
    new_buoys        = sum(1 for h in buoy_positions if len(h) == 1
                           and h[0] not in [
                               (50.91480024376357, 2.6892946991623328),
                               (50.91480912186219, 2.6893154862823034),
                               (50.914794230962976, 2.6892700529467866),
                           ])
    print(f"\nSummary: {len(buoy_positions)} buoys tracked, "
          f"{total_detections} new detection(s) matched to existing buoys, "
          f"{new_buoys} brand-new buoy(s) discovered.")


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
        # Single worker thread drives both cameras so process_pair() always
        # sees a matched left/right pair before updating the buoy list.
        threading.Thread(
            target=worker_thread, args=(left_box, right_box), daemon=True
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()