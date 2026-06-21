"""
get_mqtt.py

Background MQTT subscriber that keeps the latest boat telemetry in memory
and exposes it through simple getter functions, e.g.:

    from get_mqtt import get_boat_position, get_rpm, get_angle

    position = get_boat_position()
    if position is not None:
        print(position["latitude"], position["longitude"])

A single background thread (started lazily on first use) maintains the
connection and updates state as messages arrive; the getters just read the
latest cached value, so calling them is cheap and never blocks on network
I/O. This mirrors the parsing logic in the React dashboard's App.tsx
(GNSS fix + CAN heartbeat/angle frames) so both sides agree on topics and
byte layouts.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import threading
from typing import Optional, TypedDict

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")  # optional
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")  # optional
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "get-mqtt-client")

GNSS_LEFT_TOPIC = "sense-3C6D66019257/gnss/Left/pvt"
GNSS_RIGHT_TOPIC = "sense-3C6D66019257/gnss/Right/pvt"
HEADING_TOPIC = "sense-3C6D66019257/nmea/Left"
CAN_TOPIC = "can/ugent/tx"

ECU_HEARTBEAT_CAN_ID = 0x11
SET_ANGLE_CAN_ID = 0x204


class BoatPosition(TypedDict):
    latitude: float
    longitude: float
    heading: Optional[float]
    height: Optional[float]
    accuracy_horizontal: Optional[float]
    fix_valid: bool
    sats_in_use: int
    sats_in_view: int


class BoatVelocity(TypedDict):
    north: float
    east: float
    down: float
    forward_speed: float  # hypot(north, east), in m/s


class _State:
    """Holds the latest known values. All access goes through the lock so
    the MQTT thread (writer) and caller threads (readers) never race.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.gnss_fix_left: Optional[dict] = None
        self.gnss_fix_right: Optional[dict] = None
        self.heading: Optional[float] = None
        self.rpm: Optional[float] = None
        self.angle: Optional[float] = None
        self.temperature: Optional[float] = None
        self.humidity: Optional[float] = None


_state = _State()
_client: Optional[mqtt.Client] = None
_client_lock = threading.Lock()


def _decode_can_data(data_b64: str) -> Optional[bytes]:
    try:
        return base64.b64decode(data_b64)
    except (ValueError, TypeError):
        return None


def _parse_ecu_heartbeat(data: bytes) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(data) < 6:
        return None, None, None
    # byte 1: temperature offset by -50, byte 2: humidity, bytes 3-4: rpm (int16 LE)
    temperature = data[1] - 50
    humidity = data[2]
    rpm = struct.unpack_from("<h", data, 3)[0]
    return temperature, humidity, rpm


def _parse_set_angle(data: bytes) -> Optional[float]:
    if len(data) < 4:
        return None
    raw_angle = struct.unpack_from("<h", data, 2)[0]
    return (raw_angle / 1000) * 45


def _parse_gphdt_heading(payload: str) -> Optional[float]:
    """Parse a $GPHDT NMEA sentence, e.g. '$GPHDT,269.21,T*0B', and return
    the true heading in degrees. The heading bus can carry other sentence
    types too, so this only returns a value when the line is actually a
    GPHDT sentence; anything else is ignored.
    """
    for line in payload.splitlines():
        line = line.strip()
        if not line.startswith("$GPHDT"):
            continue

        # Strip checksum suffix ("*0B") if present, then split fields.
        sentence = line.split("*", 1)[0]
        fields = sentence.split(",")
        # fields: ['$GPHDT', '<heading>', 'T']
        if len(fields) < 2:
            continue
        try:
            return float(fields[1])
        except ValueError:
            continue
    return None


def _on_message(_client: mqtt.Client, _userdata, message: mqtt.MQTTMessage) -> None:
    if message.topic == GNSS_LEFT_TOPIC or message.topic == GNSS_RIGHT_TOPIC:
        try:
            fix = json.loads(message.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        with _state.lock:
            if message.topic == GNSS_LEFT_TOPIC:
                _state.gnss_fix_left = fix
            else:
                _state.gnss_fix_right = fix
        return

    if message.topic == HEADING_TOPIC:
        try:
            payload = message.payload.decode("ascii", errors="ignore")
        except UnicodeDecodeError:
            return
        heading = _parse_gphdt_heading(payload)
        if heading is not None:
            with _state.lock:
                _state.heading = heading
        return

    if message.topic == CAN_TOPIC:
        try:
            frame = json.loads(message.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        can_id = frame.get("can_id")
        data_b64 = frame.get("data")
        if can_id is None or not isinstance(data_b64, str):
            return

        data = _decode_can_data(data_b64)
        if data is None:
            return

        if can_id == ECU_HEARTBEAT_CAN_ID:
            temperature, humidity, rpm = _parse_ecu_heartbeat(data)
            with _state.lock:
                if temperature is not None:
                    _state.temperature = temperature
                if humidity is not None:
                    _state.humidity = humidity
                if rpm is not None:
                    _state.rpm = rpm

        elif can_id == SET_ANGLE_CAN_ID:
            angle = _parse_set_angle(data)
            if angle is not None:
                with _state.lock:
                    _state.angle = angle


def _ensure_client_started() -> None:
    global _client
    if _client is not None:
        return

    with _client_lock:
        if _client is not None:
            return

        client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
        if MQTT_USER:
            client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        client.on_message = _on_message

        def on_connect(c: mqtt.Client, _userdata, _flags, _rc) -> None:
            c.subscribe(GNSS_LEFT_TOPIC, qos=0)
            c.subscribe(GNSS_RIGHT_TOPIC, qos=0)
            c.subscribe(HEADING_TOPIC, qos=0)
            c.subscribe(CAN_TOPIC, qos=0)

        client.on_connect = on_connect
        client.reconnect_delay_set(min_delay=1, max_delay=5)
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        client.loop_start()  # background network thread

        _client = client


def _extract_lat_lon(fix: Optional[dict]) -> Optional[tuple[float, float]]:
    if not fix or not fix.get("FixIsValid"):
        return None
    lat_lon = (fix.get("Position") or {}).get("LatLon") or {}
    latitude = lat_lon.get("Latitude")
    longitude = lat_lon.get("Longitude")
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


def get_boat_position() -> Optional[BoatPosition]:
    """Latest position, averaging the left and right GNSS receivers' lat/lon
    (using whichever ones currently have a valid fix), plus the latest
    compass heading. Returns None if neither receiver has a valid fix yet.
    """
    _ensure_client_started()
    with _state.lock:
        fix_left = _state.gnss_fix_left
        fix_right = _state.gnss_fix_right
        heading = _state.heading

    lat_lon_left = _extract_lat_lon(fix_left)
    lat_lon_right = _extract_lat_lon(fix_right)

    points = [p for p in (lat_lon_left, lat_lon_right) if p is not None]
    if not points:
        return None

    latitude = sum(p[0] for p in points) / len(points)
    longitude = sum(p[1] for p in points) / len(points)

    # Height/accuracy/sat counts: report from whichever fix we used (prefer
    # left, then right) since these aren't meaningfully averaged.
    reference_fix = fix_left if lat_lon_left is not None else fix_right
    lat_lon = (reference_fix.get("Position") or {}).get("LatLon") or {}

    return BoatPosition(
        latitude=latitude,
        longitude=longitude,
        heading=heading,
        height=lat_lon.get("Height"),
        accuracy_horizontal=(reference_fix.get("Position") or {}).get("AccuracyHorizontal"),
        fix_valid=True,
        sats_in_use=reference_fix.get("SatsInUse", 0),
        sats_in_view=reference_fix.get("SatsInView", 0),
    )


def get_boat_velocity() -> Optional[BoatVelocity]:
    """Latest velocity vector and derived forward speed, or None if no fix
    with velocity data has been received yet. Uses the left receiver's fix,
    falling back to the right one, since velocity isn't meaningfully
    averaged across two antennas."""
    _ensure_client_started()
    with _state.lock:
        fix = _state.gnss_fix_left or _state.gnss_fix_right

    if not fix:
        return None

    velocity = fix.get("Velocity")
    if not velocity:
        return None

    north = velocity.get("North")
    east = velocity.get("East")
    down = velocity.get("Down")
    if north is None or east is None:
        return None

    forward_speed = (north**2 + east**2) ** 0.5

    return BoatVelocity(
        north=north,
        east=east,
        down=down if down is not None else 0.0,
        forward_speed=forward_speed,
    )


def get_rpm() -> Optional[float]:
    """Latest engine RPM from the ECU heartbeat CAN frame, or None."""
    _ensure_client_started()
    with _state.lock:
        return _state.rpm


def get_angle() -> Optional[float]:
    """Latest set angle (degrees) from the CAN frame, or None."""
    _ensure_client_started()
    with _state.lock:
        return _state.angle


def get_temperature() -> Optional[float]:
    """Latest temperature (°C) from the ECU heartbeat CAN frame, or None."""
    _ensure_client_started()
    with _state.lock:
        return _state.temperature


def get_humidity() -> Optional[float]:
    """Latest humidity (%) from the ECU heartbeat CAN frame, or None."""
    _ensure_client_started()
    with _state.lock:
        return _state.humidity


if __name__ == "__main__":
    import time

    _ensure_client_started()
    print("Listening for telemetry, Ctrl+C to stop...")
    try:
        while True:
            time.sleep(2)
            print(
                "position=", get_boat_position(),
                "velocity=", get_boat_velocity(),
                "rpm=", get_rpm(),
                "angle=", get_angle(),
                "temp=", get_temperature(),
                "humidity=", get_humidity(),
            )
    except KeyboardInterrupt:
        pass