import json
import logging
import math
import threading
import time
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

MQTT_PORT = 1883
MQTT_BROKER = "172.17.0.1"
MQTT_CLIENT_ID = "gnss-check"

GNSS_LEFT_TOPIC = "sense-3C6D66019257/gnss/Left/pvt"
GNSS_RIGHT_TOPIC = "sense-3C6D66019257/gnss/Right/pvt"

# Tolerance for treating coordinates as "zero" (null island / no-fix sentinel)
_ZERO_COORD_EPS = 1e-9
_LISTEN_SECONDS = 2.0
_CONNECT_TIMEOUT_SECONDS = 5.0


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.gnss_fix_left: Optional[dict] = None
        self.gnss_fix_right: Optional[dict] = None


_state = _State()


def mock_check() -> bool:
    print("Not implemented yet but the code continues in order to check the statemachine")
    print("waiting 1 second")
    time.sleep(1)
    return True


def _is_zero_coordinate(latitude: float, longitude: float) -> bool:
    """
    Returns True if the coordinate is effectively (0, 0), the classic
    sentinel value emitted by GNSS receivers that have no real fix
    (e.g. before first fix, after signal loss, or on some firmware bugs
    even when FixIsValid is incorrectly set).
    """
    return math.isclose(latitude, 0.0, abs_tol=_ZERO_COORD_EPS) and math.isclose(
        longitude, 0.0, abs_tol=_ZERO_COORD_EPS
    )


def _extract_lat_lon(fix: Optional[dict]) -> Optional[tuple[float, float]]:
    if not fix or not fix.get("FixIsValid"):
        return None
    lat_lon = (fix.get("Position") or {}).get("LatLon") or {}
    latitude = lat_lon.get("Latitude")
    longitude = lat_lon.get("Longitude")
    if latitude is None or longitude is None:
        return None
    if _is_zero_coordinate(latitude, longitude):
        # FixIsValid lied to us (or the receiver hasn't actually locked yet)
        return None
    return latitude, longitude


def _extract_lat_lon_with_context(
    fix: Optional[dict], topic: str
) -> Optional[tuple[float, float]]:
    """
    Same as _extract_lat_lon but logs *why* a fix was rejected, tagged
    with the originating MQTT topic/side, so failures are diagnosable
    instead of silently swallowed.
    """
    side = (
        "Left"
        if topic == GNSS_LEFT_TOPIC
        else "Right"
        if topic == GNSS_RIGHT_TOPIC
        else "Unknown"
    )

    if not fix:
        logger.debug("GNSS[%s] (%s): no fix payload received", side, topic)
        return None

    if not fix.get("FixIsValid"):
        logger.debug("GNSS[%s] (%s): FixIsValid is False", side, topic)
        return None

    lat_lon = (fix.get("Position") or {}).get("LatLon") or {}
    latitude = lat_lon.get("Latitude")
    longitude = lat_lon.get("Longitude")

    if latitude is None or longitude is None:
        logger.warning(
            "GNSS[%s] (%s): FixIsValid=True but Latitude/Longitude missing", side, topic
        )
        return None

    if _is_zero_coordinate(latitude, longitude):
        logger.warning(
            "GNSS[%s] (%s): FixIsValid=True but coordinates are (0, 0) "
            "-> treating as no fix",
            side,
            topic,
        )
        return None

    return latitude, longitude


def _on_message(_client: mqtt.Client, _userdata, message: mqtt.MQTTMessage) -> None:
    if message.topic not in (GNSS_LEFT_TOPIC, GNSS_RIGHT_TOPIC):
        return

    try:
        fix = json.loads(message.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("GNSS (%s): failed to decode payload: %s", message.topic, exc)
        return

    fix["_mqtt_topic"] = message.topic
    fix["_mqtt_qos"] = message.qos
    fix["_mqtt_retain"] = message.retain
    fix["_received_at"] = time.time()

    coords = _extract_lat_lon_with_context(fix, message.topic)
    fix["_gnss_working"] = coords is not None

    with _state.lock:
        if message.topic == GNSS_LEFT_TOPIC:
            _state.gnss_fix_left = fix
        else:
            _state.gnss_fix_right = fix


def check() -> bool:
    connected = threading.Event()

    def on_connect(client: mqtt.Client, _userdata, _flags, rc: int) -> None:
        if rc != 0:
            logger.error("MQTT connect failed with result code %s", rc)
            return
        client.subscribe(GNSS_LEFT_TOPIC, qos=0)
        client.subscribe(GNSS_RIGHT_TOPIC, qos=0)
        connected.set()

    with _state.lock:
        _state.gnss_fix_left = None
        _state.gnss_fix_right = None

    client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
    client.on_connect = on_connect
    client.on_message = _on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
        client.loop_start()

        if not connected.wait(timeout=_CONNECT_TIMEOUT_SECONDS):
            logger.error("Timed out waiting for MQTT connection to %s:%s", MQTT_BROKER, MQTT_PORT)
            return False

        time.sleep(_LISTEN_SECONDS)

        with _state.lock:
            left = _extract_lat_lon(_state.gnss_fix_left)
            right = _extract_lat_lon(_state.gnss_fix_right)
    finally:
        client.loop_stop()
        client.disconnect()

    ok = left is not None and right is not None
    print(f"GNSS check: left={left}, right={right}")
    print("pass" if ok else "fail")
    return ok
