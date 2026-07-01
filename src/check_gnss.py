# import json
# import logging
# import math
import time
# from typing import Optional
# import threading
# import paho.mqtt.client as mqtt
# from sympy import true

# logger = logging.getLogger(__name__)
# _state = threading.Lock()
# MQTT_PORT = 1883
# MQTT_BROKER = "127.0.0.1"
# MQTT_TOPIC_TX = "can/ugent/tx"

# GNSS_LEFT_TOPIC  = "sense-3C6D66019257/gnss/Left/pvt"
# GNSS_RIGHT_TOPIC = "sense-3C6D66019257/gnss/Right/pvt"

# # Tolerance for treating coordinates as "zero" (null island / no-fix sentinel)
# _ZERO_COORD_EPS = 1e-9

def mock_check() -> bool:
    print("Not implemented yet but the code continues in order to check the statemachine")
    print("waiting 1 second")
    time.sleep(1)
    return False

# def _is_zero_coordinate(latitude: float, longitude: float) -> bool:
#     """
#     Returns True if the coordinate is effectively (0, 0), the classic
#     sentinel value emitted by GNSS receivers that have no real fix
#     (e.g. before first fix, after signal loss, or on some firmware bugs
#     even when FixIsValid is incorrectly set).
#     """
#     return math.isclose(latitude, 0.0, abs_tol=_ZERO_COORD_EPS) and math.isclose(
#         longitude, 0.0, abs_tol=_ZERO_COORD_EPS
#     )


# def _extract_lat_lon(fix: Optional[dict]) -> Optional[tuple[float, float]]:
#     if not fix or not fix.get("FixIsValid"):
#         return None
#     lat_lon = (fix.get("Position") or {}).get("LatLon") or {}
#     latitude = lat_lon.get("Latitude")
#     longitude = lat_lon.get("Longitude")
#     if latitude is None or longitude is None:
#         return None
#     if _is_zero_coordinate(latitude, longitude):
#         # FixIsValid lied to us (or the receiver hasn't actually locked yet)
#         return None
#     return latitude, longitude


# def _extract_lat_lon_with_context(
#     fix: Optional[dict], topic: str
# ) -> Optional[tuple[float, float]]:
#     """
#     Same as _extract_lat_lon but logs *why* a fix was rejected, tagged
#     with the originating MQTT topic/side, so failures are diagnosable
#     instead of silently swallowed.
#     """
#     side = "Left" if topic == GNSS_LEFT_TOPIC else "Right" if topic == GNSS_RIGHT_TOPIC else "Unknown"

#     if not fix:
#         logger.debug("GNSS[%s] (%s): no fix payload received", side, topic)
#         return None

#     if not fix.get("FixIsValid"):
#         logger.debug("GNSS[%s] (%s): FixIsValid is False", side, topic)
#         return None

#     lat_lon = (fix.get("Position") or {}).get("LatLon") or {}
#     latitude = lat_lon.get("Latitude")
#     longitude = lat_lon.get("Longitude")

#     if latitude is None or longitude is None:
#         logger.warning(
#             "GNSS[%s] (%s): FixIsValid=True but Latitude/Longitude missing", side, topic
#         )
#         return None

#     if _is_zero_coordinate(latitude, longitude):
#         logger.warning(
#             "GNSS[%s] (%s): FixIsValid=True but coordinates are (0, 0) "
#             "-> treating as no fix",
#             side,
#             topic,
#         )
#         return None

#     return latitude, longitude


# def _on_message(_client: mqtt.Client, _userdata, message: mqtt.MQTTMessage) -> None:
#     if message.topic in (GNSS_LEFT_TOPIC, GNSS_RIGHT_TOPIC):
#         try:
#             fix = json.loads(message.payload.decode("utf-8"))
#         except (json.JSONDecodeError, UnicodeDecodeError) as exc:
#             logger.warning(
#                 "GNSS (%s): failed to decode payload: %s", message.topic, exc
#             )
#             return

#         # Attach MQTT context to the fix so downstream consumers (and the
#         # zero-check / logging above) know where/when this came from.
#         fix["_mqtt_topic"] = message.topic
#         fix["_mqtt_qos"] = message.qos
#         fix["_mqtt_retain"] = message.retain
#         fix["_received_at"] = time.time()

#         coords = _extract_lat_lon_with_context(fix, message.topic)
#         fix["_gnss_working"] = coords is not None

#         with _state:
#             if message.topic == GNSS_LEFT_TOPIC:
#                 _state.gnss_fix_left = fix
#             else:
#                 _state.gnss_fix_right = fix
#         return
# def on_connect(client, userdata, flags, rc):
#     print(f"Connected with result code {rc}")
#     client.subscribe(MQTT_TOPIC_TX)
# def check() -> bool:
#     client2 = mqtt.Client(client_id="control_client")
#     client2.on_message = _on_message
#     client2.subscribe(GNSS_LEFT_TOPIC)
#     client2.subscribe(GNSS_RIGHT_TOPIC)
#     client2.connect(MQTT_BROKER, MQTT_PORT)
#     client2.loop_start()

#     time.sleep(2)  # give it a moment to receive messages

#     with _state.lock:
#         left = _extract_lat_lon(_state.gnss_fix_left)
#         right = _extract_lat_lon(_state.gnss_fix_right)
#     client2.loop_stop()
#     client2.disconnect()
#     if left is not None and right is not None:
#         print("pass")
#         return True
#     else:
#         print("fail")
#         return False

