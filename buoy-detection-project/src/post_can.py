import post_mqtt

TEMP_CAN_ID = 0x123  # ← your temperature frame's arbitration ID

def publish_temperature(celsius: float) -> None:
    # signed 16-bit, 0.1 °C resolution, little-endian
    raw = int(round(celsius * 10))      # 23.5 °C -> 235
    data = struct.pack("<h", raw)       # -> b"\xeb\x00"
    post_mqtt.publish_can_message(TEMP_CAN_ID, data)