from typing import Optional

import struct


_last_temperature_celsius: int = 25

def get_temperature() -> int:
    """Return the most recently parsed temperature value in °C."""
    return _last_temperature_celsius


def publish_temperature(celsius: float) -> None:
    if celsius is None:
        return
    _last_temperature_celsius = int(round(celsius))      # 23.5 °C -> 235
