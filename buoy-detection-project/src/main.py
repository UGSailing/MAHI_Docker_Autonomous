"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import camera


def main() -> None:
    camera.run_cameras()


if __name__ == "__main__":
    main()