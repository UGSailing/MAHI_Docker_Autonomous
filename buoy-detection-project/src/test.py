"""
main.py

Entry point: starts RTSP capture + YOLO detection for both cameras,
publishing annotated frames and detection coordinates over MQTT
(see camera.py and set_mqtt.py).
"""

import camera
import post_mqtt


def main() -> None:
    publish_detection_coordinates("1", 51.08237244887396, 2.588432716735549)
    publish_detection_coordinates("2", 51.08243266810585, 2.5884415111765375)


if __name__ == "__main__":
    main()