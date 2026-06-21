import asyncio
import base64
import json
import os
import subprocess
import cv2
import paho.mqtt.client as mqtt

from ultralytics import YOLO

RIGHT_STREAM = os.getenv(
    "VIDEO_STREAM_URL",
    "http://192.168.77.100/axis-cgi/mjpg/video.cgi?resolution=1920x1080&compression=25&camera=1"
)
RIGHT_IMAGE = os.path.join(os.path.dirname(__file__), "image_right.jpg")
RIGHT_IMAGE_URL = os.getenv(
    "RIGHT_IMAGE_URL",
    "http://192.168.77.100/axis-cgi/jpg/image.cgi"
)
RIGHT_IMAGE_USER = os.getenv("RIGHT_IMAGE_USER", "root")
RIGHT_IMAGE_PASSWORD = os.getenv("RIGHT_IMAGE_PASSWORD", "mahi1234")
MQTT_BROKER = "mqtt"
MQTT_PORT = "1883"
MQTT_TOPIC = os.getenv(
    "MQTT_TOPIC",
    "mahi/v1/sense-3C6D66019257/detections/video/left"
)

model = YOLO("../models/white_buoy_yolo11s.pt")


def fetch_right_image() -> cv2.Mat | None:
    command = [
        "curl",
        "-v",
        "--digest",
        "-u",
        f"{RIGHT_IMAGE_USER}:{RIGHT_IMAGE_PASSWORD}",
        RIGHT_IMAGE_URL,
        "-o",
        "image_right.jpg",
    ]

    subprocess.run(
        command,
        cwd=os.path.dirname(__file__),
        check=True,
    )

    return cv2.imread(RIGHT_IMAGE)


async def main():
    # client = mqtt.Client()
    # client.connect(MQTT_BROKER, MQTT_PORT, 60)
    # client.loop_start()

    while True:

        frame = fetch_right_image()

        if frame is None:
            continue

        result = model(frame)[0]

        annotated = result.plot()

        annotated = cv2.resize(
            annotated,
            (640, 360)
        )

        _, jpg = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 40]
        )

        payload = {
            "frame": base64.b64encode(jpg).decode()
        }

        # client.publish(
        #     MQTT_TOPIC,
        #     json.dumps(payload),
        #     qos=0
        # )

        print("Publish")

        await asyncio.sleep(0.2)  # 5 FPS


asyncio.run(main())