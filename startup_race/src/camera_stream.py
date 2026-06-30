#!/usr/bin/env python3

import json
import time
import uuid
import queue
import signal
import threading
import subprocess
import collections
import sys

import paho.mqtt.client as mqtt

# ============================================================
# CONFIG
# ============================================================

BROKER = "192.168.77.254"
PORT = 1883

OUTPUT_FILE = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

# ============================================================
# GLOBALS
# ============================================================

MY_UUID = str(uuid.uuid4())

CONTROL_TOPIC = f"{MY_UUID}/video/status_external"

discovered_topic = None
camera_name = None

stream_started = False

nal_queue = queue.Queue(maxsize=2000)

parameter_sets = collections.deque(maxlen=20)

running = True

total_written = 0
dropped = 0

# ============================================================
# START FFMPEG
# ============================================================

print(f"[OUTPUT] {OUTPUT_FILE}")

ffmpeg = subprocess.Popen(
    [
        "ffmpeg",
        "-y",

        "-loglevel", "warning",

        # Low latency
        "-fflags", "nobuffer",
        "-flags", "low_delay",

        # Input format
        "-f", "hevc",
        "-i", "-",

        # No audio
        "-an",

        # Copy HEVC directly
        "-c:v", "copy",

        # Prevent corrupted MP4 on interruption
        "-movflags", "+faststart+frag_keyframe+empty_moov",

        OUTPUT_FILE
    ],
    stdin=subprocess.PIPE,
    bufsize=0
)

# ============================================================
# ENABLE CAMERA STREAM
# ============================================================

def enable_stream(client):
    if not camera_name:
        return

    payload = {
        "source": camera_name,
        "periodicdr": True
    }

    client.publish(
        CONTROL_TOPIC,
        json.dumps(payload),
        qos=1
    )

    print(f"[ENABLE] Stream requested for {camera_name}")


# ============================================================
# WRITER THREAD
# ============================================================

def writer_thread():
    global total_written
    global running

    buffer = bytearray()

    while running:

        try:
            nal = nal_queue.get(timeout=0.1)

            if nal is None:
                break

            # Annex-B start code
            buffer.extend(b"\x00\x00\x00\x01")
            buffer.extend(nal)

            total_written += 1

        except queue.Empty:
            pass

        # Flush continuously
        if buffer:

            try:
                ffmpeg.stdin.write(buffer)
                ffmpeg.stdin.flush()
                buffer.clear()

            except BrokenPipeError:
                print("[ERROR] ffmpeg pipe closed")
                running = False
                break

        if total_written % 500 == 0 and total_written > 0:
            print(
                f"[STATS] "
                f"Written={total_written} "
                f"Queue={nal_queue.qsize()} "
                f"Dropped={dropped}"
            )

    # Final flush
    try:
        if buffer:
            ffmpeg.stdin.write(buffer)
            ffmpeg.stdin.flush()
    except:
        pass


writer = threading.Thread(
    target=writer_thread,
    daemon=True
)

writer.start()

# ============================================================
# SEND NAL TO QUEUE
# ============================================================

def send_nal(nal: bytes):
    global dropped

    try:
        nal_queue.put_nowait(nal)

    except queue.Full:
        dropped += 1

        if dropped % 50 == 1:
            print(f"[WARN] Dropped {dropped} frames")


# ============================================================
# MQTT CALLBACKS
# ============================================================

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected ({rc})")

    # Listen for discovery
    client.subscribe("#")


def on_message(client, userdata, msg):
    global discovered_topic
    global camera_name
    global stream_started

    topic = msg.topic
    payload = msg.payload

    # ========================================================
    # DISCOVERY PHASE
    # ========================================================

    if discovered_topic is None:

        if "/video/rtsp/" in topic and "/nal/" in topic:

            discovered_topic = topic

            print(f"[DISCOVER] Video topic:")
            print(discovered_topic)

            parts = topic.split("/")

            try:
                camera_name = parts[3]
            except:
                camera_name = "camera"

            print(f"[DISCOVER] Camera: {camera_name}")

            # Ask camera to stream
            enable_stream(client)

            # Subscribe only to video NAL units
            base = discovered_topic.rsplit("/", 1)[0]

            client.unsubscribe("#")

            client.subscribe(f"{base}/+")

            print(f"[SUBSCRIBE] {base}/+")

            return

    # ========================================================
    # VIDEO DATA
    # ========================================================

    if not discovered_topic:
        return

    base = discovered_topic.rsplit("/", 1)[0]

    if not topic.startswith(base + "/"):
        return

    # Payload too small
    if len(payload) < 12:
        return

    # Remove trailing timestamp
    nal = payload[:-8]

    if len(nal) < 2:
        return

    # NAL type from topic
    try:
        nal_type = int(topic.split("/")[-1])
    except:
        return

    # ========================================================
    # STORE VPS/SPS/PPS
    # ========================================================

    if nal_type in (32, 33, 34):

        if nal not in parameter_sets:
            parameter_sets.append(nal)

    # ========================================================
    # WAIT FOR FIRST IDR FRAME
    # ========================================================

    if not stream_started:

        if nal_type == 20:

            print("[STREAM] First IDR frame")

            # Send VPS/SPS/PPS first
            for p in parameter_sets:
                send_nal(p)

            stream_started = True

        else:
            return

    # ========================================================
    # SEND VIDEO DATA
    # ========================================================

    send_nal(nal)


# ============================================================
# KEEPALIVE
# ============================================================

def keepalive(client):

    while running:

        try:
            enable_stream(client)
        except:
            pass

        time.sleep(30)


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

def shutdown(sig=None, frame=None):
    global running

    if not running:
        return

    print("\n[SHUTDOWN] Finalizing MP4...")

    running = False

    # Stop writer thread
    try:
        nal_queue.put(None)
    except:
        pass

    writer.join(timeout=5)

    # IMPORTANT:
    # Send EOF to ffmpeg
    try:
        ffmpeg.stdin.flush()
    except:
        pass

    try:
        ffmpeg.stdin.close()
    except:
        pass

    # Wait for MP4 finalization
    try:
        ffmpeg.wait(timeout=15)

    except subprocess.TimeoutExpired:
        print("[WARN] ffmpeg did not exit cleanly")
        ffmpeg.kill()

    print(f"[DONE] Saved: {OUTPUT_FILE}")

    print(
        f"[STATS] "
        f"Written={total_written} "
        f"Dropped={dropped}"
    )

    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ============================================================
# MAIN
# ============================================================

def main():

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION1
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[CONNECT] {BROKER}:{PORT}")

    client.connect(BROKER, PORT, 60)

    # Keep stream alive
    threading.Thread(
        target=keepalive,
        args=(client,),
        daemon=True
    ).start()

    client.loop_forever()


if __name__ == "__main__":
    main()