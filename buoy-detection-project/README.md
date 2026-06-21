# Buoy Detection Project

This project implements a buoy detection system using a YOLO model to process video streams and publish detection results along with GPS coordinates via MQTT.

## Project Structure

```
buoy-detection-project
├── src
│   └── main.py          # Main application logic
├── models
│   └── buoy.pt         # Pre-trained YOLO model for buoy detection
├── Dockerfile           # Dockerfile for building the application image
├── docker-compose.yml    # Docker Compose configuration for services
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd buoy-detection-project
   ```

2. **Install Docker and Docker Compose:**
   Ensure you have Docker and Docker Compose installed on your machine.

3. **Build the Docker image:**
   ```
   docker-compose build
   ```

4. **Run the application:**
   ```
   docker-compose up
   ```

## Usage

- The application subscribes to the GPS coordinates from the MQTT topic `sense-48B02DF8FC00/gnss/Left/pvt`.
- It processes the video stream from `localhost:1234/video` using the YOLO model located at `./models/buoy.pt`.
- Detection results and the video stream are published to the MQTT topic `/detections/video`.
- The calculated buoy coordinates based on detection diameter and camera offset are published to the topic `/detections/coordinates`.

## Dependencies

The project requires the following Python libraries, which are listed in `requirements.txt`:

- paho-mqtt
- opencv-python
- torch
- torchvision

## License

This project is licensed under the MIT License. See the LICENSE file for details.