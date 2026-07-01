import asyncio
import base64
import json
import os
import re
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

import math
import cv2
import numpy as np

from ultralytics import YOLO


camera_angle_horizontal = 105.2  # degrees. volgens specificaties 114
camera_angle_vertical = 62  # degrees
tan_horizontal = np.tan(np.radians(camera_angle_horizontal / 2))
tan_vertical = np.tan(np.radians(camera_angle_vertical / 2))
ball_radius = .55 / 2  # radius of witte boei in meter

# laat toe om de camera's in de XY plan te roteren:
# geef dan hier dynamisch de offsets van de cams mee - tussen -90 en 90°, positieve offset = naar rechts, negatief naar links! (Simeon gaat blij zijn)
angle_offset_cam_left = 0
angle_offset_cam_right = 0
angle_offset_cam_left_radians = math.radians(angle_offset_cam_left)
angle_offset_cam_right_radians = math.radians(angle_offset_cam_right)

s = .545 # separation_distance between cameras in meter
r = 1.5 # TODO: pas aan : distance van camera's tot midden vd boot


RIGHT_STREAM_URL = os.getenv(
    "RIGHT_STREAM_URL",
    "http://192.168.77.100/axis-cgi/mjpg/video.cgi?camera=1&fps=5"
)
RIGHT_STREAM_USER = os.getenv("RIGHT_STREAM_USER", "root")
RIGHT_STREAM_PASSWORD = os.getenv("RIGHT_STREAM_PASSWORD", "mahi1234")
LEFT_STREAM_URL = os.getenv(
    "LEFT_STREAM_URL",
    "http://192.168.77.99/axis-cgi/mjpg/video.cgi?camera=1&fps=5"
)
LEFT_STREAM_USER = os.getenv("LEFT_STREAM_USER", "root")
LEFT_STREAM_PASSWORD = os.getenv("LEFT_STREAM_PASSWORD", "mahi1234")
PUBLISH_URL = os.getenv("PUBLISH_URL", "http://192.168.0.190:9000/publish")

model = YOLO("buoy_yolo11s.pt")
model_lock = threading.Lock()



def calculate_3D_coordinates(M_x, M_y, depth_in_meter, pixels_width, pixels_height):
    """
    (M_x, M_y)=positie middelpunt in pixels, radius in pixels
    returnt x,y,z coördinaten van de bal, waarbij assenstelsel als volgt is gedefiniëerd:
    POSITIEVE X = STUURBOORD
    x = horizontale as, met 0 in midden van camera
    POSITIEVE Y = VÓÓR DE BOOT
    y = diepte, loodrechte afstand van bal tot camera
    POSITIEVE Z = BOVEN DE BOOT
    z = verticale as, met 0 in midden van camera, en vanuit het perspectief van de camera is meer naar boven = hogere waarden
    """
    depth = depth_in_meter
    y = depth
    meters_width = 2 * depth * tan_horizontal
    meters_height = 2 * depth * tan_vertical
    x = (M_x - pixels_width / 2) * meters_width / pixels_width
    z = -(M_y - pixels_height / 2) * meters_height / pixels_height
    return x, y, z

def calculate_depth(radius, pixels_width):
    k = ball_radius/(2*tan_horizontal)*pixels_width
    if radius > 0:
        depth = k / radius
        return depth



def viewer_offset_lat_lon(lat, lon, heading_deg, x, y):
    """
    Given a viewer at (lat, lon) facing heading_deg (0° = north, clockwise),
    and an object at offset (x, y) in the viewer's local frame
    (y = meters straight ahead, x = meters to the right),
    return the (lat, lon) of the object.
    """
    R = 6378137  # Earth radius in meters

    # distance and bearing of the object relative to the viewer
    distance_m = math.hypot(x, y)
    relative_bearing_rad = math.atan2(x, y)  # 0 when x=0 (straight ahead)
    bearing_deg = heading_deg + math.degrees(relative_bearing_rad)

    bearing_rad = math.radians(bearing_deg)
    lat_rad = math.radians(lat)

    delta_north = distance_m * math.cos(bearing_rad)
    delta_east = distance_m * math.sin(bearing_rad)

    delta_lat = delta_north / R
    delta_lon = delta_east / (R * math.cos(lat_rad))

    new_lat = lat + math.degrees(delta_lat)
    new_lon = lon + math.degrees(delta_lon)

    return new_lat, new_lon



def lat_lon_to_viewer_xy(viewer_lat, viewer_lon, heading_deg, obj_lat, obj_lon):
    """
    Given a viewer at (viewer_lat, viewer_lon) facing heading_deg (0° = north, clockwise),
    and an object at (obj_lat, obj_lon), return (x, y) in the viewer's local frame:
    y = meters straight ahead, x = meters to the right.

    This is the inverse of viewer_offset_lat_lon — same flat-earth approximation,
    so round-tripping through both functions returns (approximately) the original x, y.
    """
    R = 6378137  # Earth radius in meters

    lat_rad = math.radians(viewer_lat)

    delta_lat = math.radians(obj_lat - viewer_lat)
    delta_lon = math.radians(obj_lon - viewer_lon)

    delta_north = delta_lat * R
    delta_east = delta_lon * R * math.cos(lat_rad)

    distance_m = math.hypot(delta_east, delta_north)
    bearing_deg = math.degrees(math.atan2(delta_east, delta_north))  # absolute bearing, 0=N

    relative_bearing_deg = bearing_deg - heading_deg
    relative_bearing_rad = math.radians(relative_bearing_deg)

    x = distance_m * math.sin(relative_bearing_rad)
    y = distance_m * math.cos(relative_bearing_rad)

    return x, y



def process_cam(boat_pos, update_list, buoy_list_meters, frame, left = True):
    latitude, longitude, heading = boat_pos
    pixels_height, pixels_width = frame.shape[:2]

    with model_lock:
        result = model(frame, conf=0.3)[0]


    box = None
    if len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            radius = abs(x2 - x1) / 2
            M_x = (x1 + x2) / 2
            M_y = (y1 + y2) / 2
            depth = calculate_depth(radius, pixels_width)
            if depth is not None:
                x, y, z = calculate_3D_coordinates(M_x, M_y, depth, pixels_width, pixels_height)
                # obv z-coördinaat discarden als die te onrealistisch is
                if left == True:
                    x += s/2
                else:
                    x -= s/2
                y += r
                

                for i, buoy in enumerate(buoy_list_meters):
                    x_buoy, y_buoy, smallest_distance = buoy
                    distance = math.sqrt((x-x_buoy)**2 + (y-y_buoy)**2)
                    if distance < smallest_distance:
                        buoy_list_meters[i] = (x_buoy, y_buoy, distance)
                        obj_lat, obj_lon = viewer_offset_lat_lon(latitude, longitude, heading, x, y)
                        update_list[i] = (obj_lat, obj_lon)
    return update_list, buoy_list_meters




def process_pair(boat_pos, buoy_list, left_frame: cv2.Mat, right_frame: cv2.Mat, distance_allowed = 1) -> None:

    latitude, longitude, heading = boat_pos

    buoy_list_meters = []
    for list in buoy_list:
        tup = list[0] # we pakken enkel de a priori boei-positie hier
        obj_lat, obj_lon = tup
        x, y = lat_lon_to_viewer_xy(latitude, longitude, heading, obj_lat, obj_lon)
        buoy_list_meters += [(x,y,distance_allowed)]


    update_list = [None for _ in range(len(buoy_list))]
    update_list, buoy_list_meters = process_cam(boat_pos, update_list, buoy_list_meters, left_frame, left = True)
    update_list, buoy_list_meters = process_cam(boat_pos, update_list, buoy_list_meters, right_frame, left = False)
    
    print("update list")
    print(update_list)
    
    updated = False
    for i, buoy in enumerate(update_list):
        if buoy is not None:
            buoy_list[i] += [update_list[i]]
            updated = True
    print("buoy list")
    print(buoy_list)
    return updated, buoy_list



boat_pos = (50.91479228756449, 2.689219528227875, 90)
buoy_list = [[(50.91480024376357, 2.6892946991623328),(50.91480912186219, 2.6893154862823034)], [(50.91480912186219, 2.6893154862823034)], [(50.914794230962976, 2.6892700529467866)]]
left_frame = cv2.imread("buoy_test_2.jpeg")
right_frame = cv2.imread("buoy_test_3.jpeg")
updated, buoy_list = process_pair(boat_pos, buoy_list, left_frame, right_frame,1)

print(buoy_list)