import cv2
import numpy as np

def detectOpenMouth(frame, landmarks):
    mouth_start = 48
    mouth_end = 67

    mouth_points = np.array([(landmarks.part(position).x, landmarks.part(position).y) for position in range(mouth_start, mouth_end+1)])

    mouth_h = (
        np.linalg.norm(mouth_points[19] - mouth_points[13]) + 
        np.linalg.norm(mouth_points[17] - mouth_points[15])
        ) / 2
    mouth_w = np.linalg.norm(mouth_points[6] - mouth_points[0])
    ratio = mouth_h / mouth_w

    return ratio > 0.1 

