import cv2
import numpy as np

def detectOpenMouth(h, w, landmarks):
    upper_lip, lower_lip = landmarks[13], landmarks[14]
    left_lip, right_lip = landmarks[78], landmarks[308]

    upper_lip = np.array([upper_lip.x * w, upper_lip.y *h])
    lower_lip = np.array([lower_lip.x * w, lower_lip.y * h])
    left_lip = np.array([left_lip.x * w, left_lip.y * h])
    right_lip = np.array([right_lip.x * w, right_lip.y * h])

    mouth_h = np.linalg.norm(upper_lip - lower_lip)
    mouth_w = np.linalg.norm(right_lip - left_lip)
    ratio = mouth_h / mouth_w

    return ratio > 0.15 

