import cv2
import numpy as np

LEFT_EYE = (33, 468, 133)
RIGHT_EYE = (362, 473, 263)

def detectEyesPosition(h, w, landmarks):
    left_ratio = getRatio(h, w, landmarks, LEFT_EYE)
    right_ratio = getRatio(h, w, landmarks, RIGHT_EYE)

    ratio = (left_ratio + right_ratio) / 2
    if ratio < 0.35:
        return 1 #left
    elif ratio > 0.65:
        return 2 #right
    else:
        return 0
    
def getRatio(frame_h, frame_w, landmarks, indexes):
    left_idx, center_idx, right_idx = indexes
    left, center, right = landmarks[left_idx], landmarks[center_idx], landmarks[right_idx]
    left = np.array([left.x * frame_w, left.y * frame_h])
    center = np.array([center.x * frame_w, center.y * frame_h])
    right = np.array([right.x * frame_w, right.y * frame_h])

    l2r = np.linalg.norm(right-left)
    l2c = np.linalg.norm(center-left)

    return l2c / l2r

