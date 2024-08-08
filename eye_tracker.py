import cv2
import numpy as np
import dlib

def detectEyesPosition(frame, landmarks):
    left_eye_position = getEyePosition(frame, landmarks, left=True)
    right_eye_position = getEyePosition(frame, landmarks, left=False)

    if left_eye_position == right_eye_position and left_eye_position!=0:
        if left_eye_position==1:
            return 1 #left
        elif left_eye_position==2:
            return 2 #right
        elif left_eye_position==3:
            return 3 #up
    return 0 #normal


def getEyePosition(frame, landmarks, left=True):
    if left:
        positions = [36, 37, 38, 39, 40, 41]
    else:
        positions = [42, 43, 44, 45, 46, 47]
    eye_points = np.array([(landmarks.part(position).x, landmarks.part(position).y) for position in positions])

    eye_center = getEyeCenter(frame, eye_points)

    if eye_center is not None:
        eye_center = np.array(eye_center)
        horizontal_ratio = np.linalg.norm(eye_center-eye_points[0]) / \
                           np.linalg.norm(eye_points[3]-eye_points[0])
        if horizontal_ratio < 0.35:
            return 1  #look right
        elif horizontal_ratio > 0.65:
            return 2 #look left
    return 0 #normal
        
def getEyeCenter(frame, eye_points):
    thresh = 40

    if frame.shape[-1] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()

    mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, eye_points, 255)
    eye_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
    eye_frame[~mask.astype(bool)] = 255 #change background to white

    _, eye_frame = cv2.threshold(eye_frame, thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(eye_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy
    except:
        return None