import cv2
import numpy as np
import dlib

from open_mouth import detectOpenMouth
from eye_tracker import detectEyesPosition


def getEyePosition(gray_frame, landmarks, left=True):
    margin = 5
    eye_frame = getEye(gray_frame, landmarks, left, margin)
    h, w = eye_frame.shape[:2]
    iris_center = getIris(eye_frame, thresh=40)
    if iris_center is not None:
        cx, cy = iris_center
        print('Iris center', cx, cy, 'Img w, h', w, h)
        horizontal_ratio = (cx-margin) / (w - 2*margin)
        vertical_ratio = (cy-margin) / (h - 2*margin)
        print(vertical_ratio)
        if horizontal_ratio < 0.35:
            return 1  #look right
        elif horizontal_ratio > 0.65:
            return 2 #look left
        elif vertical_ratio < 0.3:
            return 3 #look up
    return 0 #normal
        
def getEye(gray_frame, landmarks, left=True, margin=5):
    if left:
        positions = [36, 37, 38, 39, 40, 41]
    else:
        positions = [42, 43, 44, 45, 46, 47]
    eye_points = np.array([(landmarks.part(position).x, landmarks.part(position).y) for position in positions])

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(eye_points), 255)
    eye_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
    eye_frame[~mask.astype(bool)] = 255 #change background to white

    #Crop the eye
    min_x = np.min(eye_points[:, 0]) - margin
    min_y = np.min(eye_points[:, 1]) - margin
    max_x = np.max(eye_points[:, 0]) + margin
    max_y = np.max(eye_points[:, 1]) + margin
    eye_frame = eye_frame[min_y:max_y, min_x:max_x]
    
    return eye_frame

def getIris(eye_frame, thresh):
    _, eye_frame = cv2.threshold(eye_frame, thresh, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Eye threshold', eye_frame)
    contours, _ = cv2.findContours(eye_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy
    except:
        pass


cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        eyes_pos = detectEyesPosition(gray, landmarks)
        if eyes_pos>=1:
            if eyes_pos==1:
                text = 'Looking right'
            elif eyes_pos==2:
                text = 'Looking left'
            elif eyes_pos==3:
                text = 'Looking up'
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        if detectOpenMouth(frame, landmarks):
            cv2.putText(frame, f'Mouth open', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()