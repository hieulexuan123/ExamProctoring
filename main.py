import cv2
import numpy as np
import dlib

from open_mouth import detectOpenMouth
from eye_tracker import detectEyesPosition
from head_pose import calculateHeadAngle

cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0))
        landmarks = landmark_predictor(gray, face)

        x_angle, y_angle = calculateHeadAngle(frame, landmarks)
        text = ''
        if x_angle > 25:
            text += 'Looking right'
        elif x_angle < -30:
            text += 'Looking left'
        if y_angle > 0 and y_angle < 165:
            text += 'Looking up'
        elif y_angle < 0 and y_angle > -172:
            text += 'Looking down'
        if text!='': #if head is not straight
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        else: #if head is straight, detect eye direction
            eyes_pos = detectEyesPosition(gray, landmarks)
            if eyes_pos>=1:
                if eyes_pos==1:
                    text = 'Looking right'
                elif eyes_pos==2:
                    text = 'Looking left'
                cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        if detectOpenMouth(landmarks):
            cv2.putText(frame, f'Mouth open', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()