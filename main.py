import cv2
import numpy as np
import dlib
import mediapipe as mp

import time
import random

from open_mouth import detectOpenMouth
from eye_tracker import detectEyesPosition
from head_pose import calculateHeadAngle
from detect_prohibited_items import ItemDetector
from detect_occlusion import OcclusionDetector

cap = cv2.VideoCapture(0)
#face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
item_detector = ItemDetector('best.pt')
occlusion_detector = OcclusionDetector('occlussion_classify_best.pt')

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)

start = time.time()
with FaceDetector.create_from_options(options) as detector:

    while True:
        _, frame = cap.read()
        #item_detector.detect(frame)
        
        frame_h, frame_w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        faces = detector.detect_for_video(mp_image, int((time.time() - start)*1000)).detections

        if len(faces)==0:
            cv2.putText(frame, 'No person found', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        else:
            for face in faces:
                box = face.bounding_box
                x1, y1, w, h = int(box.origin_x), int(box.origin_y), int(box.width), int(box.height)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

                #detect occlusion
                crop_face = frame[y1:y2, x1:x2]
                if occlusion_detector.detect(crop_face):
                    cv2.putText(frame, 'Face is occluded', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                #cv2.imwrite(f'face/{random.random()}.jpg', crop_face)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_detector(gray)
        # for face in faces:
        #     cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0))
        #     landmarks = landmark_predictor(gray, face)

        #     x_angle, y_angle = calculateHeadAngle(frame, landmarks)
        #     text = ''
        #     if x_angle > 25:
        #         text += 'Looking right'
        #     elif x_angle < -30:
        #         text += 'Looking left'
        #     if y_angle > 0 and y_angle < 165:
        #         text += 'Looking up'
        #     elif y_angle < 0 and y_angle > -172:
        #         text += 'Looking down'
        #     if text!='': #if head is not straight
        #         cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        #     else: #if head is straight, detect eye direction
        #         eyes_pos = detectEyesPosition(gray, landmarks)
        #         if eyes_pos>=1:
        #             if eyes_pos==1:
        #                 text = 'Looking right'
        #             elif eyes_pos==2:
        #                 text = 'Looking left'
        #             cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        #     if detectOpenMouth(landmarks):
        #         cv2.putText(frame, f'Mouth open', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()