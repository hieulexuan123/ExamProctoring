import cv2
import numpy as np
import dlib
import mediapipe as mp
import face_recognition

import time
import sys
from multiprocessing import Queue, Process, Event

from open_mouth import detectOpenMouth
from eye_tracker import detectEyesPosition
from head_pose import estimateHeadPose
from detect_prohibited_items import ItemDetector
from detect_occlusion import OcclusionDetector
from auth import Authentication

def detectItems(q, stop_flag):
    cap = cv2.VideoCapture(0)
    item_detector = ItemDetector('best.pt')

    while not stop_flag.is_set():
        _, frame = cap.read()        
        frame = cv2.flip(frame, 1)
        
        #Detect unprohibited item
        item_detector.detect(frame)

        q.put(frame)
    cap.release()
    q.cancel_join_thread()
    

def processPeople(q, stop_flag, user):
    occlusion_detector = OcclusionDetector('occlussion_classify_best.pt')

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    landmark_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=2,
        running_mode=VisionRunningMode.VIDEO)

    start = time.time()
    with FaceLandmarker.create_from_options(landmark_options) as landmarker:
        while not stop_flag.is_set():
            #Get the frame that was already processed by ItemDetector
            if q.empty():
                continue
            frame = q.get()

            #Detect faces
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            faces = landmarker.detect_for_video(mp_image, int((time.time() - start)*1000)).face_landmarks
            if len(faces)==0:
                cv2.putText(frame, 'No person found', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            elif len(faces)>1:
                cv2.putText(frame, 'Too many people found', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            else:
                face = faces[0]
                h, w = frame.shape[:2]
                x_min, x_max, y_min, y_max = w, 0, h, 0

                face_2d, face_3d = [], []  #for head pose estimation
                for idx, landmark in enumerate(face):
                    landmark_x, landmark_y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, landmark_x)
                    x_max = max(x_max, landmark_x)
                    y_min = min(y_min, landmark_y)
                    y_max = max(y_max, landmark_y)

                    if idx==33 or idx==263 or idx==1 or idx==61 or idx==291 or idx==199:
                        face_2d.append([landmark_x, landmark_y])
                        face_3d.append([landmark_x, landmark_y, landmark.z])
                
                #Draw bounding box for the face
                x_min, x_max, y_min, y_max = max(x_min, 0), min(x_max, w), max(y_min, 0), min(y_max, h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))

                #Face verification
                # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # crop_face_encoding = face_recognition.face_encodings(rgb_frame, [(y_min, x_max, y_max, x_min)])
                # isMatch = face_recognition.compare_faces(crop_face_encoding, np.array(user['face_encoding']))[0]
                # if isMatch:
                #     label = 'True person'
                #     text_color = (0, 255, 0)
                # else:
                #     label = 'Wrong person'
                #     text_color = (0, 0, 255)
                # label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # label_x = x_min
                # label_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
                # cv2.rectangle(frame, (label_x, label_y - label_size[1] - 2), 
                #             (label_x + label_size[0], label_y + 2), text_color, -1)
                # cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                #             0.5, (255, 255, 255), 1)

                #Head pose
                head_pose = estimateHeadPose(w, (w/2, h/2), face_2d, face_3d)
                if head_pose==1:
                    cv2.putText(frame, 'Looking up', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                elif head_pose==2:
                    cv2.putText(frame, 'Looking down', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                elif head_pose==3:
                    cv2.putText(frame, 'Looking right', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                elif head_pose==4:
                    cv2.putText(frame, 'Looking left', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                else:
                    #Detect occlusion
                    crop_face = frame[y_min:y_max, x_min:x_max]
                    if occlusion_detector.detect(crop_face):
                        cv2.putText(frame, 'Face is occluded', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    else:
                        #Detect open mouth
                        is_mouth_opened = detectOpenMouth(h, w, face)
                        if is_mouth_opened:
                            cv2.putText(frame, 'Mouth is opened', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

                        #Detect eye gaze
                        eyes_pos = detectEyesPosition(h, w, face)
                        if eyes_pos==1:
                            cv2.putText(frame, 'Looking left', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                        elif eyes_pos==2:
                            cv2.putText(frame, 'Looking right', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    
                
                #cv2.imwrite(f'face/{random.random()}.jpg', crop_face)

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) == 27:
                stop_flag.set()
    cv2.destroyAllWindows()

def proctor(user):
    item_queue = Queue()
    stop_flag = Event()

    p1 = Process(target=detectItems, args=(item_queue, stop_flag))
    p2 = Process(target=processPeople, args=(item_queue, stop_flag, user))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

if __name__=='__main__':
    auth = Authentication(proctor)