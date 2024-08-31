import cv2
import numpy as np
import mediapipe as mp
import face_recognition

import time
import multiprocessing
import functools

from open_mouth import detectOpenMouth
from eye_tracker import detectEyesPosition
from head_pose import estimateHeadPose
from detect_prohibited_items import ItemDetector
from detect_occlusion import OcclusionDetector
from auth import Authentication
from fas import FAS
from util import FPSCounter

def processItems():
    global frame_array, shape, item_detector
    
    with frame_array.get_lock():  # Ensure exclusive access
        frame = np.frombuffer(frame_array.get_obj(), dtype=np.uint8).reshape(shape)
    
    # Detect items
    item_detector.detect(frame)
    
    with frame_array.get_lock():  # Ensure exclusive access
        np.copyto(np.frombuffer(frame_array.get_obj(), dtype=np.uint8), frame.flatten())

def processFaces(face_data):
    global frame_array, shape, start, occlusion_detector, landmarker

    with frame_array.get_lock():  # Ensure exclusive access
        frame = np.frombuffer(frame_array.get_obj(), dtype=np.uint8).reshape(shape)
    
    # Process faces
    #Detect faces
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    faces = landmarker.detect_for_video(mp_image, int((time.time() - start)*1000)).face_landmarks
    if len(faces)==0:
        cv2.putText(frame, 'No person found', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        #Update face_data
        face_data['face'] = None
    elif len(faces)>1:
        cv2.putText(frame, 'Too many people found', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        #Update face_data
        face_data['face'] = None
    else:
        face = faces[0]
        h, w = frame.shape[:2]
        x_min, x_max, y_min, y_max = w, 0, h, 0

        face_2d, face_3d = [], []  #for head pose estimation
        #Get the bounding box of face
        for idx, landmark in enumerate(face):
            landmark_x, landmark_y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, landmark_x)
            x_max = max(x_max, landmark_x)
            y_min = min(y_min, landmark_y)
            y_max = max(y_max, landmark_y)

            if idx==33 or idx==263 or idx==1 or idx==61 or idx==291 or idx==199:
                face_2d.append([landmark_x, landmark_y])
                face_3d.append([landmark_x, landmark_y, landmark.z])
        
        #Update face_data
        x_min, x_max, y_min, y_max = max(x_min, 0), min(x_max, w), max(y_min, 0), min(y_max, h)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_data['face'] = rgb_frame[y_min:y_max, x_min:x_max]

        #Draw bounding box for the face
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))

        #Annote whether person is matched or not
        if face_data['is_match'] is not None:
            if face_data['is_match']:
                label = 'True person'
                color = (0, 255, 0)
            else:
                label = 'Wrong person'
                color = (0, 0, 255)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = x_min
            label_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
            cv2.rectangle(frame, (label_x, label_y - label_size[1] - 2), 
                        (label_x + label_size[0], label_y + 2), color, -1)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_face = rgb_frame[y_min:y_max, x_min:x_max]

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
    
    with frame_array.get_lock():  # Ensure exclusive access
        np.copyto(np.frombuffer(frame_array.get_obj(), dtype=np.uint8), frame.flatten())

def smap(f):
    return f()

def init_worker(arg_frame_array, arg_shape):
    global frame_array, shape, item_detector, occlusion_detector, landmarker, start
    frame_array = arg_frame_array
    shape = arg_shape

    item_detector = ItemDetector()
    occlusion_detector = OcclusionDetector()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    landmark_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='model/face_landmarker.task'),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=2,
        running_mode=VisionRunningMode.VIDEO)
    landmarker = FaceLandmarker.create_from_options(landmark_options)

    start = time.time()

def verifyFace(face_data, user):
    spoof_detector = FAS()
    while True:
        face = face_data.get('face')
        
        if face is not None:
            is_fake = spoof_detector.detect(face)
            if is_fake:
                face_data['is_match'] = False #Fake face
            else:
                face_encoding = face_recognition.face_encodings(face, [(0, face.shape[1], face.shape[0], 0)])
                is_match = face_recognition.compare_faces(face_encoding, np.array(user['face_encoding']))[0]
                face_data['is_match'] = is_match
        else:
            face_data['is_match'] = None
            time.sleep(0.1)

def proctor(user):
    cap = cv2.VideoCapture(0)

    #Face data for verification
    manager = multiprocessing.Manager()
    face_data = manager.dict({'face': None, 'is_match': False})

    #Verify face in the seperate process
    verify_process = multiprocessing.Process(target=verifyFace, args=(face_data, user))
    verify_process.start()

    #Shared frame data
    shape = (480, 640, 3)
    frame_array = multiprocessing.Array('B', shape[0]*shape[1]*shape[2], lock=True)
 
    with multiprocessing.Pool(initializer=init_worker, initargs=(frame_array, shape)) as pool:
        fps = 0
        fps_counter = FPSCounter()
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            
            # Copy the frame data to the shared array
            with frame_array.get_lock():
                np.copyto(np.frombuffer(frame_array.get_obj(), dtype=np.uint8), frame.flatten())
            
            # Process the frame with multiprocessing
            func1 = functools.partial(processItems)
            func2 = functools.partial(processFaces, face_data)
            start_time = time.time()
            pool.map(smap, [func1, func2])
            print('Time', (time.time()-start_time)*1000)
            # Convert the shared array back to a frame
            with frame_array.get_lock():
                updated_frame = np.frombuffer(frame_array.get_obj(), dtype=np.uint8).reshape(shape)

            # Calculate fps
            fps_counter.update()
            if fps_counter.frame_count % 5 == 0:
                fps = fps_counter.get_fps()
                fps_counter.reset()
            print('fps', fps)
            cv2.putText(updated_frame,
                        f'FPS: {fps:.2f}', 
                        (500, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 0, 0))
            
            # Display the annotated frame
            cv2.imshow('Annotated Webcam Feed', updated_frame)
            
            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                verify_process.terminate()
                break

if __name__=='__main__':
    auth = Authentication(proctor)