import cv2
import numpy as np

def estimateHeadPose(focal_length, center, face_2d, face_3d):
    face_2d = np.array(face_2d, dtype='double')
    face_3d = np.array(face_3d, dtype='double')
    
    camera_matrix = np.array(
                            [[focal_length, 0, center[1]],
                            [0, focal_length, center[0]],
                            [0, 0, 1]], 
                            dtype = 'double'
                        )
    dist_coefs = np.zeros((4, 1), dtype='double')

    success, rotation_vect, translation_vect = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_coefs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vect)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    
    if pitch>12:
        return 1 #up
    elif pitch<-10:
        return 2 #down
    if yaw>12:
        return 3 #right
    elif yaw<-12:
        return 4 #left
    return 0

