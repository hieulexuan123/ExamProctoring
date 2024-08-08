import cv2
import numpy as np

def calculateHeadAngle(frame, landmarks):
    positions = [30, 8, 36, 45, 48, 54]
    img_points = np.array([(landmarks.part(position).x, landmarks.part(position).y) for position in positions], dtype='double')

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ], dtype='double')
    
    h, w = frame.shape[:2]
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], 
                            dtype = 'double'
                        )
    dist_coefs = np.zeros((4, 1))
    success, rotation_vect, translation_vect = cv2.solvePnP(model_points, img_points, camera_matrix, dist_coefs)

    nose_end_point_3d = np.array([0, 0, 1000], dtype='double')
    nose_end_point_2d, jacobian = cv2.projectPoints(nose_end_point_3d, rotation_vect, translation_vect, camera_matrix, dist_coefs)
    p1 = (int(img_points[0, 0]), int(img_points[0, 1]))
    p2 = (int(nose_end_point_2d[0, 0, 0]), int(nose_end_point_2d[0, 0, 1]))
    cv2.line(frame, p1, p2, (0, 255, 0))

    rotation_matrix, _ = cv2.Rodrigues(rotation_vect)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    x_angle = angles[1]
    y_angle = angles[0]
    
    return x_angle, y_angle

