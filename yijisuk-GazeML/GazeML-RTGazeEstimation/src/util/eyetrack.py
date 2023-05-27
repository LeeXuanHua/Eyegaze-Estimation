import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [474, 475, 476, 477]

# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [469, 470, 471, 472]

mp_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)


def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]

    mesh_coord = np.array([np.multiply([p.x, p.y], [img_width, img_height]).astype(
        int) for p in results.multi_face_landmarks[0].landmark])

    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    return mesh_coord


def return_landmarks(frame):
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        frame = cv.resize(frame, None, fx=1.5, fy=1.5,
                          interpolation=cv.INTER_CUBIC)
        results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)

            # left_iris = [np.array([mesh_coords[i] for i in LEFT_IRIS], dtype=np.int32)]
            # right_iris = [np.array([mesh_coords[i] for i in RIGHT_IRIS], dtype=np.int32)]

            # (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
            #     mesh_coords[LEFT_IRIS])
            # (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
            #     mesh_coords[RIGHT_IRIS])
            _, l_radius = cv.minEnclosingCircle(
                mesh_coords[LEFT_IRIS])
            _, r_radius = cv.minEnclosingCircle(
                mesh_coords[RIGHT_IRIS])
            # left_center = np.array([l_cx, l_cy], dtype=np.int32)
            # right_center = np.array([r_cx, r_cy], dtype=np.int32)

            l_eyeball_radius = 2*l_radius
            r_eyeball_radius = 2*r_radius

            # left_eye = [np.array([mesh_coords[i] for i in LEFT_EYE], dtype=np.int32)]
            # right_eye = [np.array([mesh_coords[i] for i in RIGHT_EYE], dtype=np.int32)]

            LE = {
                # 'eyelid_landmarks': left_eye,
                # 'iris_landmarks': left_iris,
                # 'iris_centre': left_center,
                'eyeball_radius': l_eyeball_radius
            }

            RE = {
                # 'eyelid_landmarks': right_eye,
                # 'iris_landmarks': right_iris,
                # 'iris_centre': right_center,
                'eyeball_radius': r_eyeball_radius
            }

            return [LE, RE]
