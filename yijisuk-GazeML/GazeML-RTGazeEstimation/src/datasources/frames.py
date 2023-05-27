"""Data source of stream of frames."""
import bz2
import dlib
import queue
import shutil
import threading
import time
from typing import Tuple
import os
from urllib.request import urlopen

import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy as np
import tensorflow as tf

from core import BaseDataSource


class FramesSource(BaseDataSource):
    """Preprocessing of stream of frames."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 eye_image_shape: Tuple[int, int],
                 staging: bool = False,
                 **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._eye_image_shape = eye_image_shape
        self._proc_mutex = threading.Lock()
        self._read_mutex = threading.Lock()

        self._frame_read_queue = queue.Queue(maxsize=1)
        self._frame_read_thread = threading.Thread(target=self.frame_read_job, name='frame_read')
        self._frame_read_thread.daemon = True
        self._frame_read_thread.start()

        self._current_index = 0
        self._last_frame_index = 0
        self._indices = []
        self._frames = {}
        self._open = True

        # Call parent class constructor
        super().__init__(tensorflow_session, batch_size=batch_size, num_threads=1,
                         fread_queue_capacity=batch_size, preprocess_queue_capacity=batch_size,
                         shuffle=False, staging=staging, **kwargs)

    _short_name = 'Frames'

    @property
    def short_name(self):
        """Short name specifying source."""
        return self._short_name

    def frame_read_job(self):
        """Read frame from webcam."""
        generate_frame = self.frame_generator()
        while True:
            before_frame_read = time.time()
            bgr = next(generate_frame)
            if bgr is not None:
                after_frame_read = time.time()
                with self._read_mutex:
                    self._frame_read_queue.queue.clear()
                    self._frame_read_queue.put_nowait((before_frame_read, bgr, after_frame_read))
        self._open = False

    def frame_generator(self):
        """Read frame from webcam."""
        raise NotImplementedError('Frames::frame_generator not implemented.')

    def entry_generator(self, yield_just_one=False):
        """Generate eye image entries by detecting faces and facial landmarks."""
        try:
            while range(1) if yield_just_one else True:
                # Grab frame
                with self._proc_mutex:
                    before_frame_read, bgr, after_frame_read = self._frame_read_queue.get()
                    bgr = cv.flip(bgr, flipCode=1)  # Mirror
                    current_index = self._last_frame_index + 1
                    self._last_frame_index = current_index

                    grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                    frame = {
                        'frame_index': current_index,
                        'time': {
                            'before_frame_read': before_frame_read,
                            'after_frame_read': after_frame_read,
                        },
                        'bgr': bgr,
                        'grey': grey,
                    }
                    self._frames[current_index] = frame
                    self._indices.append(current_index)

                    # Keep just a few frames around
                    frames_to_keep = 120
                    if len(self._indices) > frames_to_keep:
                        for index in self._indices[:-frames_to_keep]:
                            del self._frames[index]
                        self._indices = self._indices[-frames_to_keep:]

                # Eye image segmentation pipeline
                self.detect_faces(frame)
                self.detect_landmarks(frame)
                self.calculate_smoothed_landmarks(frame)
                self.segment_eyes(frame)
                self.update_face_boxes(frame)
                frame['time']['after_preprocessing'] = time.time()

                for i, eye_dict in enumerate(frame['eyes']):
                    yield {
                        'frame_index': np.int64(current_index),
                        'eye': eye_dict['image'],
                        'eye_index': np.uint8(i),
                    }

        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Preprocess segmented eye images for use as neural network input."""
        eye = entry['eye']
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if self.data_format == 'NHWC' else 0)
        entry['eye'] = eye
        return entry

    def detect_faces(self, frame):
        """Detect all faces in a frame."""
        frame_index = frame['frame_index']
        previous_index = self._indices[self._indices.index(frame_index) - 1]
        previous_frame = self._frames[previous_index]
        if ('last_face_detect_index' not in previous_frame or
                frame['frame_index'] - previous_frame['last_face_detect_index'] > 59):

            mp_face_detection = mp.solutions.face_detection
            height, width, _ = frame['bgr'].shape
            faces = []

            with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
                results = face_detection.process(cv.cvtColor(frame['bgr'], cv.COLOR_BGR2RGB))

                for detection in results.detections:
                    # TODO: Error handling using try-except
                    bbox = detection.location_data.relative_bounding_box

                    bbox_points = {
                        'xmin': int(bbox.xmin * width),
                        'ymin': int(bbox.ymin * height),
                        'xmax': int(bbox.width * width + bbox.xmin * width),
                        'ymax': int(bbox.height * height + bbox.ymin * height)
                    }

                    xmin, ymin = bbox_points['xmin'], bbox_points['ymin']
                    xmax, ymax = bbox_points['xmax'], bbox_points['ymax']

                    faces.append((xmin, ymin, xmax, ymax))

            # faces.sort(key=lambda bbox: bbox[0])
            frame['faces'] = faces
            frame['last_face_detect_index'] = frame['frame_index']

            # Clear previous known landmarks. This is to disable smoothing when new face detect
            # occurs. This allows for recovery of drifted detections.
            previous_frame['landmarks'] = []
        else:
            frame['faces'] = previous_frame['faces']
            frame['last_face_detect_index'] = previous_frame['last_face_detect_index']

    def detect_landmarks(self, frame):
        mp_face_mesh = mp.solutions.face_mesh
        height, width, _ = frame['bgr'].shape
        frame_landmarks = []

        def tuple_from_coord(coord):
            return (coord[0], coord[1])

        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            landmarks = face_mesh.process(cv.cvtColor(frame['bgr'], cv.COLOR_BGR2RGB))

            if landmarks.multi_face_landmarks:
                for face_landmarks in landmarks.multi_face_landmarks:
                    landmarkCoord = face_landmarks.landmark

                    leftEyesOutPoint, leftEyesInPoint = landmarkCoord[33], landmarkCoord[133]
                    leftEyesOutCoord = _normalized_to_pixel_coordinates(
                        leftEyesOutPoint.x, leftEyesOutPoint.y, width, height)
                    leftEyesInCoord = _normalized_to_pixel_coordinates(
                        leftEyesInPoint.x, leftEyesInPoint.y, width, height)

                    nosePoint = landmarkCoord[94]
                    noseCoord = _normalized_to_pixel_coordinates(
                        nosePoint.x, nosePoint.y, width, height)

                    rightEyesOutPoint, rightEyesInPoint = landmarkCoord[263], landmarkCoord[362]
                    rightEyesOutCoord = _normalized_to_pixel_coordinates(
                        rightEyesOutPoint.x, rightEyesOutPoint.y, width, height)
                    rightEyesInCoord = _normalized_to_pixel_coordinates(
                        rightEyesInPoint.x, rightEyesInPoint.y, width, height)

                    coords = [leftEyesOutCoord, leftEyesInCoord,
                              rightEyesInCoord, rightEyesOutCoord, noseCoord]

                    frame_landmarks.append(np.array([tuple_from_coord(coord) for coord in coords]))

                frame['landmarks'] = frame_landmarks

    _smoothing_window_size = 10
    _smoothing_coefficient_decay = 0.5
    _smoothing_coefficients = None

    def calculate_smoothed_landmarks(self, frame):
        """If there are previous landmark detections, try to smooth current prediction."""
        # Cache coefficients based on defined sliding window size
        if self._smoothing_coefficients is None:
            coefficients = np.power(self._smoothing_coefficient_decay,
                                    list(reversed(list(range(self._smoothing_window_size)))))
            coefficients /= np.sum(coefficients)
            self._smoothing_coefficients = coefficients.reshape(-1, 1)

        # Get a window of frames
        current_index = self._indices.index(frame['frame_index'])
        a = current_index - self._smoothing_window_size + 1
        if a < 0:
            """If slice extends before last known frame."""
            return
        window_indices = self._indices[a:current_index + 1]
        window_frames = [self._frames[idx] for idx in window_indices]
        window_num_landmark_entries = np.array([len(f['landmarks']) for f in window_frames])
        if np.any(window_num_landmark_entries == 0):
            """Any frame has zero faces detected."""
            return
        if not np.all(window_num_landmark_entries == window_num_landmark_entries[0]):
            """Not the same number of faces detected in entire window."""
            return

        # Apply coefficients to landmarks in window
        window_landmarks = np.asarray([f['landmarks'] for f in window_frames])
        frame['smoothed_landmarks'] = np.sum(
            np.multiply(window_landmarks.reshape(self._smoothing_window_size, -1),
                        self._smoothing_coefficients),
            axis=0,
        ).reshape(window_num_landmark_entries[-1], -1, 2)

    def segment_eyes(self, frame):
        """From found landmarks in previous steps, segment eye image."""
        eyes = []

        # Final output dimensions
        oh, ow = self._eye_image_shape

        # Select which landmarks (raw/smoothed) to use
        frame_landmarks = (frame['smoothed_landmarks'] if 'smoothed_landmarks' in frame
                           else frame['landmarks'])

        for face, landmarks in zip(frame['faces'], frame_landmarks):
            # Segment eyes
            # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
            for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
                x1, y1 = landmarks[corner1, :]
                x2, y2 = landmarks[corner2, :]
                eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
                if eye_width == 0.0:
                    continue
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                # Centre image on middle of eye
                translate_mat = np.asmatrix(np.eye(3))
                translate_mat[:2, 2] = [[-cx], [-cy]]
                inv_translate_mat = np.asmatrix(np.eye(3))
                inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

                # Rotate to be upright
                roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
                rotate_mat = np.asmatrix(np.eye(3))
                cos = np.cos(-roll)
                sin = np.sin(-roll)
                rotate_mat[0, 0] = cos
                rotate_mat[0, 1] = -sin
                rotate_mat[1, 0] = sin
                rotate_mat[1, 1] = cos
                inv_rotate_mat = rotate_mat.T

                # Scale
                scale = ow / eye_width
                scale_mat = np.asmatrix(np.eye(3))
                scale_mat[0, 0] = scale_mat[1, 1] = scale
                inv_scale = 1.0 / scale
                inv_scale_mat = np.asmatrix(np.eye(3))
                inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

                # Centre image
                centre_mat = np.asmatrix(np.eye(3))
                centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
                inv_centre_mat = np.asmatrix(np.eye(3))
                inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

                # Get rotated and scaled, and segmented image
                transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
                inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                     inv_centre_mat)
                eye_image = cv.warpAffine(frame['grey'], transform_mat[:2, :], (ow, oh))
                if is_left:
                    eye_image = np.fliplr(eye_image)
                eyes.append({
                    'image': eye_image,
                    'inv_landmarks_transform_mat': inv_transform_mat,
                    'side': 'left' if is_left else 'right',
                })
        frame['eyes'] = eyes

    def update_face_boxes(self, frame):
        """Update face bounding box based on detected landmarks."""
        frame_landmarks = (frame['smoothed_landmarks'] if 'smoothed_landmarks' in frame
                           else frame['landmarks'])
        for i, (face, landmarks) in enumerate(zip(frame['faces'], frame_landmarks)):
            x_min, y_min = np.amin(landmarks, axis=0)
            x_max, y_max = np.amax(landmarks, axis=0)
            x_mid, y_mid = 0.5 * (x_max + x_min), 0.5 * (y_max + y_min)
            w, h = x_max - x_min, y_max - y_min
            new_w = 2.2 * max(h, w)
            half_w = 0.5 * new_w
            frame['faces'][i] = (int(x_mid - half_w), int(y_mid - half_w), int(new_w), int(new_w))


_face_detector = None
_landmarks_predictor = None
