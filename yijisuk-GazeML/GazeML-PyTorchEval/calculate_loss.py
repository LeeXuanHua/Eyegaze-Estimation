import torch
import numpy as np

from estimate_gaze_from_landmarks import *
from utils import *


def calculate_loss(eye, gt_headpose, gt_gaze, elg_model):
    eye *= 2.0 / 255.0
    eye -= 1.0
    eye = np.expand_dims(eye, -1)
    eye = np.array([eye, eye])
    eye = eye.reshape((2, 1, 36, 60))
    eye = torch.from_numpy(eye)

    heatmaps_predict, ldmks_predict, radius_predict = elg_model(eye)

    ldmks = ldmks_predict.cpu().detach().numpy()
    iris_ldmks = np.array(ldmks[0][0:8])
    iris_center = np.array(ldmks[0][-2])
    eyeball_center = np.array(ldmks[0][-1])
    eyeball_radius = radius_predict.cpu().detach().numpy()[0][0]

    gaze_predict = estimate_gaze_from_landmarks(
        iris_ldmks, iris_center, eyeball_center, eyeball_radius)
    est_gaze = gaze_predict.reshape(1, 2)
    gt_gaze = gt_gaze.reshape(1, 2)
    
    loss = np.mean(angular_error(gt_gaze, est_gaze))

    return loss
