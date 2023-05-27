import torch
import numpy as np
import cv2 as cv


def spherical_to_vector(spherical_coords):
    """Convert spherical coordinates to 3d Cartesian vector"""
    spherical_coords = spherical_coords[0]
    theta, phi = spherical_coords[0], spherical_coords[1]

    out = np.empty((1, 3))
    out[:, 0] = np.sin(phi) * np.cos(theta)
    out[:, 1] = np.sin(phi) * np.sin(theta)
    out[:, 2] = np.cos(phi)

    return out


def pitchyaw_to_vector(pitchyaws):
    # Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)

    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])

    return out


radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    # Calculate angular error (via cosine similarity).
    # a: ground truth, b: prediction
    a = spherical_to_vector(a)
    b = spherical_to_vector(b)

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees
