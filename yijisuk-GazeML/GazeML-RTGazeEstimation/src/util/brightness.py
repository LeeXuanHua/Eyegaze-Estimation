import cv2 as cv
from PIL import Image, ImageEnhance
import numpy as np


def increase_brightness(frame, rate):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    enhancer = ImageEnhance.Brightness(frame)
    frame = enhancer.enhance(rate)

    frame = np.array(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    return frame
