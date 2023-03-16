import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def show_image(data, mode = {'tool' : 'pyplot', 'draw_mode' : 'auto', 'draw_time' : 50}):
    """
    show each slice of image.

    Args:
        data: narray, a slice of image or kspace
        mode: dict, 'tool' : str, plot by 'pyplot' or 'cv'
                    'draw_mode' : str, 'auto' or 'by_hand'
                    'draw_time' : int, if draw_mode if 'auto', the interval of each frame.
    Returns: None
    """
    if mode['tool'] == 'pyplot':
        plt.imshow(data, cmap='jet', vmin=-100, vmax=100)
        if mode['draw_mode'] == 'auto':
            plt.pause(mode['draw_time'] / 1000)
            plt.clf()
        elif mode['draw_mode'] == 'by_hand':
            plt.show()
    elif mode['tool'] == 'cv':
        if mode['draw_mode'] == 'auto':
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            cv.imshow('show image', data)
            cv.waitKey(mode['draw_time'])
        else:
            cv.waitKey(0)

def generate_2d_mask(mask, shape):
    pass