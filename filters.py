import numpy as np
from matplotlib import pyplot as plt
from numba import prange, njit
import copy
import cv2 as cv2

import math

from scipy.signal import find_peaks


@njit(fastmath=True, parallel=True)
def histogram_parallel(pixels, new_array):
    for j in prange(pixels.shape[0]):
        for i in prange(1, pixels.shape[1] - 1):
            avg = np.round((int(pixels[j][i - 1]) + int(pixels[j][i]) + int(pixels[j][i + 1])) / 3)
            new_array[j][i] = avg
    return new_array


@njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_mean_element(t, pixels, height, width, size, res_image):
    new_size = 0
    while size > 1:
        if size % 2 == 1:
            size -= 2
            new_size += 1
    for i in prange(height):
        for j in prange(width):
            if i < new_size or j < new_size or (i >= (height - new_size)) or (j >= (width - new_size)):
                res_image[i][j] = 0
                pass
            else:
                sum_elements = 0
                count_elements = 0
                for k in prange(-new_size, new_size + 1):
                    for l in prange(-new_size, new_size + 1):
                        sum_elements += pixels[i + k][j + l][0]
                        count_elements += 1
                C = np.round(sum_elements / count_elements)
                if res_image[i, j][0] - C > t:
                    res_image[i, j] = 255
                else:
                    res_image[i, j] = 0
    return res_image


@njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_min_max(t, pixels, height, width, size, res_image):
    new_size = 0
    while size > 1:
        if size % 2 == 1:
            size -= 2
            new_size += 1
    for i in prange(height):
        for j in prange(width):
            if i < new_size or j < new_size or (i >= (height - new_size)) or (j >= (width - new_size)):
                res_image[i][j] = 0
                pass
            else:
                min_element = 300
                max_element = 0
                for k in prange(-new_size, new_size + 1):
                    for l in prange(-new_size, new_size + 1):
                        if max_element < pixels[i + k][j + l][0]:
                            max_element = pixels[i + k][j + l][0]
                        elif min_element > pixels[i + k][j + l][0]:
                            min_element = pixels[i + k][j + l][0]
                C = np.round((max_element + min_element) / 2)
                if res_image[i, j][0] - C > t:
                    res_image[i, j] = 255
                else:
                    res_image[i, j] = 0
    return res_image


# @njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_median_3_3(t, pixels, height, width, res_image):
    for i in prange(1, height - 1):
        for j in prange(1, width - 1):
            arr = [pixels[i, j][0], pixels[i - 1, j][0], pixels[i + 1, j][0], pixels[i, j - 1][0],
                   pixels[i, j + 1][0], pixels[i - 1, j - 1][0],
                   pixels[i + 1, j + 1][0], pixels[i - 1, j + 1][0], pixels[i + 1, j - 1][0]]
            sorted_arr = sorted(arr)
            C = sorted_arr[int(len(sorted_arr) / 2)]
            if int(res_image[i, j][0]) - C > t:
                res_image[i, j] = 255
            else:
                res_image[i, j] = 0
    return res_image


# @njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_median_5_5(pixels, height, width, res_image):
    for i in prange(2, height - 3):
        for j in prange(2, width - 3):
            arr = [pixels[i - 2, j - 2][0], pixels[i - 2, j - 1][0], pixels[i - 2, j][0], pixels[i - 2, j + 1][0],
                   pixels[i - 2, j + 2][0],
                   pixels[i - 1, j - 2][0], pixels[i - 1, j - 1][0], pixels[i - 1, j][0], pixels[i - 1, j + 1][0],
                   pixels[i - 1, j + 2][0],
                   pixels[i, j - 2][0], pixels[i, j - 1][0], pixels[i, j][0], pixels[i, j + 1][0], pixels[i, j + 2][0],
                   pixels[i + 1, j - 2][0], pixels[i + 1, j - 1][0], pixels[i + 1, j][0], pixels[i + 1, j + 1][0],
                   pixels[i + 1, j + 2][0],
                   pixels[i + 2, j - 2][0], pixels[i + 2, j - 1][0], pixels[i + 2, j][0], pixels[i + 2, j + 1][0],
                   pixels[i + 2, j + 2][0]]
            sorted_arr = sorted(arr)
            C = sorted_arr[int(len(sorted_arr) / 2)]
            if int(res_image[i, j][0]) - C > 15:
                res_image[i, j] = 255
            else:
                res_image[i, j] = 0
    return res_image


def secondPeaks(img):
    container = copy.deepcopy(img)
    new = histogram_parallel(container, container)
    new_s = np.asarray(new).ravel()
    fig, ax = plt.subplots()
    ax.hist(new_s, 256, density=True, facecolor='b')
    plt.title('Еще раз сглаженная гистограмма')
    new_peaks, _ = find_peaks(new_s, height=125)
    print(str(new_peaks.size))
    plt.show()
    return new
