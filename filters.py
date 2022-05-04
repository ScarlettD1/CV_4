import numpy as np
from numba import prange, njit
import cv2 as cv2

import math


@njit(fastmath=True, parallel=True)
def sobel_filter_three_1(origin, adele_new_img):
    # for i in prange(1, origin.shape[0] - 1):
    #     for j in prange(1, origin.shape[1] - 1):
    #         z1 = origin[i - 1, j - 1]
    #         z2 = origin[i, j - 1]
    #         z3 = origin[i + 1, j - 1]
    #
    #         z4 = origin[i - 1, j]
    #         # z5 = origin[x, y]
    #         z6 = origin[i + 1, j]
    #
    #         z7 = origin[i - 1, j + 1]
    #         z8 = origin[i, j + 1]
    #         z9 = origin[i + 1, j + 1]
    #
    #         G_x = z7 + 2 * z8 + z9 - (z1 + 2 * z2 + z3)
    #         G_y = z3 + 2 * z6 + z9 - (z1 + 2 * z4 + z7)
    #
    #         adele_new_img[i, j] = math.sqrt(G_x ** 2 + G_y ** 2)
    # return adele_new_img
    matrix_x = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    matrix_y = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    for i in prange(1, adele_new_img.shape[0] - 1):
        for j in prange(1, adele_new_img.shape[1] - 1):
            # array[0, 0] = origin[i - 1, j - 1]
            # array[0, 1] = origin[i, j - 1]
            # array[0, 2] = origin[i + 1, j - 1]
            #
            # array[1, 0] = origin[i - 1, j]
            # array[1, 1] = origin[i, j]
            # array[1, 2] = origin[i + 1, j]
            #
            # array[2, 0] = origin[i - 1, j + 1]
            # array[2, 1] = origin[i, j + 1]
            # array[2, 2] = origin[i + 1, j + 1]
            array = np.zeros((3, 3), dtype="int")
            array[0, 0] = origin[i - 1, j - 1]
            array[1, 0] = origin[i, j - 1]
            array[2, 0] = origin[i + 1, j - 1]

            array[0, 1] = origin[i - 1, j]
            array[1, 1] = origin[i, j]
            array[2, 1] = origin[i + 1, j]

            array[0, 2] = origin[i - 1, j + 1]
            array[1, 2] = origin[i, j + 1]
            array[2, 2] = origin[i + 1, j + 1]

            g_x = 0
            g_y = 0
            for x in prange(3):
                for y in prange(3):
                    g_x += array[x, y] * matrix_x[x, y]
                    g_y += array[x, y] * matrix_y[x, y]

            g = (g_x ** 2 + g_y ** 2) ** 0.5
            if g > 255:
                g = 255
                adele_new_img[i, j] = g
            else:
                adele_new_img[i, j] = g

    return adele_new_img


def getsobel(x, y, origin):
    # Вычисляем значения яркости пикселей в 3x3 окрестности точки с координатами (x,y)
    z1 = origin[x - 1, y - 1]
    z2 = origin[x, y - 1]
    z3 = origin[x + 1, y - 1]
    z4 = origin[x - 1, y]
    z5 = origin[x, y]
    z6 = origin[x + 1, y]
    z7 = origin[x - 1, y + 1]
    z8 = origin[x, y + 1]
    z9 = origin[x + 1, y + 1]
    # Вычисляем значения частных производных
    G_x = z7 + 2 * z8 + z9 - (z1 + 2 * z2 + z3)
    G_y = z3 + 2 * z6 + z9 - (z1 + 2 * z4 + z7)
    # Возвращаем значение градиента в точке с координатами (x,y)
    return math.sqrt(G_x ** 2 + G_y ** 2)


@njit(fastmath=True, parallel=True)
def sobel_filter_five_1(origin, adele_new_img):
    matrix_x = np.array([[-2, -2, -2, -2, -2],
                         [-1, -1, -1, -1, -1],
                         [0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [2, 2, 2, 2, 2]])

    matrix_y = np.array([[-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2],
                         [-2, -1, 0, 1, 2]])

    for i in prange(2, origin.shape[0] - 2):
        for j in prange(2, origin.shape[1] - 2):
            array = np.zeros((5, 5), dtype="int")

            array[0, 0] = origin[i - 2, j - 2]
            array[1, 0] = origin[i - 1, j - 2]
            array[2, 0] = origin[i, j - 2]
            array[3, 0] = origin[i + 1, j - 2]
            array[4, 0] = origin[i + 2, j - 2]

            array[0, 1] = origin[i - 2, j - 1]
            array[1, 1] = origin[i - 1, j - 1]
            array[2, 1] = origin[i, j - 1]
            array[3, 1] = origin[i + 1, j - 1]
            array[4, 1] = origin[i + 2, j - 1]

            array[0, 2] = origin[i - 2, j]
            array[1, 2] = origin[i - 1, j]
            array[2, 2] = origin[i, j]
            array[3, 2] = origin[i + 1, j]
            array[4, 2] = origin[i + 2, j]

            array[0, 3] = origin[i - 2, j + 1]
            array[1, 3] = origin[i - 1, j + 1]
            array[2, 3] = origin[i, j + 1]
            array[3, 3] = origin[i + 1, j + 1]
            array[4, 3] = origin[i + 2, j + 1]

            array[0, 4] = origin[i - 2, j + 2]
            array[1, 4] = origin[i - 1, j + 2]
            array[2, 4] = origin[i, j + 2]
            array[3, 4] = origin[i + 1, j + 2]
            array[4, 4] = origin[i + 2, j + 2]

            g_x = 0
            g_y = 0
            for x in prange(5):
                for y in prange(5):
                    g_x += array[x, y] * matrix_x[x, y]
                    g_y += array[x, y] * matrix_y[x, y]

            g = (g_x ** 2 + g_y ** 2) ** 0.5
            if g > 255:
                g = 255
                adele_new_img[i, j] = g
            else:
                adele_new_img[i, j] = g
    return adele_new_img


@njit(fastmath=True, parallel=True)
def sobel_filter_seven_1(origin, adele_new_img):
    matrix_x = np.array([[-3, -3, -3, -3, -3, -3, -3],
                         [-2, -2, -2, -2, -2, -2, -2],
                         [-1, -1, -1, -1, -1, -1, -1],
                         [0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [2, 2, 2, 2, 2, 2, 2],
                         [3, 3, 3, 3, 3, 3, 3]])

    matrix_y = np.array([[-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3],
                         [-3, -2, -1, 0, 1, 2, 3]])

    for i in prange(3, origin.shape[0] - 3):
        for j in prange(3, origin.shape[1] - 3):
            array = np.zeros((7, 7), dtype="int")

            array[0, 0] = origin[i - 3, j - 3]
            array[1, 0] = origin[i - 2, j - 3]
            array[2, 0] = origin[i - 1, j - 3]
            array[3, 0] = origin[i, j - 3]
            array[4, 0] = origin[i + 1, j - 3]
            array[5, 0] = origin[i + 2, j - 3]
            array[6, 0] = origin[i + 3, j - 3]

            array[0, 1] = origin[i - 3, j - 2]
            array[1, 1] = origin[i - 2, j - 2]
            array[2, 1] = origin[i - 1, j - 2]
            array[3, 1] = origin[i, j - 2]
            array[4, 1] = origin[i + 1, j - 2]
            array[5, 1] = origin[i + 2, j - 2]
            array[6, 1] = origin[i + 3, j - 2]

            array[0, 2] = origin[i - 3, j - 1]
            array[1, 2] = origin[i - 2, j - 1]
            array[2, 2] = origin[i - 1, j - 1]
            array[3, 2] = origin[i, j - 1]
            array[4, 2] = origin[i + 1, j - 1]
            array[5, 2] = origin[i + 2, j - 1]
            array[6, 2] = origin[i + 3, j - 1]

            array[0, 3] = origin[i - 3, j]
            array[1, 3] = origin[i - 2, j]
            array[2, 3] = origin[i - 1, j]
            array[3, 3] = origin[i, j]
            array[4, 3] = origin[i + 1, j]
            array[5, 3] = origin[i + 2, j]
            array[6, 3] = origin[i + 3, j]

            array[0, 4] = origin[i - 3, j + 1]
            array[1, 4] = origin[i - 2, j + 1]
            array[2, 4] = origin[i - 1, j + 1]
            array[3, 4] = origin[i, j + 1]
            array[4, 4] = origin[i + 1, j + 1]
            array[5, 4] = origin[i + 2, j + 1]
            array[6, 4] = origin[i + 3, j + 1]

            array[0, 5] = origin[i - 3, j + 2]
            array[1, 5] = origin[i - 2, j + 2]
            array[2, 5] = origin[i - 1, j + 2]
            array[3, 5] = origin[i, j + 2]
            array[4, 5] = origin[i + 1, j + 2]
            array[5, 5] = origin[i + 2, j + 2]
            array[6, 5] = origin[i + 3, j + 2]

            array[0, 6] = origin[i - 3, j + 3]
            array[1, 6] = origin[i - 2, j + 3]
            array[2, 6] = origin[i - 1, j + 3]
            array[3, 6] = origin[i, j + 3]
            array[4, 6] = origin[i + 1, j + 3]
            array[5, 6] = origin[i + 2, j + 3]
            array[6, 6] = origin[i + 3, j + 3]

            g_x = 0
            g_y = 0
            for x in prange(7):
                for y in prange(7):
                    g_x += array[x, y] * matrix_x[x, y]
                    g_y += array[x, y] * matrix_y[x, y]

            g = (g_x ** 2 + g_y ** 2) ** 0.5
            if g > 255:
                g = 255
                adele_new_img[i, j] = g
            else:
                adele_new_img[i, j] = g
    return adele_new_img


@njit(fastmath=True, parallel=True)
def histogram_parallel(pixels, size, new_array):
    for i in prange(size - 2):
        avg = np.round((int(pixels[i]) + int(pixels[i + 1]) + int(pixels[i + 2])) / 3)
        new_array[i] = avg
    return new_array


@njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_mean_element(pixels, height, width, size, res_image):
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
                if res_image[i, j][0] - C > 15:
                    res_image[i, j] = 255
                else:
                    res_image[i, j] = 0
    return res_image


@njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_min_max(pixels, height, width, size, res_image):
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
                if res_image[i, j][0] - C > 15:
                    res_image[i, j] = 255
                else:
                    res_image[i, j] = 0
    return res_image


# @njit(fastmath=True, parallel=True)
def adaptive_threshold_parallel_median_3_3(pixels, height, width, res_image):
    for i in prange(1, height - 1):
        for j in prange(1, width - 1):
            arr = [pixels[i, j][0], pixels[i - 1, j][0], pixels[i + 1, j][0], pixels[i, j - 1][0],
                   pixels[i, j + 1][0], pixels[i - 1, j - 1][0],
                   pixels[i + 1, j + 1][0], pixels[i - 1, j + 1][0], pixels[i + 1, j - 1][0]]
            sorted_arr = sorted(arr)
            C = sorted_arr[int(len(sorted_arr) / 2)]
            if int(res_image[i, j][0]) - C > 15:
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
