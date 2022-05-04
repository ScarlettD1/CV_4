from tkinter import *
import random
import cv2 as cv2
from PIL import Image, ImageTk, ImageEnhance
import copy
import time
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np
import math
from numba import prange, njit
import threading
import tkinter.ttk as ttk

from scipy.signal import find_peaks
from skimage.feature import canny
from skimage.filters import sobel
from scipy import ndimage as ndi


# from filters import sobel_filter_three_1, getsobel, sobel_filter_five_1, sobel_filter_seven_1, deffOfGaussian, \
#     laplasianOfGaussian
from filters import histogram_parallel, adaptive_threshold_parallel_mean_element, adaptive_threshold_parallel_min_max, \
    adaptive_threshold_parallel_median_3_3, adaptive_threshold_parallel_median_5_5


class App:
    def __init__(self):
        self.root = Tk()
        self.root['bg'] = "#fafafa"
        self.root.title('Lab 4 CV')
        self.root.geometry('1455x695')

        # создаем рабочую область
        self.frame = Frame(self.root)
        self.frame.grid()
        # pixelVirtual = PhotoImage(width=1, height=1)

        self.newImage = 0
        self.image = cv2.imread("pictures/cat.jpg")
        self.imageOrigin = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.imageOriginColored = self.image
        self.readypic = Image.fromarray(self.imageOrigin)

        self.video = cv2.VideoCapture("videos/Road.mp4")

        # Добавим метку 1
        self.photo = ImageTk.PhotoImage(self.readypic)

        # Добавим метку 2
        self.photo_2 = ImageTk.PhotoImage(self.readypic)

        # Добавим метку 3
        self.photo_3 = ImageTk.PhotoImage(self.readypic)

        self.g1 = 0
        self.g2 = 0
        g3 = 0

        # вставляем кнопки T1
        Button(self.frame, text="Вернуть", command=self.picture_origin).grid(row=0, column=0)
        Button(self.frame, text="canny", command=self.canny).grid(row=1, column=0)
        Button(self.frame, text="kmean4", command=self.kmean4).grid(row=2, column=0)
        Button(self.frame, text="adaptive_threshold", command=self.adaptive_threshold).grid(row=3, column=0)

        # вставляем кнопки T2
        Button(self.frame, text="Вернуть", command=self.picture_origin_2).grid(row=0, column=1)
        Button(self.frame, text="P-tile", command=self.Ptile).grid(row=1, column=1)
        Button(self.frame, text="kmean8", command=self.kmean8).grid(row=2, column=1)

        # вставляем кнопки T3
        Button(self.frame, text="Вернуть", command=self.picture_origin_3).grid(row=0, column=3, columnspan=2)
        Button(self.frame, text="following", command=self.following).grid(row=1, column=3, columnspan=2)
        Button(self.frame, text="kmean20", command=self.kmean20).grid(row=2, column=3, columnspan=2)
        Button(self.frame, text="originHist", command=self.buildHistogramm).grid(row=3, column=3, columnspan=2)
        Button(self.frame, text="smoothGist", command=self.peaks).grid(row=4, column=3, columnspan=2)

        # Buttons for video
        # Button(self.frame, text="Показать", command=self.video_origin).grid(row=0, column=5, columnspan=2)

        # Добавим изображение T1
        self.canvas = Canvas(self.root, height=640, width=480)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=2, column=0)

        # Добавим изображение T2
        self.canvas_2 = Canvas(self.root, height=640, width=480)
        self.n_image = self.canvas_2.create_image(0, 0, anchor='nw', image=self.photo_2)
        self.canvas_2.grid(row=2, column=1)

        # Добавим изображение T3
        self.canvas_3 = Canvas(self.root, height=640, width=480)
        self.a_image = self.canvas_3.create_image(0, 0, anchor='nw', image=self.photo_3)
        self.canvas_3.grid(row=2, column=2)

        # # Добавим video
        # self.canvas_4 = Canvas(self.root, height=640, width=480)
        # self.a_image = self.canvas_4.create_image(0, 0, anchor='nw', image=self.photo_3)
        # self.canvas_4.grid(row=2, column=3)

        self.root.mainloop()

    # Функции для 1 картинки
    def picture_origin(self):
        self.newImage = 0
        self.image = cv2.imread('pictures/tora_dora.jpg')
        self.readypic = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        self.photo = ImageTk.PhotoImage(self.readypic)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=2, column=0)

    def new_picture(self):
        self.newImage = Image.fromarray(self.newImage)
        self.photo = ImageTk.PhotoImage(self.newImage)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=2, column=0)

    def canny(self):
        origin = self.imageOriginColored
        _, thresh = cv2.threshold(origin, np.mean(origin), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((origin.shape[0], origin.shape[1]), np.uint8)
        masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
        dst = cv2.bitwise_and(origin, origin, mask=mask)
        segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        self.newImage = segmented
        self.new_picture()

    def kmean4(self):
        origin = self.imageOriginColored
        k = 4
        pixels = copy.deepcopy(origin)
        pixel_vals = pixels.reshape(-1, 3)
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(pixels.shape)

        self.newImage = segmented_image
        self.new_picture()

    def kmean8(self):
        origin = self.imageOriginColored
        k = 8
        pixels = copy.deepcopy(origin)
        pixel_vals = pixels.reshape(-1, 3)
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(pixels.shape)

        self.newImage = segmented_image
        self.new_picture_2()

    def kmean20(self):
        origin = self.imageOriginColored
        k = 20
        pixels = copy.deepcopy(origin)
        pixel_vals = pixels.reshape(-1, 3)
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(pixels.shape)

        self.newImage = segmented_image
        self.new_picture_3()

    # Функции для 2 картинки
    def picture_origin_2(self):
        self.newImage = 0
        self.image_2 = cv2.imread('pictures/volvo.png')
        self.readypic_2 = Image.fromarray(cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY))
        # print("readypic_2:  ", self.readypic_2)
        self.photo_2 = ImageTk.PhotoImage(self.readypic_2)
        # print("photo_2:  ", self.photo_2)
        # cv2.imshow("self.photo_2", self.photo_2)
        self.n_image = self.canvas_2.create_image(0, 0, anchor='nw', image=self.photo_2)
        self.canvas_2.grid(row=2, column=1)

    def new_picture_2(self):
        self.newImage = Image.fromarray(self.newImage)
        self.photo_2 = ImageTk.PhotoImage(self.newImage)
        self.n_image = self.canvas_2.create_image(0, 0, anchor='nw', image=self.photo_2)
        self.canvas_2.grid(row=2, column=1)

    def Ptile(self):
        origin = self.imageOrigin
        t = 150
        binary = origin > t
        self.newImage = binary
        self.new_picture_2()

    def buildHistogramm(self):
        origin = self.imageOrigin
        count = np.asarray(origin)
        fig, ax = plt.subplots()
        ax.hist(count.ravel(), 256, density=True, facecolor='b')
        plt.title('Изначальная гистограмма')
        old_peaks, _ = find_peaks(origin.ravel(), height=125)
        print(str(old_peaks.size))
        plt.show()

    def peaks(self):
        origin = self.imageOrigin
        origin = origin.ravel()
        container = copy.deepcopy(origin)
        size = container.shape[0]
        new = histogram_parallel(container, size, container)
        fig, ax = plt.subplots()
        ax.hist(new, 256, density=True, facecolor='b')
        plt.title('Сглаженная гистограмма')
        new_peaks, _ = find_peaks(new, height=125)
        print(str(new_peaks.size))
        plt.show()

    def adaptive_threshold(self):
        pixels = copy.deepcopy(self.imageOriginColored)
        height = self.imageOrigin.shape[0]
        width = self.imageOrigin.shape[1]
        size = 1
        matrix_size = (2 * size) + 1
        res_image = copy.deepcopy(pixels)
        mean = adaptive_threshold_parallel_mean_element(pixels, height, width, matrix_size, res_image)
        self.newImage = mean
        self.new_picture()
        min_max = adaptive_threshold_parallel_min_max(pixels, height, width, matrix_size, res_image)
        self.newImage = min_max
        self.new_picture_2()
        median = adaptive_threshold_parallel_median_3_3(pixels, height, width, res_image)
        self.newImage = median
        self.new_picture_3()
        # self.threshold_image = adaptive_threshold_parallel_median_5_5(pixels, height, width, res_image)

    # Функции для 3 картинки
    def picture_origin_3(self):
        self.newImage = 0
        self.image_3 = cv2.imread('pictures/volvo.png')
        self.readypic_3 = Image.fromarray(cv2.cvtColor(self.image_3, cv2.COLOR_BGR2GRAY))
        self.photo_3 = ImageTk.PhotoImage(self.readypic_3)
        self.a_image = self.canvas_3.create_image(0, 0, anchor='nw', image=self.photo_3)
        self.canvas_3.grid(row=2, column=2)

    def new_picture_3(self):
        self.newImage = Image.fromarray(self.newImage)
        self.photo_3 = ImageTk.PhotoImage(self.newImage)
        self.a_image = self.canvas_3.create_image(0, 0, anchor='nw', image=self.photo_3)
        self.canvas_3.grid(row=2, column=2)

    def following(self):
        origin = self.imageOrigin
        t = 127
        told = 0
        m1ar = []
        m2ar = []
        while told != t:
            for i in range(origin.shape[0]):
                for j in range(origin.shape[1]):
                    if origin[i, j] < t:
                        m1ar.append(origin[i, j])
                    elif origin[i, j] > t:
                        m2ar.append(origin[i, j])
            told = t
            m1 = int(np.mean(m1ar))
            m2 = int(np.mean(m2ar))
            t = int((m1 + m2) / 2)
            print(t)
        binary = origin > t
        self.newImage = binary
        self.new_picture_3()

    # # Functions for video
    # def video_origin(self):
    #     cv2.destroyAllWindows()
    #     self.video = cv2.VideoCapture("videos/Road.mp4")
    #
    #     while True:
    #         succes, img = self.video.read()
    #         new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         # x = cv2.Sobel(new_img, cv2.CV_16S, 1, 0)
    #         # y = cv2.Sobel(new_img, cv2.CV_16S, 0, 1)
    #         #
    #         # absX = cv2.convertScaleAbs(x)
    #         # absY = cv2.convertScaleAbs(y)
    #         # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    #         cv2.imshow("Results", new_img)
    #         if cv2.waitKey(1) & 0xFF == ord('q') | np.any(new_img) == 0:
    #             break


app = App()
