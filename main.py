from tkinter import *
import random
import cv2 as cv2
from PIL import Image, ImageTk, ImageEnhance
import copy
import time
from scipy import ndimage, misc
import numpy as np
import math
from numba import prange, njit
import threading
import tkinter.ttk as ttk

# from filters import sobel_filter_three_1, getsobel, sobel_filter_five_1, sobel_filter_seven_1, deffOfGaussian, \
#     laplasianOfGaussian


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

        self.image = cv2.imread("pictures/volvo.png")
        self.imageOrigin = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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

        # вставляем кнопки T2
        Button(self.frame, text="Вернуть", command=self.picture_origin_2).grid(row=0, column=1)

        # вставляем кнопки T3
        Button(self.frame, text="Вернуть", command=self.picture_origin_3).grid(row=0, column=3, columnspan=2)

        # Buttons for video
        Button(self.frame, text="Показать", command=self.video_origin).grid(row=0, column=5, columnspan=2)

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

        # Добавим video
        self.canvas_4 = Canvas(self.root, height=640, width=480)
        self.a_image = self.canvas_4.create_image(0, 0, anchor='nw', image=self.photo_3)
        self.canvas_4.grid(row=2, column=3)

        self.root.mainloop()

    # Функции для 1 картинки
    def picture_origin(self):
        self.newImage = 0
        self.image = cv2.imread('pictures/volvo.png')
        self.readypic = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        self.photo = ImageTk.PhotoImage(self.readypic)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=2, column=0)

    def new_picture(self):
        self.newImage = Image.fromarray(self.newImage)
        self.photo = ImageTk.PhotoImage(self.newImage)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=2, column=0)


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


    # Functions for video
    def video_origin(self):
        cv2.destroyAllWindows()
        self.video = cv2.VideoCapture("videos/Road.mp4")

        while True:
            succes, img = self.video.read()
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # x = cv2.Sobel(new_img, cv2.CV_16S, 1, 0)
            # y = cv2.Sobel(new_img, cv2.CV_16S, 0, 1)
            #
            # absX = cv2.convertScaleAbs(x)
            # absY = cv2.convertScaleAbs(y)
            # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            cv2.imshow("Results", new_img)
            if cv2.waitKey(1) & 0xFF == ord('q') | np.any(new_img) == 0:
                break


app = App()