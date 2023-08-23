import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import zipfile
from os import path, getcwd, chdir
import PIL
import PIL.Image
from keras.models import load_model
import h5py
import pathlib
import cv2
import tensorflow.keras as keras
import os
import zipfile
import numpy as np
from keras.preprocessing import image
import cv2
from tensorflow import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
import matplotlib.pyplot as plt
from playsound import playsound
import winsound
from pygame import mixer
import project_classes as PC


####################################################################
####################################################################
# input = gray frame
# this function gets the larges face in the frame
# according to the area of the faces
# and return its , output = x,y,width,height
####################################################################
####################################################################
def get_larges_face(gray_frame):
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = facecascade.detectMultiScale(gray_frame)
    # to make sure that there is a face in the image
    if len(faces) > 0:
        # to get the biggest face in the image
        max_w = 0
        max_h = 0
        index = -1
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            # we compare between the area of the faces if there is more than one face
            if ((w * h) > (max_w * max_h)):
                max_w = w
                max_h = h
                index = i
        # here we get the width and the hight of the face to crop the face from the image
        (x, y, width, height) = faces[index]

        gray_face = gray_frame[y: y + height, x: x + width]
        return x, y, width, height, gray_face
    # if the number of faces is zero "no face exist return all zeros"
    else:
        return 0, 0, 0, 0, gray_frame


####################################################################
####################################################################
# input = img, x, y, width , height , color of rectangle ,
# ,text on the top of the rectangle , thickness of the rectangle side
####################################################################
# this function draw a rectangle inside the img where x,y is its center
# and after that it put text on the top of the rectangle
# and return the img after the modification (after drawing)
####################################################################
####################################################################
def draw_rectangle_with_text(img, x, y, w, h, color, text, thickness):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # write over the rectangle the percentage of being open
    cv2.putText(img, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
    return img


###############################################################################################################
# in this function we check how many eye are closed and how many are open and acourding to that
# we draw the rectangle around the face
# if both are open the color of rectangle is green and decrease the drowsycounter gradualy and the thickness
# as well as the drowsy counter
# else if both are closed the color of rectangle is red and increase the drowsycounter gradualy and the thickness
# aand if one is closed and one is open the color is blue
###############################################################################################################
def eyesanddrowsyness(left_eye, right_eye, img, drowsycounter):
    if left_eye.eye_is_open and right_eye.eye_is_open:
        color = (0, 255, 0)  # green
        thickness = int((
                                    drowsycounter + 2) / 2)  # decrease the thickness, and the "+2" is just to make sure that it will one or more
        drowsycounter = (drowsycounter > 0) * (drowsycounter - 1)  # decrement the drowsycounter by one

        # draw and write on the img , the image is 400 x 400 , so i used 15, 15, to leave some space from up and left
        # and made the width and height 370 so that will leave the same space from right and down "15+370=385"
        img = draw_rectangle_with_text(img, 15, 15, 370, 370, color, text='both are open', thickness=thickness)


    elif left_eye.eye_is_open or right_eye.eye_is_open:
        color = (255, 0, 0)  # blue ... "yes, in opencv it is (B,G,R) not (R,G,B)" so this is blue not red
        thickness = int((drowsycounter + 2))
        drowsycounter = (drowsycounter > 0) * (drowsycounter - 1)  # decrement the drowsycounter by one

        # draw and write on the img , the image is 400 x 400 , so i used 15, 15, to leave some space from up and left
        # and made the width and height 370 so that will leave the same space from right and down "15+370=385"
        img = draw_rectangle_with_text(img, 15, 15, 370, 370, color, text="one is open", thickness=thickness)

    else:
        color = (0, 0, 255)  # red ... "yes, in opencv it is (B,G,R) not (R,G,B)" so this is red not blue
        thickness = int((drowsycounter + 2) / 1.5)  # increase the thickness,by 2
        drowsycounter = (drowsycounter + 1 * (
                    drowsycounter <= 20))  # increase the thickness,by 1 + make sure it will not exceed 20

        # draw and write on the img , the image is 400 x 400 , so i used 15, 15, to leave some space from up and left
        # and made the width and height 370 so that will leave the same space from right and down "15+370=385"
        img = draw_rectangle_with_text(img, 15, 15, 370, 370, color, text="both are closed", thickness=thickness)

    print('drowsy counter: ', drowsycounter)
    return img, drowsycounter


################################################################################################################
################################################################################################################
# we can consider this function as a main function
# it takes the frame and the drowsycounter
# get the largest face parameters ( x , y , w , h ) using get_larges_face() function
# check if width and height are zeros which means there is no faces in thr frame
# so return the frame directly with no changes
# else if there is a face in the frame
# we create face object using face and point class in the project_classes.py file
# and initialize it with the face parameters face_OB = PC.FaceAndPoints(frame, x, y, w, h, gray_face)
# get the extracted face and the facial points using face_OB.get_face_keypoints() function
# then use the returned (driver Face) and (pointes) to intialize the two eyes objects
################################################################################################################
################################################################################################################
def project_func(frame, drowsycounter):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the larges (nearest) face in the frame if there are more than one
    # if  one return it
    x, y, w, h, gray_face = get_larges_face(gray_frame)

    # check if width and height are zere means no faces so retern the fram dirctly with no change
    if w == 0 and h == 0:
        return frame, drowsycounter

    # face object
    face_OB = PC.FaceAndPoints(frame, x, y, w, h, gray_face)

    # get the extracted face and the facial keypoints
    the_driver_face, the_driver_facial_keypoints = face_OB.get_face_keypoints()

    # Eye object for the left eye
    # PC.Eye(X_Point, Y_point, Width, Height, points, driver face image)
    left_eye_OB = PC.Eye(0, 1, 55, 55, the_driver_facial_keypoints, the_driver_face)

    # predect if the eye is clossed of open
    left_eye_OB.classify_the_eye()

    # draw rectangle around the left eye and put the percentage on it
    frame_with_lefteye_draw = left_eye_OB.draw_eye()

    # Eye object for the left eye
    # PC.Eye(X_Point, Y_point, Width, Height, points, driver face image)
    right_eye_OB = PC.Eye(2, 3, 55, 55, the_driver_facial_keypoints, frame_with_lefteye_draw)

    # predect if the eye is clossed of open
    right_eye_OB.classify_the_eye()

    # draw rectangle around the right eye and put the percentage on it
    frame_with_eyes = right_eye_OB.draw_eye()

    # draw rectangle around the face with different colors and different thikness accourding to the drowsycounter
    # and the closed and open eyes and increase or decrease the drowsycounter according to the case
    # then return the img with the drawed rectangle and the drowsy detection
    final_frame, drowsycounter = eyesanddrowsyness(left_eye_OB, right_eye_OB, frame_with_eyes, drowsycounter)

    return final_frame, drowsycounter


def adjust_brighness(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def increaseBrightness(img, value=65):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def increaseContrast(img , clipLimit=6):
    # https://learnopencv.com/color-spaces-in-opencv-cpp-python/
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    contrasted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = contrasted

    # r, g, b = cv2.split(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl1 = clahe.apply(r)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl2 = clahe.apply(g)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl3 = clahe.apply(b)
    #
    # limg = cv2.merge((cl1, cl2, cl3))

    return img


def decreaseNoice(img):
    # ----- apply gaussian blurring -------------------------------------------
    gausian_blured = cv2.GaussianBlur(img, (3, 3), 0)

    # ----- apply median blurring -------------------------------------------
    median_blured = cv2.medianBlur(img, 3)

    # img = gausian_blured
    img = median_blured

    # # ----- Create kernals for sharpening -------------------------------------------
    # kernel1 = np.array([[-1, -1, -1],
    #                     [-1, 9, -1],
    #                     [-1, -1, -1]])
    #
    # kernel2 = np.array([[-1.5, -1.5, -1.5],
    #                     [-1.5, 13.5, -1.5],
    #                     [-1.5, -1.5, -1.5]])
    #
    # kernel3 = np.array([[-0.5, -0.5, -0.5],
    #                     [-0.5, 4.5, -0.5],
    #                     [-0.5, -0.5, -0.5]])
    #
    # kernel4 = np.array([[-2, -2, -2],
    #                     [-2, 18, -2],
    #                     [-2, -2, -2]])
    #
    # kernel5 = np.array([[-0.25, -0.25, -0.25],
    #                     [-0.25, 2.25, -0.25],
    #                     [-0.25, -0.25, -0.25]])
    #
    # # -----Applying the kernal-------------------------------------------
    # sharpened = cv2.filter2D(img, -1, kernel3)
    # img = sharpened
    return img


def preprocess_the_frame(img, condition=0):
    # conition = 0 for high light
    # conition = 1 for good light
    # conition = 2 for bad light

    # if condition==0:
    #     return img
    #
    # elif(condition==1):
    #     img = adjust_brighness(img, 1)
    #     # img = increaseBrightness(img,50)
    #     img = increaseContrast(img,2)
    #     # img = decreaseNoice(img)
    #
    # elif condition==2:
    #     img = adjust_brighness(img, 2.5)
    #     # img = increaseBrightness(img,50)
    #     img = increaseContrast(img,2.5)
    #     # img = decreaseNoice(img)

    img = adjust_brighness(img,7)
    img = increaseContrast(img)
    img = decreaseNoice(img)
    # img = cv2.fastNlMeansDenoisingColored(img , None)
    return img


# laptopcamera = 0
# secondarycamera = 1

# cap = cv2.VideoCapture(0)  # get stream of frames
drowsycounter = 0  # initialization of the drowsyness counting
mixer.init()
sound = mixer.Sound('alarm.mp3')  # this is the alarm sound
#
# while True:
#     _, frame = cap.read()
#
#     # make it flipped like mirror
#     frame = cv2.flip(frame, 1)
#
#     # conition = 0 for high light
#     # conition = 1 for good light
#     # conition = 2 for bad light
#     frame = preprocess_the_frame(frame, condition=0)
#
#     # get the latest frame with all the drawing on it after applying the models
#     frame, drowsycounter = project_func(frame, drowsycounter)
#
#     # resize to a fixed size
#     frame = cv2.resize(frame, (600, 600))
#
#     # display
#     cv2.imshow('', frame)
#
#     # if his both eyes was closed for 7 frames so alarm play
#     if drowsycounter >= 5:
#         sound.play()
#
#     if drowsycounter < 5:
#         sound.stop()
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
