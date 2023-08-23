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
import os
import zipfile
import numpy as np
# from keras.preprocessing import image
import cv2
# from tensorflow import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from playsound import playsound
import winsound
from pygame import mixer
import warnings
warnings.filterwarnings("ignore")

####################################################################
####################################################################
# here we load the trained model the give us the facial key points
####################################################################
####################################################################
export_path_sm = "./models/facial_points.h5"
keypoints_model = tf.keras.models.load_model(export_path_sm)
####################################################################
####################################################################
# here we load the trained model the predict whether the eye is an
# open eye or closed eye
####################################################################
####################################################################
export_path_sm = "./models/driver_drowsiness_softmax_13.h5"
eye_model = tf.keras.models.load_model(export_path_sm)
####################################################################
####################################################################

# this function take the image and the center of the rectangle wanted to be drawn
# and also the text and draw the rectangle and put the text above it
def draw_rectangle_with_text(img, x ,y , w , h ,color , text):
    cv2.rectangle(img, (x - w, y - h), (x + w, y + h), color, 2)
    # write over the rectangle the percentage of being open
    cv2.putText(img, text, (x - w - 6, y - h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    return img

# this fucntion take (96 x 96) and use the facial keypoint model to get 15 important point in the face
# but we just use the eyes center points
def getpoints(img):
    temp = np.expand_dims(img, axis=2)
    temp=temp/(255.0)
    temp = np.expand_dims(temp, axis=0)
    points = keypoints_model.predict(temp)
    return points

# this is the face class
class FaceAndPoints:
    def __init__(self, frame, x, y, w, h, gray_face):
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame = frame
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.gray_face = gray_face #get_larges_face(self.gray_frame)

    # using this function we call the getpoint function ti get the points
    # and resize the face to (400 x 400) to be mor visible
    # then return the np.array of points and the face
    def get_face_keypoints(self):
        face_for_keypoints = cv2.resize(self.gray_face, (96, 96))
        keypoints = getpoints(face_for_keypoints)

        face = self.frame[self.y: self.y + self.height, self.x: self.x + self.width]
        face = cv2.resize(face, (400, 400))

        return face, keypoints

# this is the eye class
class Eye:
    def __init__(self, x_index, y_index, w, h, points, face_image):
        self.x_index = x_index
        self.y_index = y_index
        self.points = points
        self.x_center = int(int(int(self.points[0][self.x_index]))*400/96)
        self.y_center = int(int(int(self.points[0][self.y_index]))*400/96)
        self.width = w
        self.height = h
        self.eye_is_open = False
        self.face_image = face_image

        # extract the eye from the image
        self.eye_img = self.face_image[self.y_center - h - 10: self.y_center + h, self.x_center - w-5: self.x_center + w+5]
        self.eye_class = 0
        self.color = (0,0,0)

    # in this  function we predict whether the eye is closed of open using the eye model
    def classify_the_eye(self):
        # here we resize the  eye image, because the model accept 150 x 150 images only
        resized_eye_img = cv2.resize(self.eye_img, (150, 150))
        # here we prepare the eye image for the model to be np.array(150 x 150 x 1)
        resized_eye_for_predection = np.array([resized_eye_img])
        self.eye_class = eye_model.predict(resized_eye_for_predection)[0][1]

    # in the draw model we draw rectangle around the eye with green if it is open and red if it is closes
    def draw_eye(self):
        if self.eye_class > 0.5:
            # if self.eye_class <= 0.65 : self.eye_class = (self.eye_class + 0.35)
            print('this eye: ', self.eye_class)
            self.color = (0, 255, 0)  # green color for open eyes
            self.eye_is_open = True  # set the boolean to true


        else:
            print('eye: ', self.eye_class)
            self.color = (0, 0, 255)
            self.eye_is_open = False

        text_on_the_eye = 'this eye is ' + str(int(float(self.eye_class) * 100)) + "% open"
        self.face_image = draw_rectangle_with_text(self.face_image, self.x_center, self.y_center,
                                                   self.width, self.height, self.color, text=text_on_the_eye)

        # m4 mwgoda f l file elt m3akom , bs deh bt7ot point fe el center bta3 l 3aan

        self.face_image = draw_rectangle_with_text(self.face_image, self.x_center, self.y_center,
                                                   1, 1, self.color, text="")

        #
        # j=4
        # while(j<29):
        #
        #     self.face_image = draw_rectangle_with_text(self.face_image, int(int(int(self.points[0][j])) * 400 / 96),
        #                                                int(int(int(self.points[0][j+1])) * 400 / 96),
        #                                                1, 1, self.color, text="")
        #     j+=2


        return self.face_image