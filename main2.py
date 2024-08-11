import os
from keras.models import load_model, model_from_json
from time import sleep
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# classifier = model_from_json(open("model.json", "r").read())
# classifier.load_weights('model.keras')
# # model = load_model('static\Fer2013.h5')
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap=cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(r'C:\Users\likhi\OneDrive\Desktop\ML Projects\Facial Emotion Detection\Detection_app\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\likhi\OneDrive\Desktop\ML Projects\Facial Emotion Detection\Detection_app\model.h5')
print(classifier.summary())
