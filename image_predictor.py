import cv2
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
from keras.models import load_model
import tensorflow_addons as tfa


import tensorflow as tf

model = None
rms = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

def load_model1():
    model = load_model('resnet_model.h5')
    print("model loaded")
    model.compile(loss='categorical_crossentropy',
                   optimizer=rms,
                   metrics=['accuracy'])
    print("model compiled successfully")
    return model

# def read_imagefile(file)->Image.Image:
#     image = cv2.imread(file)
#     return image

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image    

def predict(image1: Image.Image):
    global model
    if model is None:
        model = load_model1()

    image_resized=  np.asarray(image1.resize((224,224)))[..., :3]   
    finalimg = np.expand_dims(image_resized,axis=0)
    finalimg = tf.keras.applications.mobilenet_v2.preprocess_input(finalimg)
    predictions = model.predict(finalimg)
    final_prediction = max(predictions)
    predicted_class1 = np.argmax(predictions)
    print(predicted_class1)
    item = image_name(predicted_class1)
    return item

def image_name(predicted_class):
    food_items = ["donuts", "fries", "noodles", "pizza", "samosa"]
    return food_items[predicted_class]    