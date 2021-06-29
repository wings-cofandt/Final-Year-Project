# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 00:29:27 2021

@author: Imran Khan
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
import numpy as np

# Dimensions of our images.
img_width, img_height = 150, 150  #lowering computations

# train_data_dir = 'D:\Datasets\dataset\Training'
# validation_data_dir = 'D:\Datasets\dataset\Validation'
# test_data_dir = 'D:\Datasets\dataset\Testing'
# nb_train_samples = 383 
# nb_validation_samples = 221 
# epochs = 2
# batch_size = 32

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

# # Load pretrained VGG16 model.
# # The last (top) layers doing the final classification are not included.
# vgg16 = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)

# # Freeze the weights for the first layers.
# for layer in vgg16.layers[:17]:
#     layer.trainable = False
# # vgg16.summary()

# # Add custom layers.
# x = vgg16.output
# x = Flatten()(x)
# x = Dense(units=64, activation="relu")(x)
# x = Dropout(0.5)(x)
# predictions = Dense(units=3, activation="softmax")(x)
# print(vgg16.input)
# # Create final model.
# #model = Model(inputs = vgg16.input, output = predictions)
# model = Model(inputs=vgg16.input, outputs=predictions)
# model.compile(loss='CategoricalCrossentropy',
#               optimizer=SGD(lr=0.001, momentum=0.9),
#               metrics=['accuracy'])
# history = model.summary()

# tf.keras.utils.plot_model(
#   model, to_file='/content/model.png', show_shapes=True, show_dtype=False,
#   show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
#   )

# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1. / 255)
# from keras.callbacks import ModelCheckpoint
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')


# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')
# filepath = "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
# saver=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=2)

# print(test_generator)
'''
model.fit(
    train_generator,
    callbacks=[saver],
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
'''

from keras.models import load_model
import cv2
model = load_model('saved-model-02-0.95.hdf5')
import os
def live_camera():
    vid = cv2.VideoCapture(0)
    while(True):
      
        # Capture the video frame
        # by frame
        ret, img = vid.read()
        frame = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)
       
        array = np.expand_dims(frame, axis=0)
        # print(array.shape)
        preds = model.predict(preprocess_input(array))
        array_preds = max(preds)
        labels = ['Averge Personality', 'Fair Personality', 'Good Personality']
        index_pred = np.argmax(array_preds)
        # print(labels[index_pred])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,labels[index_pred],(10,50), font, 1,(255,255,255),2)
        # Display the resulting frame
        cv2.imshow('Predictions', img)
          
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # break
  
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows() 
live_camera()


def image_predict(path):
    img=cv2.imread(path)
    # img = image.load_img(path, target_size=(150, 150))
    frame = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)
    labels = ['Averge Personality', 'Fair Personality', 'Good Personality']
    array = image.img_to_array(frame)
    # print(array.shape)
    array = np.expand_dims(array, axis=0)
    # print(array.shape)
    preds = model.predict(preprocess_input(array))
    # res = decode_predictions(preds, top=1)[0]
    
    array_preds = max(preds)
    
    index_pred = np.argmax(array_preds)
    # print(labels[index_pred])
    # cv2.putText(img, labels[index_pred], (10,500), fontFace, fontScale, color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,labels[index_pred],(10,50), font, 1,(255,255,255),2)
    cv2.imshow('Predictions',img)
    cv2.waitKey(10)
path=r'D:\Sir Umer\Training\Test Images\10.jpg'
# image_predict(path)