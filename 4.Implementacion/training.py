# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# !pip install tensorflow-gpu == 2.0.0.alpha0
!git clone https://github.com/victorjoab98/GuatemalanCurrencyDataSet.git

import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K 
# %matplotlib inline
tf.__version__

# import os
# os.chdir("/content/")
#!git pull
!ls

data_entrenamiento = './GuatemalanCurrencyDataSet/Entrenamiento'
  data_validacion = './GuatemalanCurrencyDataSet/Validation'
  altura = 100
  longitud = 100
  epocas = 20
  filtrosConv1 = 32
  filtrosConv2 = 64
  tamanio_filtro1 = (3,3)
  tamanio_filtro2 = (2,2)
  pasos = 1000
  pasos_validacion = 200
  tamanio_pool = (2,2)
  clases = 2
  lr = 0.0005

entrenamiento_datagen = ImageDataGenerator(
      rescale = 1./255,
      shear_range = 0.3,
      zoom_range = 0.3,
      # horizontal_flip = True
  )

  validacion_datagen = ImageDataGenerator(
      rescale = 1./255
  )

  imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
      data_entrenamiento, 
      target_size=(altura, longitud),
      batch_size=32,
      class_mode='categorical'
  )

  imagen_validacion = entrenamiento_datagen.flow_from_directory(
      data_validacion, 
      target_size=(altura, longitud),
      batch_size=32,
      class_mode='categorical'
  )

cnn = Sequential()

cnn.add(Convolution2D(
    filtrosConv1, 
    tamanio_filtro1, 
    padding='same', 
    input_shape=(altura, longitud, 3),
    activation='relu'))

cnn.add(MaxPooling2D(pool_size= tamanio_pool))
cnn.add(Convolution2D(
    filtrosConv2,
    tamanio_filtro2,
    padding='same',
    activation='relu'
))
cnn.add(MaxPooling2D(pool_size=tamanio_pool))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data = imagen_validacion, validation_steps=pasos_validacion)
dir = './modelo'
if not os.path.exists(dir):
  os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud = 100
altura = 100 
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)


def predict(file):
  x=load_img(file, target_size=(longitud, altura))
  x=img_to_array(x)
  x = np.expand_dims(x, axis=0)
  arreglo = cnn.predict(x)
  resultado = arreglo[0]
  respuesta=np.argmax(resultado)

  if respuesta==0: 
    print('10')
  elif respuesta==1:
    print('100')   
  return respuesta

predict('./GuatemalanCurrencyDataSet/Validation/100/IMG_20210427_183236.jpg')

from google.colab import files
files.download('./modelo/modelo.h5')
files.download('./modelo/pesos.h5')
