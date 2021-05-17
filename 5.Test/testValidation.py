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
