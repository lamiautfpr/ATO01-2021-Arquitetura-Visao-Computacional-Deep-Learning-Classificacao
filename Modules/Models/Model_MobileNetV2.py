import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import Accuracy, AUC, FalseNegatives, Precision
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D



def compiled_model(INPUT_SHAPE:list, QNT_CLASS:int)-> tf.keras.Model:
  """
    A função retorna o modelo compilado.

    Return a compiled model.
  """
  INPUT_SHAPE = tuple(INPUT_SHAPE) 

  base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor= Input(shape= INPUT_SHAPE, name='inputs'))


  for layer in base_model.layers: 
    layer.trainable = False 

  mod = base_model.output 
  mod = AveragePooling2D()(mod)
  mod = Flatten()(mod)
  mod = Dropout(0.5)(mod)
  mod = Dense(QNT_CLASS, activation='softmax')(mod) 


  mod_retorno = Model(inputs = base_model.input, outputs = mod) 

  mod_retorno.compile(loss= CategoricalCrossentropy(), optimizer= Adagrad() , metrics=[Accuracy(), Precision(), AUC(), FalseNegatives()] )
  return mod_retorno 


if __name__ == "__main__":
  import sys
  sys.exit(1)






