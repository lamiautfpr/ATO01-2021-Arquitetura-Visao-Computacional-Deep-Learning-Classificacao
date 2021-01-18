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
  """
  INPUT_SHAPE = tuple(INPUT_SHAPE) #Transforma o parametro para tupla por que o tf só aceita assim
  #Carrega o modelo com alguns parametros que o deixam sem outputs, com o input configurado e com transferlearning do treino com imagenet
  base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor= Input(shape= INPUT_SHAPE, name='inputs'))


  for layer in base_model.layers: #Percorre camada por camada contida no modelo base.
    layer.trainable = False #Para que as camadas da MobileNetV2 nao sejam treinadas, caso contrário iria demorar muito o treino, e nao a necessidade

  mod = base_model.output #Recebe a ultima camada do modelo base.
  mod = AveragePooling2D()(mod) #Cria uma camada que aplica um pooling por media de 2x2.
  mod = Flatten()(mod) #Cria uma camada que flatena (transforma para uma dimensão todos os mapas de caracteristicas).
  mod = Dropout(0.5)(mod) #Cria uma camada que desliga 50% dos neuronios que chegam até aqui para evitar overfitting.
  mod = Dense(QNT_CLASS, activation='softmax')(mod) #Cria uma camada de saída com função de ativação softmax.

  #Monta a rede, conectando as camadas que criamos agora, com as do base_model, tendo assim um Finetuning da MobileNetV2 com transfer learning.
  mod_retorno = Model(inputs = base_model.input, outputs = mod) 

  #Compila o modelo com os erros, otimizadores e metricas necessários. Monta a rede com tudo que ela precisa saber.
  mod_retorno.compile(loss= CategoricalCrossentropy(), optimizer= Adagrad() , metrics=[Accuracy(), Precision(), AUC(), FalseNegatives()] )
  return mod_retorno #Retorna o modelo


if __name__ == "__main__":
  import sys
  sys.exit(1)






