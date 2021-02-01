import os 
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils import cria_data_frame, retorna_indices

class Generator(object):
  def __init__(self, BASE_PATH:str, PARAMS, PREPRO_FUN, PRETEST_FUN):
    """
      Classe de gerador, cria os geradores necessários para o treinamento, validação e teste.
    """
    self.batch_size = PARAMS['BATCH_SIZE']
    self.preproc_fun = PREPRO_FUN 
    self.pretest_fun = PRETEST_FUN 
    self.data_frame = cria_data_frame(BASE_PATH, PARAMS)
    self.total_img = self.data_frame.shape[0]
    self.total_class =  len(self.data_frame['classes'].unique())
    self.hold_data = [[],[]]
    
    self.train_ind, self.val_ind, self.test_ind = retorna_indices(PARAMS['SPLIT_SIZE'][0], 
                                                                  PARAMS['SPLIT_SIZE'][1], 
                                                                  PARAMS['SPLIT_SIZE'][2], 
                                                                  self.total_img)

    self.steps_train = int((len(self.train_ind) / self.batch_size) + 1) 
    self.steps_val = int((len(self.val_ind) / self.batch_size) + 1)

  def train_generator(self):
    """
      Gerador de imagens para o treinamento. O controle do gerador é feito a partir do steps_per_epoch na função fit do modelo.
    """
    batch = []
    cla = [] 
    contador = 0 

    while True: 
      
      img = cv2.imread(self.data_frame.iloc[self.train_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
      cim = self.data_frame.iloc[self.train_ind[contador]]['classes']
      
      self.preproc_fun(batch, cla, img, cim)
      
      contador += 1
      if len(batch) > self.batch_size:
        while len(batch) > self.batch_size:
          self.hold_data[0].append(batch.pop()) 
          self.hold_data[1].append(cla.pop())
        
        batch = np.array(batch, np.float32) / 255. 
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = self.hold_data[0] 
        cla =  self.hold_data[1] 

      if len(batch) == self.batch_size: 
        batch = np.array(batch, np.float32) / 255. 
        
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] 
        cla = [] 
      if contador == len(self.train_ind):
        contador = 0 
        resto = len(self.train_ind) % self.batch_size
        while not len(batch) == self.batch_size:
          self._filler(batch, cla, self.train_ind) 
        batch = np.array(batch, np.float32) / 255. 
        
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] 
        cla = [] 

  def val_generator(self):
    """
      Gerador de imagens para a validação. O controle do gerador é feito a partir do validation_steps na função fit do modelo.
    """
    batch = [] 
    cla = [] 
    contador = 0 


    while True: 
      
      img = cv2.imread(self.data_frame.iloc[self.val_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
      cim = self.data_frame.iloc[self.val_ind[contador]]['classes'] 
      
      
      self.preproc_fun(batch, cla, img, cim)
      
      contador += 1
      if len(batch) > self.batch_size:
        while len(batch) > self.batch_size:
          self.hold_data[0].append(batch.pop()) 
          self.hold_data[1].append(cla.pop())
        
        batch = np.array(batch, np.float32) / 255. 
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = self.hold_data[0] 
        cla =  self.hold_data[1]
      
      if len(batch) == self.batch_size: 
        batch = np.array(batch, np.float32) / 255. 
        
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] 
        cla = [] 
      if contador == len(self.val_ind):
        contador = 0 
        resto = len(self.val_ind) % self.batch_size
        while not len(batch) == self.batch_size:
          self._filler(batch, cla, self.val_ind) 
        batch = np.array(batch, np.float32) / 255. 
        
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] 
        cla = [] 


  def test_generator(self):
    """
      Gerador de imagens para o teste, nao retorna em batch, retorna imagem por imagem com suas respectivas classes.
    """
    contador = 0 


    while contador < len(self.test_ind):

      img = cv2.imread(self.data_frame.iloc[self.test_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
      cim = self.data_frame.iloc[self.test_ind[contador]]['classes'] 
      
      img,cim = self.pretest_fun(img, cim)

      contador += 1
      
      img = np.array(img, np.float32) / 255.
      img = img[np.newaxis]
      cim = np.array(to_categorical(cim, self.total_class), np.float32)
      cim = cim[np.newaxis]
      yield (img, cim)
      
  def eval_generator(self):
    """
      Gerador separado para função evaluate.
    """
    batch = [] 
    cla = [] 
    contador = 0 


    while contador < len(self.test_ind): 
      
      img = cv2.imread(self.data_frame.iloc[self.test_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
      cim = self.data_frame.iloc[self.test_ind[contador]]['classes'] 
      
      img, cim = self.pretest_fun(img, cim)
      batch.append(img)
      cla.append(cim)

      contador += 1
      if len(batch) == self.batch_size: 
        batch = np.array(batch, np.float32) / 255. 
        
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] 
        cla = [] 
      if contador == len(self.test_ind):
        batch = np.array(batch, np.float32) / 255. 
        
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] 
        cla = [] 


  
  def _filler(self, batch:list, cla:list, indice:list ):
    """
      Completa o ultimo batch (caso precise) com imagens (da base) aleatorias .
    """
    aleatorio = int(np.random.rand() * len(indice))
    
    img = cv2.imread(self.data_frame.iloc[indice[aleatorio]]['imagens']) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    cim = self.data_frame.iloc[indice[aleatorio]]['classes'] 
    
    
    self.preproc_fun(batch, cla, img, cim)  

  def set_indices(self, train_ind, val_ind, test_ind):
    self.train_ind = train_ind
    self.val_ind = val_ind
    self.test_ind = test_ind

  def Redefine(self, BASE_PATH:str, PARAMS, PREPRO_FUN, PRETEST_FUN):
    self = self.__init__(BASE_PATH, PARAMS, PREPRO_FUN, PRETEST_FUN)
