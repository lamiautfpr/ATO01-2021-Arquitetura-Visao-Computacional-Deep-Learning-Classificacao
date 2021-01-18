import os 
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils import cria_data_frame, retorna_indices

class Generator(object):
  def __init__(self, BASE_PATH:str, SPLIT_POR:list, BATCH_SIZE:int, PREPRO_FUN):
    """

    """
    self.batch_size = BATCH_SIZE #Cria um atributo para armazenar o valor do batch size.
    self.preproc_fun = PREPRO_FUN #Cria um atributo com a função de preprocessamento para que possamos utilizalas em todas as etapas.
    self.data_frame = cria_data_frame(BASE_PATH) #Cria o dataframe com os caminhos e as classes das imagens.
    self.total_img = self.data_frame.shape[0] #Quantidade de imagens que temos na base de dados.
    self.total_class =  len(self.data_frame['classes'].unique())
    self.hold_data = [[],[]]
    #Cria os indices embaralhados de cada uma das etapas.
    self.train_ind, self.val_ind, self.test_ind = retorna_indices(SPLIT_POR[0],SPLIT_POR[1],SPLIT_POR[2],self.total_img)

    self.steps_train = int((self.train_ind / self.batch_size) + 1) #Passos por epoch, quantos batchs por epoca vao passar na rede.
    self.steps_val = int((self.val_ind / self.batch_size) + 1) #Passos por epoch, quantos batchs por epoca vao passar na validação.

  def train_generator(self):
    batch = [] #Array que vai carregar o batch de imagens.
    cla = [] #Array que vai carregar as classes correspondentes as imagens do batch.
    contador = 0 #Controle de puxada das imagens.


    while True: #Loop infinito que vai ser controlado na função de treinamento do modelo.
      #Faz a leitura da imagem apartir de seu caminho econtrado no data_frame, o train_ind esta embaralhado, pegando imagens "aleatoriamente".
      img = cv2.imread(self.data_frame.iloc[self.train_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Transforma a imagem para RGB por que o opencv le as imagens em BGR.
      cim = self.data_frame.iloc[self.train_ind[contador]]['classes'] #Carrega a classe da imagem através do data_frame.
      
      #Faz o preprocessamento na imagem e acrescenta no batch.
      self.preproc_fun(batch, cla, img, cim)
      
      if len(batch) > self.batch_size:
        while len(batch) > self.batch_size:
          self.hold_data[0].append(batch.pop()) #Guarda a imagem para depois.
          self.hold_data[1].append(cla.pop()) #Guarda a classe para depois
        #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
        batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = self.hold_data[0] #Carrega o holdout do batch.
        cla =  self.hold_data[1] #Carrega o holdout de classe.

      if len(batch) == self.batch_size: #Verifica se o batch chegou no tamanho para ser passado.
        batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
        #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] #Reinicia o batch.
        cla = [] #Reinicia a lista de classe.
      if contador == len(self.train_ind):
        contador = 0 #Reinicia o contador.
        resto = len(self.train_ind) % self.batch_size
        while not resto == 0:
          self._filler(batch, cla, resto, self.train_ind) #Preenche o que falta de imagens aleatorias da base para fechar um batch.
          batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
          #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
          yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
          batch = [] #Reinicia o batch.
          cla = [] #Reinicia a lista de classe.

  def val_generator(self):
    batch = [] #Array que vai carregar o batch de imagens.
    cla = [] #Array que vai carregar as classes correspondentes as imagens do batch.
    contador = 0 #Controle de puxada das imagens.


    while True: #Loop infinito que vai ser controlado na função de treinamento do modelo.
      #Faz a leitura da imagem apartir de seu caminho econtrado no data_frame, o val_ind esta embaralhado, pegando imagens "aleatoriamente".
      img = cv2.imread(self.data_frame.iloc[self.val_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Transforma a imagem para RGB por que o opencv le as imagens em BGR.
      cim = self.data_frame.iloc[self.val_ind[contador]]['classes'] #Carrega a classe da imagem através do data_frame.
      
      #Faz o preprocessamento na imagem e acrescenta no batch.
      self.preproc_fun(batch, cla, img, cim)
      
      if len(batch) > self.batch_size:
        while len(batch) > self.batch_size:
          self.hold_data[0].append(batch.pop()) #Guarda a imagem para depois.
          self.hold_data[1].append(cla.pop()) #Guarda a classe para depois
        #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
        batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = self.hold_data[0] #Carrega o holdout do batch.
        cla =  self.hold_data[1] #Carrega o holdout de classe..
      
      if len(batch) == self.batch_size: #Verifica se o batch chegou no tamanho para ser passado.
        batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
        #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] #Reinicia o batch.
        cla = [] #Reinicia a lista de classe.
      if contador == len(self.val_ind):
        contador = 0 #Reinicia o contador.
        resto = len(self.val_ind) % self.batch_size
        while not resto == 0:
          self._filler(batch, cla, self.val_ind) #Preenche o que falta de imagens aleatorias da base para fechar um batch.
          batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
          #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
          yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
          batch = [] #Reinicia o batch.
          cla = [] #Reinicia a lista de classe.


  def test_generator(self):
    batch = [] #Array que vai carregar o batch de imagens.
    cla = [] #Array que vai carregar as classes correspondentes as imagens do batch.
    contador = 0 #Controle de puxada das imagens.


    while True: #Loop infinito que vai ser controlado na função de treinamento do modelo.
      #Faz a leitura da imagem apartir de seu caminho econtrado no data_frame, o test_ind esta embaralhado, pegando imagens "aleatoriamente".
      img = cv2.imread(self.data_frame.iloc[self.test_ind[contador]]['imagens']) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Transforma a imagem para RGB por que o opencv le as imagens em BGR.
      cim = self.data_frame.iloc[self.test_ind[contador]]['classes'] #Carrega a classe da imagem através do data_frame.
      
      #Faz o preprocessamento na imagem e acrescenta no batch.
      self.preproc_fun(batch, cla, img, cim)

      if len(batch) > self.batch_size:
        while len(batch) > self.batch_size:
          self.hold_data[0].append(batch.pop()) #Guarda a imagem para depois.
          self.hold_data[1].append(cla.pop()) #Guarda a classe para depois
        #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
        batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = self.hold_data[0] #Carrega o holdout do batch.
        cla =  self.hold_data[1] #Carrega o holdout de classe.

      if len(batch) == self.batch_size: #Verifica se o batch chegou no tamanho para ser passado.
        batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
        #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
        yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
        batch = [] #Reinicia o batch.
        cla = [] #Reinicia a lista de classe.
      if contador == len(self.test_ind):
        contador = 0 #Reinicia o contador.
        resto = len(self.test_ind) % self.batch_size
        while not resto == 0:
          self._filler(batch, cla, self.test_ind) #Preenche o que falta de imagens aleatorias da base para fechar um batch.
          batch = np.array(batch, np.float32) / 255. #Normaliza os valores do batch, transformando os pixels para 0-1.
          #Retorna o batch e a lista de classes no formato correto para a rede, as classes ficam em formato categorico ex: 3 => [0,0,1].
          yield (batch, np.array(to_categorical(cla, self.total_class), np.float32))
          batch = [] #Reinicia o batch.
          cla = [] #Reinicia a lista de classe.
  
  def _filler(self, batch:list, cla:list, indice:list ):
    aleatorio = int(np.random.rand() * self.total_img)
    #Faz a leitura da imagem apartir de seu caminho econtrado no data_frame.
    img = cv2.imread(self.data_frame.iloc[indice[aleatorio]]['imagens']) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Transforma a imagem para RGB por que o opencv le as imagens em BGR.
    cim = self.data_frame.iloc[indice[aleatorio]]['classes'] #Carrega a classe da imagem através do data_frame.
    
    #Faz o preprocessamento na imagem e acrescenta no batch.
    self.preproc_fun(batch, cla, img, cim)  

  def set_indices(self, train_ind, val_ind, test_ind):
    self.train_ind = train_ind
    self.val_ind = val_ind
    self.test_ind = test_ind

  def Redefine(self, pot:int):
    self = self.__init__()
