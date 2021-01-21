import os
import pandas as pd
import numpy as np
import cv2
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def cria_data_frame(BASE_PATH:str, PARAMS) -> pd.DataFrame:
  """
    Função responsável por criar o DataFrame da base de dados
  """
  or_path = os.path.abspath('.')
  os.chdir(BASE_PATH) 
  diretorios = [dir for dir in os.listdir('.') if os.path.isdir(dir)]
  diretorios = sorted(diretorios) 
  PARAMS['QNT_CLA'] = len(diretorios)

  paths = []
  cla = [] 
  for c, d in enumerate(diretorios):
    for f in sorted([fil for fil in os.listdir(d) if fil.split('.')[-1] in ['jpg','jpeg','png']]):
      paths.append(BASE_PATH + '/' + d + '/' + f)
      cla.append(c)

  DF = pd.DataFrame({'imagens' : paths , 'classes' : cla})
  os.chdir(or_path)
  return DF #Retorna o DataFrame.

def retorna_indices(train:float, val:float, test:float, total:int) -> np.array:
  """
    Cada parametro (exceto total) recebe a porcentagem ->(em valores floats 15% = 0.15)<- do data set que vai ser dividida:
      -train = Porcentagem de imagens que vao ser utilizadas para treino;
      -val = Porcentagem de imagens que vao ser utilizadas para validação;
      -teste = Porcentagem de imanges que vao ser utilizadas para teste;

    O parametro total é a quantidade de imagens total no dataset, recebe um inteiro.

    A função retorna 3 arrays sendo o primeiro os indices de treinamento, o segundo os de validação e o terceiro o de teste:
    return train_ind, val_ind, test_ind

  """
  indices = np.arange(total) #Cria um array com valores entre 0 e (total).
  np.random.shuffle(indices) #Embaralha os elementos do array.

  ind_train = indices[0:int(total*train)] #Corta o array de 0 até seu total multiplicado pela procentagem de treino.
  #ex: Um array de 10 elementos, se eu quero 0.8 dele, vou de 0 até 10*0.8 = 8. 
  ind_val = indices[int(total*train):int(total*(train + val))] #Corta o array para validação.
  ind_teste = indices[int(total*(train + val)):] #Conrta o array para teste.

  return ind_train, ind_val, ind_teste #Retornas os arrays para que possamos utilizalos.

class Cross_val():
  def __init__(self, NUM:int, FOLDS:int):
    self.num_folds = FOLDS
    self.indices = np.arange(NUM) #Cria um vetor com numeros de 0 a NUM.
    np.random.shuffle(self.indices) #Embaralha os indices do vetor.

    self.folds = [] #Onde vai ficar os nossos folds de indices.

    tamanho = int(NUM/self.num_folds) #O tamanho de cada fold sem excesso, os excessos vao ser tratados abaixo.

    for sub in range(0, NUM, tamanho): #A sub fica com um fold por interação do laço de repetição.
      self.folds.append(self.indices[sub:sub + tamanho].tolist()) #Cria os folds.

    if len(self.folds) > FOLDS: #Se tiver um fold a mais na variavel significa que teve sobra, e deve ser tratada.
      resto = self.folds.pop() #Removo e guardo o fold com os restos.
      for i in range(len(resto)): 
        self.folds[i].append(resto[i]) #Adiciono um extra para os primeiros folds.
    
    self._make_table_folds() #Cria a tabela de ordem de folds.
  
  def _make_table_folds(self):
    or_table = list(np.arange(self.num_folds)) #Cria um array de 0 a num_folds.
    or_table.extend(or_table) #Extende a si mesmo ex: [1,2,3,1,2,3]
    
    self.table = [] #Tabela a ser armazenada.

    for id in range(self.num_folds):
      subar = or_table[id:id+self.num_folds] #Subarray do tamanho da quantidade de folds.
      aux = [[],[],[]] #Auxiliar.
      aux[2] = [subar.pop()] #Fold de teste.
      aux[1] = [subar.pop()] #Fold de validação.
      aux[0] = subar         #Folds de treino.
      self.table.append(aux) #Adiciona a tabela.
    

  def get_set_distribuition(self, NUM:int):
    return self.table[NUM] #Retorna os folds a serem usados como treino, val, e teste.


def matrix_confusion(pred, real, path):
  total = len(pred[0])
  pred = np.argmax(pred,axis=-1)
  real = np.argmax(real,axis=-1) 

  conf_m = confusion_matrix(real, pred, labels=np.arange(total))
  fig, ax = plt.subplots(figsize = (10,10))

  cm = ConfusionMatrixDisplay(conf_m, np.arange(total))

  cm.plot(include_values=True, cmap='Blues', ax= ax, xticks_rotation='vertical')
  plt.savefig(path + '.jpg')


if __name__ == "__main__":
  import sys
  sys.exit(1)