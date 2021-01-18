import os
import pandas as pd
import numpy as np
import cv2
import sys


def cria_data_frame(BASE_PATH:str) -> pd.DataFrame:
  """
    Função responsável por criar o DataFrame da base de dados
    comentários:
      Se nao tivesse teste poderiamos ter utilizado as seguintas linhas
        diretorios = [fn for fn in diretorios if len(fn.split('.')) < 2]  ===> (1)
        var = [fn for fn in os.listdir(diretorios) if fn.split('.')[-1] in ext] ===> (2) / ext = vetor com as extensões
  """
  os.chdir(BASE_PATH) #Abre o caminho da base de dados já validada.
  diretorios = os.listdir('.') #Lista os diretótorios da pasta. (1)
  diretorios = sorted(diretorios) #coloca em ordem alfabetica (tambem funciona com números).

  paths = [] #Lista para guardar os caminhos.
  cla = [] #Lista para guardar as classes das imagens.
  for c, d in enumerate(diretorios): #A variavel c serve como o valor que representa a classe. (2)
    for f in sorted(os.listdir(d)): #Percorre as imagens das classes.
      paths.append(BASE_PATH + '/' + d + '/' + f) #Fazendo um append (juntando) o caminho de cada imagem.
      cla.append(c) #Fazendo um append (juntando) o valor de classe para cada imagem.

  DF = pd.DataFrame({'imagens' : paths , 'classes' : cla}) #Cria um dataframe com os dados que obtivemos acima.
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
      self.folds.append(self.indices[sub:sub + tamanho]) #Cria os folds.

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



if __name__ == "__main__":
  import sys
  sys.exit(1)