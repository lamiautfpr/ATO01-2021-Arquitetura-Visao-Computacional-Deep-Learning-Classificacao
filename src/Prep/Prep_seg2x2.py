import cv2

def pre_train(batch, cla, img, cim):
  """
    Função de preprocessamento que redimensiona a imagem para o tamnho de 224x224.
      batch = Lista de imagens no tamanho de um batch, aqui deve fazer append das imagens.
      cla = Lista de classes das imagens, deve ter o mesmo tamanho do batch, e o indice da imagem no batch deve corresponder com o indice nessa lista.
      img = Imagem.
      cim = Classe da imagem
  """
  simg = img[0:int(img.shape[0]/2) , 0:int(img.shape[1]/2),:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  simg = img[int(img.shape[0]/2) : img.shape[0] , 0:int(img.shape[1]/2),:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  simg = img[0:int(img.shape[0]/2) , int(img.shape[1]/2) : img.shape[1],:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  simg = img[int(img.shape[0]/2) : img.shape[0] , int(img.shape[1]/2) : img.shape[1],:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

def pre_test(img, cim):
  batch = []
  cla = []

  simg = img[0:int(img.shape[0]/2) , 0:int(img.shape[1]/2),:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  simg = img[int(img.shape[0]/2) : img.shape[0] , 0:int(img.shape[1]/2),:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  simg = img[0:int(img.shape[0]/2) , int(img.shape[1]/2) : img.shape[1],:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  simg = img[int(img.shape[0]/2) : img.shape[0] , int(img.shape[1]/2) : img.shape[1],:]   
  simg = cv2.resize(simg,Config.input_hw,cv2.INTER_CUBIC)
  batch.append(simg)
  cla.append(cim)

  return batch, cla
  
if __name__ == "__main__":
  import sys
  sys.exit(1)