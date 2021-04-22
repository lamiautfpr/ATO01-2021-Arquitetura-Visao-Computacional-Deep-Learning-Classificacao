import cv2

def pre_train(batch, cla, img, cim):
  """
    Função de preprocessamento que redimensiona a imagem para o tamnho de 224x224.
      batch = Lista de imagens no tamanho de um batch, aqui deve fazer append das imagens.
      cla = Lista de classes das imagens, deve ter o mesmo tamanho do batch, e o indice da imagem no batch deve corresponder com o indice nessa lista.
      img = Imagem.
      cim = Classe da imagem
  """
  img = cv2.resize(img, (224,224), cv2.INTER_CUBIC)
  batch.append(img)
  cla.append(cim)

def pre_test(img, cim):
  img = cv2.resize(img, (224,224), cv2.INTER_CUBIC)

  return img, cim

if __name__ == "__main__":
  import sys
  sys.exit(1)