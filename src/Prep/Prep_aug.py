import cv2

def pre_train(batch, cla, img, cim):
  """
    Função de preprocessamento que redimensiona a imagem para o tamnho de 224x224.
      batch = Lista de imagens no tamanho de um batch, aqui deve fazer append das imagens.
      cla = Lista de classes das imagens, deve ter o mesmo tamanho do batch, e o indice da imagem no batch deve corresponder com o indice nessa lista.
      img = Imagem.
      cim = Classe da imagem
  """
  rot = alb.augmentations.transforms.Rotate(always_apply= True)
  gns = alb.augmentations.transforms.GaussNoise(always_apply=True)
  bri = alb.augmentations.transforms.RandomBrightness(always_apply=True)

  batch.append(img)
  cla.append(cim) 

  aimg = rot.apply(img,np.random.randint(10,70))
  batch.append(aimg)
  cla.append(cim) 
  
  aimg = bri.apply(img,np.random.uniform(0.6,0.9))
  batch.append(aimg)
  cla.append(cim) 

  aimg = gns.apply(img)
  batch.append(aimg)
  cla.append(cim) 

def pre_test(img, cim):
  return img, cim

if __name__ == "__main__":
  import sys
  sys.exit(1)