def prepro(batch, cla, img, cim):
  """
    Função de preprocessamento simples, retorna a imagem normal.
      batch = Lista de imagens no tamanho de um batch, aqui deve fazer append das imagens.
      cla = Lista de classes das imagens, deve ter o mesmo tamanho do batch, e o indice da imagem no batch deve corresponder com o indice nessa lista.
      img = Imagem.
      cim = Classe da imagem
  """

  batch.append(img)
  cla.append(cim)

if __name__ == "__main__":
  import sys
  sys.exit(1)