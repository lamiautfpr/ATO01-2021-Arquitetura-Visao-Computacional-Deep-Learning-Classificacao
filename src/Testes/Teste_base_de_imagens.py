import os
import argparse
import sys

arg = argparse.ArgumentParser()

#Argumento responsável pelo caminho da base de dados a ser testada.
arg.add_argument('-d', '--dataset', required=True, help='Path/Caminho para teste e validação')

args = arg.parse_args() #Cria um objeto contendo atributos com os argumentos.

os.chdir(args.dataset) #Abre o caminho a ser testado.

diretorios = os.listdir('.') #Lista os diretórios que se encontram no path atual.

#Esse bloco verifica se temos files soltos na pasta, o que nao é permitido.
for d in diretorios: #Percorre cada diretorio da pasta.
  if os.path.isfile(d): #Verifica se o caminho representa um arquivo ou nao.
    print('DATABASE SEM ARVORE VALIDA. SOMENTE PASTAS DAS CLASSES NO DIR, NADA MAIS.')
    sys.exit(1) #Retorna 1, para que possamos finalizar o programa, ou tomar outra ação (versão 2.0).

#Esse bloco verifica se temos apenas imagens dentro das classes.
ext = ['png', 'jpg', 'jpeg'] #Extensões que vao ser permitidas.

for d in diretorios: #Percorre os diretorios.
  for f in os.listdir(d): #Percorre os files que temos dentro de cada pasta/classe.
    if not os.path.isfile(d + '/' + f): #Verifica se é um arquivo
      print('DATABASE SEM ARVORE VALIDA. SUBPASTAS NAO DEVEM EXISTIR COM AS FOTOS')
      sys.exit(1) #Retorna 1, para que possamos finalizar o programa, ou tomar outra ação (versão 2.0).

    if f.split('.')[-1] not in ext: #Verifica se existe apenas arquivos de extensão valida.
      print('DATABASE SEM ARVORE VALIDA. SUBPASTAS NAO DEVEM EXISTIR COM AS FOTOS')
      sys.exit(1) #Retorna 1, para que possamos finalizar o programa, ou tomar outra ação (versão 2.0).

sys.exit(0) #saida de que está tudo bem.
