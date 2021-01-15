import os #funcionalidades do sistema.
import argparse #Biblioteca para adicionar argumentos por linha de comando.
from Models.Model_ResNet50 import compiled_model

arg = argparse.ArgumentParser() 

#Argumento responsável por receber o caminho do database.
arg.add_argument('-d', '--database', required=True, help='Path/Caminho da base de dados') 

#Argumento responsável por receber o caminho do arquivo de parametros.
arg.add_argument('-p', '--parameters', help='Path/Caminho para o arquivo de parametros')

#Argumento responsável por receber o caminho do arquivo de output.
arg.add_argument('-o', '--output', help='Path/Caminho em que deseja o resultado (com .json)')

#Cria um objeto que contem atributos para os respectivos argumentos.
args = arg.parse_args()




# if __name__ == "__main__":
#   main()