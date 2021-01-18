import os
import argparse
import sys

arg = argparse.ArgumentParser()

arg.add_argument('-m', '--model', required=True, help='Modelo a ser testado.') #Cria argumento para passarmos o modelo a ser testado.

args = arg.parse_args() #Cria um objeto em que os atributos sao os argumentos passados.

#Atualiza um pythonpath para que eu possa pegar os modulos diretamente de src, e nao da pasta local.
sys.path.insert(0, os.path.abspath(os.curdir)) 

Module_model = __import__(args.model, fromlist=['compiled_model'])  #Importa o modulo

if 'compiled_model' not in dir(Module_model): #Verifica se o modulo tem a função compiled_model.
  print('NÃO EXISTE A FUNÇÃO COM O NOME CORRETO DENTRO NO ARQUIVO.') 
  sys.exit(1) 

compiled_model = Module_model.compiled_model #Termina de importar a função.

#Dicionario com o formato padrão da função.
dic_correto = {'INPUT_SHAPE': "<class 'list'>", 'QNT_CLASS': "<class 'int'>", 'return': "<class 'tensorflow.python.keras.engine.training.Model'>"}

if 'INPUT_SHAPE' not in compiled_model.__annotations__: #Verifica se a função recebe o parametro INPUT_SHAPE.
  print('(INPUT_SHAPE) A FUNCAO NAO CONTEM UM DOS PARAMETROS OBRIGATÓRIO.')
  sys.exit(1)

if 'QNT_CLASS' not in compiled_model.__annotations__: #Verifica se a funlção recebe o parametro QNT_CLASSES.
  print('(QNT_CLASS) A FUNCAO NAO CONTEM UM DOS PARAMETROS OBRIGATÓRIO.')
  sys.exit(1)

if 'return' not in compiled_model.__annotations__: #Verifica se a função retorna alguma coisa.
  print('(return) A FUNCAO NAO RETORNA.')
  sys.exit(1)

if not dic_correto['return'] == str(compiled_model.__annotations__['return']): #Verifica se o tipo do retorno é o correto.
  print('A FUNCAO NAO RETORNA UM OBJETO VAIDO.')
  sys.exit(1)

sys.exit(0)