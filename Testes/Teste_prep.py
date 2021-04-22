import os
import sys
import argparse

arg = argparse.ArgumentParser()

arg.add_argument('-p', '--prep', required=True, help='Pré-processamento a ser testado.')

args = arg.parse_args()

#Atualiza um pythonpath para que eu possa pegar os modulos diretamente de src, e nao da pasta local.
sys.path.insert(0, os.path.abspath(os.curdir)) 

Module_prep = __import__(args.prep, fromlist=['*'])

if 'pre_test' not in dir(Module_prep) and 'pre_train' not in dir(Module_prep):
  print("IMPLEMENTE TODAS AS FUNCOES NECESSÁRIAS PARA VALIDAÇÃO DO ARQUIVO.")
  sys.exit(1)

sys.exit(0)
