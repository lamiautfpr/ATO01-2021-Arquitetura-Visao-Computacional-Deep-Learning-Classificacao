import os
import sys
import argparse

arg = argparse.ArgumentParser()

arg.add_argument('-p', '--posp', required=True, help='Pós-processamento a ser testado.')

args = arg.parse_args()

#Atualiza um pythonpath para que eu possa pegar os modulos diretamente de src, e nao da pasta local.
sys.path.insert(0, os.path.abspath(os.curdir)) 

Module_posp = __import__(args.posp, fromlist=['*'])

if 'evaluate' not in dir(Module_posp):
  print("IMPLEMENTE TODAS AS FUNCOES NECESSÁRIAS PARA VALIDAÇÃO DO ARQUIVO.")
  sys.exit(1)

sys.exit(0)
