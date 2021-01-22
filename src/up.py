import os
import argparse 

arg = argparse.ArgumentParser()

arg.add_argument('-t', '--type', required=True, help='prep = pre-processamento | posp = pos-processamento | model = Modelo.')
arg.add_argument('-p', '--path', required=True, help='Caminho do arquivo.')

args = arg.parse_args()

if os.path.basename(args.path) == '*' and '*' in os.path.basename(args.path):
  print('NOME INVALIDO.')
  sys.exit(1)

if args.type == 'prep':
  if os.path.basename(args.path) in os.listdir('prep'):
    print('TROQUE O NOME DO ARQUIVO.')
    sys.exit(1)

  os.system("Copy \"{}\" \"{}\"".format(args.path, os.path.join(os.path.abspath('.'),args.type)))
  if os.system('python Teste/Teste_prep.py -p Prep.{}'.format(os.path.basename(args.path)))
    os.system('del Posp/{}'.format(os.path.basename(args.path)))
    sys.exit(1)
  sys.exit(0)

if args.type == 'posp':
  if os.path.basename(args.path) in os.listdir('posp'):
    print('TROQUE O NOME DO ARQUIVO.')
    sys.exit(1)
    
  os.system("Copy \"{}\" \"{}\"".format(args.path, os.path.join(os.path.abspath('.'),args.type)))
  if os.system('python Teste/Teste_posp.py -p Posp.{}'.format(os.path.basename(args.path)))
    os.system('del Posp/{}'.format(os.path.basename(args.path)))
    sys.exit(1)
  sys.exit(0)

if args.type == 'Models':
  if os.path.basename(args.path) in os.listdir('Models'):
    print('TROQUE O NOME DO ARQUIVO.')
    sys.exit(1)
    
  os.system("Copy \"{}\" \"{}\"".format(args.path, os.path.join(os.path.abspath('.'),args.type + 's')))
  if os.system('python Teste/Teste_modelo.py -p Models.{}'.format(os.path.basename(args.path)))
    os.system('del Models/{}'.format(os.path.basename(args.path)))
    sys.exit(1)
  sys.exit(0)


print("Copy \"{}\" \"{}\"".format(args.path, os.path.join(os.path.abspath('.'),args.type + 's') ))


#(                             os.path.join()
# os.system("Copy \"{}\" \"{}\"".format(args.path,os.path.abspath(".") + "\\" + args.type + "\\" )) 
