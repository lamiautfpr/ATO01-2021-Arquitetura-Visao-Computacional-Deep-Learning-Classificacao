import os #funcionalidades do sistema.
import argparse #Biblioteca para adicionar argumentos por linha de comando.
from Generators.Generator import Generator
import json
from utils import Cross_val
import datetime

arg = argparse.ArgumentParser() 

#Argumento responsável por receber o caminho do database.
arg.add_argument('-d', '--database', required=True, help='Path/Caminho da base de dados') 

#Argumento responsável por receber o caminho do modelo.
arg.add_argument('-m', '--model', required=True, help='Path/Caminho (SEM BARRA SEPARADO POR PONTO - DEVE ESTAR DENTRO DE MODELS) para o modelo.py')

#Argumento responsável por receber o caminho do arquivo de parametros.
arg.add_argument('-p', '--parameters', help='Path/Caminho para o arquivo de parametros')

#Argumento responsável por receber o caminho do arquivo de output.
arg.add_argument('-o', '--output', help='Path/Caminho em que deseja o resultado (com .json)')

#Argumento responsável por receber a quantia de folds do cross_validaion
arg.add_argument('-cv','--cross_val', type= int ,default=0, help='Cross_validation (numero de FOLDS para sim) e 0 para nao.')

#Argumento responsável por receber o caminho do arquivo de pre-processamento.
arg.add_argument('-pre','--pre_proc', help='Modulo que vai ser usado para pré-processamento.')

#Argumento responsável por recever o caminho do arquivo de pos-processamento.
arg.add_argument('-pos','--pos_proc', help='Modulo de por-processamento')

#Argumento do path de output
arg.add_argument('-op','--output_path', help='output path')

#Cria um objeto que contem atributos para os respectivos argumentos.
args = arg.parse_args()

""" Quando for fazer o teste para o preproc, lembra de verificar o valor do batch_size. """

if os.system('python Testes/Teste_modelo.py -m Models.Model_ResNet50'):
  print("MODELO INCORRETO.")
  import sys
  sys.exit(1)

with open('PARAMS.json') as f:
  Params = json.load(f)

if os.system(f'python Testes/Teste_base_de_imagens.py -d "{args.database}"'):
  import sys
  sys.exit(1)
else:
  Params['QNT_CLA'] = len(os.listdir(args.database))

Params['ARQUIVO'] = args.output_path + '/' + Params['PREFIX'] + '-{}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'

Module_model = __import__(args.model, fromlist=['compiled_model']) #Importa o model
Model = Module_model.compiled_model(Params['INPUT_SHAPE'], Params['QNT_CLA'])

Module_prep = __import__(args.pre_proc, fromlist=['pre_train', 'pre_test'])
pre_train = Module_prep.pre_train
pre_test = Module_prep.pre_test

Gen = Generator(args.database, Params['SPLIT_SIZE'], Params['BATCH_SIZE'], pre_train, pre_test)

Module_pos = __import__(args.pos_proc, fromlist=['evaluate'])
evaluate = Module_pos.evaluate

if args.cross_val > 0:
  for EXP in range(args.cross_val):
    cv = Cross_val(Gen.total_img, 4)

    Gen.Redefine(args.database, Params['SPLIT_SIZE'], Params['BATCH_SIZE'], pre_train, pre_test)

    train, val, tes = cv.get_set_distribuition(EXP)
    
    Gen.set_indices(train, val, tes)

    Model.fit(x = Gen.train_generator(), 
            batch_size= Params['BATCH_SIZE'], 
            epochs=Params['EPOCHS'], 
            steps_per_epoch = Gen.steps_train, 
            validation_data =Gen.val_generator(), 
            validation_steps = Gen.steps_val,
            callbacks = []
            )
    
    evaluate(Model, Gen.test_generator, Gen.eval_generator, Params['ARQUIVO'].format('-FOLD|' + str(EXP)+'--'))
    

else:
  Model.fit(x = Gen.train_generator(), 
            batch_size= Params['BATCH_SIZE'], 
            epochs=Params['EPOCHS'], 
            steps_per_epoch = Gen.steps_train, 
            validation_data =Gen.val_generator(), 
            validation_steps = Gen.steps_val,
            callbacks = []
            )

  evaluate(Model, Gen.test_generator, Gen.eval_generator, Params['ARQUIVO'].format('-'))




# if __name__ == "__main__":
#   main()