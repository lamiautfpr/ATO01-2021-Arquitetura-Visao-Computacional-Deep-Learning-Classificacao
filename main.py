import os
import json
import argparse
import datetime
from Modules.Utils.utils import Cross_val
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import CSVLogger
from Modules.Generators.Generator import Generator

arg = argparse.ArgumentParser() 

arg.add_argument('-d', '--database', required=True,           help='Path/Caminho da base de dados') 
arg.add_argument('-m', '--model', required=True,              help='Nome do model (DEVE ESTAR DENTRO DE MODELS)')
arg.add_argument('-p', '--parameters',                        help='Path/Caminho para o arquivo de parametros')
arg.add_argument('-o', '--output',                            help='Path/Caminho em que deseja a saída dos resultados.')
arg.add_argument('-cv','--cross_val', type=int, default=0,    help='Cross_validation (numero de FOLDS para sim) e 0 para nao.')
arg.add_argument('-pre','--pre_proc',                         help='Modulo que vai ser usado para pré-processamento.')
arg.add_argument('-pos','--pos_proc',                         help='Modulo de pos-processamento')
arg.add_argument('-test','--teste', type=bool, default=True,  help='Se quer que os arquivos passem por testes ou não.')


#Cria um objeto que contem atributos para os respectivos argumentos.
args = arg.parse_args()

#Aplica os testes.
if args.teste:
  if os.system(f'python Testes/Teste_modelo.py -m Modules.Models.{args.model}'):
    print("MODELO INCORRETO.")
    import sys
    sys.exit(1)
else:
  pass

#Realiza as importações de Modelo, função de pré e pós-processamento.
Module_model = __import__('Modules.Models.' + args.model, fromlist=['compiled_model'])

Module_prep = __import__('Modules.Prep.' + args.pre_proc, fromlist=['pre_train', 'pre_test'])
pre_train = Module_prep.pre_train
pre_test = Module_prep.pre_test

Module_posp = __import__('Modules.Posp.' + args.pos_proc, fromlist=['evaluate'])
evaluate = Module_posp.evaluate

#Abre o json com os parametros necessários.
with open(args.parameters) as f:
  Params = json.load(f)

# Params['ARQUIVO'] = args.output + '/' + Params['PREFIX'] + '-' +datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/' + Params['PREFIX'] + '-{}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'
Params['ARQUIVO'] = os.path.join(args.output, 
                                Params['PREFIX'] + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                Params['PREFIX'] + '-{}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'
                                )

os.mkdir(os.path.dirname(Params['ARQUIVO']))

#Instancia o gerador de imagens.
Gen = Generator(args.database, Params, pre_train, pre_test)

#Instancia o modelo.
Model = Module_model.compiled_model(Params['INPUT_SHAPE'], Params['QNT_CLA'])


#Treino e validação.
if args.cross_val > 0:
  for EXP in range(args.cross_val):
    path_out = os.path.join(os.path.dirname(Params['ARQUIVO']), 'FOLD-{}'.format(EXP))
    os.mkdir(path_out)

    CsvLog = CSVLogger(os.path.join(path_out, 'logs-{}.csv'.format(EXP)))

    CV = Cross_val(Gen.total_img, args.cross_val)

    Gen.Redefine(args.database, Params, pre_train, pre_test)

    train, val, tes = CV.get_set_distribuition(EXP)
    
    Gen.set_indices(train, val, tes)

    Model.fit(x = Gen.train_generator(), 
            batch_size= Params['BATCH_SIZE'], 
            epochs=Params['EPOCHS'], 
            steps_per_epoch = Gen.steps_train, 
            validation_data =Gen.val_generator(), 
            validation_steps = Gen.steps_val,
            callbacks = [CsvLog]
            )
    
    evaluate(Model, Gen.test_generator, Gen.eval_generator, os.path.join(path_out,'FOLD-' + str(EXP)))
    
    save_model(Model, os.path.join(path_out,'MODEL-' + str(EXP) + '.tf'),save_format="tf")

else:
  CsvLog = CSVLogger(os.path.join(os.path.dirname(Params['ARQUIVO']), 'logs.csv'))

  Model.fit(x = Gen.train_generator(), 
            batch_size= Params['BATCH_SIZE'], 
            epochs=Params['EPOCHS'], 
            steps_per_epoch = Gen.steps_train, 
            validation_data =Gen.val_generator(), 
            validation_steps = Gen.steps_val,
            callbacks = [CsvLog]
            )

  evaluate(Model, Gen.test_generator, Gen.eval_generator, Params['ARQUIVO'].format('-'))

  save_model(Model, os.path.join(os.path.dirname(Params['ARQUIVO']), "MODEL-W.tf"))