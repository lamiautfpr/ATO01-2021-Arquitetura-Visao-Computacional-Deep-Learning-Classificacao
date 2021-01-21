import os
import numpy as np
import json
from utils import matrix_confusion

def evaluate(Model, gen_test, gen_eval, arquivo):
  eval = Model.evaluate(gen_eval(), return_dict=True) #Evaluation do model.
  
  with open(arquivo[0:-5]  + '-eval.json', 'w') as f:
    json.dump(eval, f)
  
  list_pred = []
  list_or = []
  for (img,cim) in gen_test():
    pred = Model.predict(img)

    #/////////////////////////CODE FROM HERE//////////////
    
    list_pred.append(pred[0].tolist()) #0 por que tava retornando uma lista de lista.
    list_or.append(cim[0].tolist())
  
  dic = {"pred" : list_pred , "real" : list_or}
    
  with open(arquivo, 'w') as f:
    json.dump(dic, f)

  matrix_confusion(list_pred, list_or, arquivo[0:-5])  

