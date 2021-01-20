import os
import numpy as np
import json

def evaluate(Model, gen_test, gen_eval, arquivo):
  Model.evaluate(gen_eval()) #Evaluation do model.

  
  list_pred = []
  list_or = []
  for (img,cim) in gen_test():
    pred = Model.predict(img)
    '''
      Essa parte que deve ser mudada. Aqui é implementado o tratamento da predição.
    ''' 
    list_pred.append(pred[0].tolist())
    list_or.append(cim[0].tolist())
  
  dic = {"pred" : list_pred , "real" : list_or}
    
  with open(arquivo, 'w') as f:
    json.dump(dic, f)
    # f.write(dic)
