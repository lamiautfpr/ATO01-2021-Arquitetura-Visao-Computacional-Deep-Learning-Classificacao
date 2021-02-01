# Ambiente de Visão Computacional

Esse ambiente tem como objetivo diminuir a necessidade de código para experimentação rápidas na area de visão computacional, ainda tem a necessidade de diversos ajustes que serão feitos com o tempo.

## Como utilizar

O Google Colab é uma boa opção para quem nao tem acesso a GPUs para treinamento:

```cmd
!python -m Model_nomemodelo -pre nomepreprocessamento -pos nomeposprocessamento -p nomeparametros.json -o pathoutput
``` 

O exemplo acima mostra uma execução simples, mas tambem existe a possibilidade de Cross Validation:

```cmd
!python -m Model_nomemodelo -pre nomepreprocessamento -pos nomeposprocessamento -p nomeparametros.json -o pathoutput -cv numfolds
```

O arquivo de 'PARAMS.json' é um exemplo dos hyperparametros que devem estar nesse arquivo.

```json
{
  "PREFIX" : "EXP",
  "BATCH_SIZE" : 32,
  "EPOCHS" : 3,
  "INPUT_SHAPE" : [224,224,3],
  "SPLIT_SIZE" : [0.8,0.1,0.1]
}
```

PREFIX = Prefixo do output.
SPLIT_SIZE = As porcentagens de treino, validation, teste.



## Meus modelos e funções

Existe a possibilidade de fazer um "upload" de modelos a parte, e tambem de funções de pré e pós-processamento, para isso utilize o arquivo 'up.py' que fara os testes básicos para que seu arquivo possa ser passado para o ambiente. 

```cmd
up.py --type model --path Caminho/do/arquivo
```
```
--type:
- prep = pré-processamento.
- posp = pós-processamento.
- model = modelo.
```


Depois que 'up.py' passou seu modelo para o ambiente, basta colocar o nome do arquivo (sem extensão) no parametro -m da função 'main.py'.

Os modelos sempre devem ter a função 'compiled_model(INPUT_SHAPE:list, QNT_CLASS:int)-> tf.keras.Model' retornando um modelo compilado.

Os pré-processamentos sempre devem ter as funções 'pre_train(batch, cla, img, cim)' sem retorno, apenas adicionando ao batch e cla, e 'pre_test(img, cim)' retornando a img e a classe. 
```
img = Imagem
cim = Classe da imagem
batch = Lista onde deve ser feito o append das imagens.
cla = Lista onde deve ser feito o append da classe da imagem.
```
Os pós-processamentos sempre devem ter a função 'evaluate(Model, gen_test, gen_eval, arquivo)'
```
Model = Modelo
gen_test = Gerador de teste, faz yield de imagem por imagem.
gen_eval = Gerador de validação, faz yield em batchs.
arquivo = caminho que vai ser salvo os arquivos de output, basta apenas mudar a extensão (arquivo[0:-5] + extsão)
```
## Erros e possíveis soluções
Ainda nao passei por todos os erros, mas alguns sao:

- pasta .pynb_checkpoints no Colab = Exclua a pasta se ela estiver em sua base de dados. Verifique pelo drive.

Caso tenha algum problema verifique sempre se o formato das funções estão corretos.

## Melhorias futuras
- Melhorar os testes dos arquivos.
- Melhorar a visualização de resultados (API e web application em desenvolvimento).
- Prevenir mais erros.
