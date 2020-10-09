# importação as bibliotecas
import pandas as pd
import numpy as np

#pré-processamento
from sklearn.preprocessing import LabelEncoder

#Modelo
from sklearn.linear_model import LogisticRegression


#leituras dos dados csv
df = pd.read_csv("http://pycourse.s3.amazonaws.com/temperature.csv")

#transformando o tipo da coluna date para datetime
df['date'] = pd.to_datetime(df['date'])
df.dtypes

#setando o índice
df = df.set_index('date')
#df
#extração de x e y, para temperatura e a classificação
x, y = df[['temperatura']].values, df[['classification']].values
#conversão de y para valores numéricos
le = LabelEncoder() #Estanciando a classe, label enconder
y = le.fit_transform(y.ravel()) #fit- faz um calculo para absorver, e o transforme aplica o calculo.
#print(f'y:\n{y}')#não existe ordem de grandeza

clf = LogisticRegression()
clf.fit(x, y)

#gerando 100 valores de temperatura
#linearmente espaçandos entre 0 e 45
#predição em novos valores de temperaturas
x_test = np.linspace(start=0., stop=45., num=100).reshape(-1, 1)

#predição dessas classes
y_pred = clf.predict(x_test)

#output - criando um dicionario para que possamos usar.
output = {'new_temp': x_test.ravel(), 'new_class': y_pred.ravel()}
output = pd.DataFrame(output)

#print(output)

#sistema automático
def classify_temp():
  """Classifica o input do usuário."""

  ask = True
  while ask:
    #input de temperatura
    temp = input('Insira a temperatura (Graus Celsius): ')

    #transformar para numpy array
    temp = np.array(float(temp)).reshape(-1, 1)

    #realiza classificação
    class_temp = clf.predict(temp)

    #transformação inversa para retornar a string original
    class_temp = le.inverse_transform(class_temp)

    #classificação
    print(f'A classificação da temperatura {temp.ravel()[0]} é:', class_temp[0])

    #perguntar
    ask = input('Nova classificação (y/n): ') == 'y'

#chamando a função, rodando o programa.
classify_temp()