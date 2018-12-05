#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:34:56 2018

@author: cassiolm
"""


#importa a biblioteca pandas
import pandas as pd
 
#importando dados
#fonte: https://www.kaggle.com/c/titanic/data

#cria base teste com pandas
test = pd.read_csv("test.csv")
#cria base treino com pandas
train = pd.read_csv("train.csv")
 
#verificar as dimensões dos dataframes
train.shape
test.shape

train.head()
#criando variável para tabela pivot com relação a sexo x sobreviveu
sex_pivot = train.pivot_table(index = 'Sex', values= 'Survived')
#plotando para gráfico de barra
sex_pivot.plot.bar()
#mostra o gráfico
plt.show()

#verificando a coluna Pclass e criando uma variável
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar(color='r') # r para indicar a cor vermelha(red)
plt.show()


#verifica a distribuição de idades no treino e descreve com medidas
train["Age"].describe()


#criando um histograma para visualizar como foi o grau de sobrevivência de acordo com as idades
survived = train[train["Survived"] == 1] #sobrevivencia com base na coluna survived sendo no binário igual a 1
died = train[train["Survived"] == 0] #sobrevivencia com base na coluna survived sendo no binário igual a 0
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50) #sobrevivência com base em age plotando em vermelho
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50) #falecimento em azul, com base em age.
plt.legend(['Survived','Died']) #cria uma legenda para o histograma
plt.show() #plota o gráfico

#O relacionamento parece bem complicado mas analisando com atenção, podemos ver que existe um maior índice de sobrevivência em algumas idades. Para facilitar a vida do modelo de Machine Learning, vamos transformar a variável contínua Age para do tipo categórica. Faremos isso dividindo as idades em intervalos através da função pandas.cut(). Essa função possui dois parâmetros obrigatórios: a coluna que desejamos “cortar” e uma lista de números que definem os limites dos nossos cortes. Também iremos usar um parâmetro opcional, que leva uma lista de nomes para cada parte do corte. Isso facilitará a compreensão de nossos resultados.
#Aliado a tudo isso, vamos usar o método pandas.fillna() para substituir todos os valores faltantes por -0.5 (você já vai entender).  Vamos então segmentar as idades entre os seguintes intervalos:
#Missing, from -1 to 0
#Infant, from 0 to 5
#Child, from 5 to 12
#Teenager, from 12 to 18
#Young Adult, from 18 to 35
#Adult, from 35 to 60
#Senior, from 60 to 100

#para facilitar o trabalho do algoritmo, vamos criar ranges fixos de idades. 
# e ao mesmo tempo vamos tratar os missing values
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)  #preenche todos valores missing por -0.5
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names) #corta a variável age em intervalos (slices), abaixo:
    return df
 
cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
 
train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)


#Verificaremos agora de forma mais agradável o índice de sobrevivência

pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar(color='g')
plt.show()

#É necessário tornar variávels categóricas em numéricas para que o algoritmo consiga considerá-la.

#removendo a relação numerica presente na coluna P class
def create_dummies(df,column_name): #cria a coluna dummy
    dummies = pd.get_dummies(df[column_name],prefix=column_name) #define a variável dummies
    df = pd.concat([df,dummies],axis=1) #concatena a variavel dummy
    return df
 #substitui as colunas categóricas no dataset
for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)
    
#exporta da biblioteca scikitlearn a logistic Regression
from sklearn.linear_model import LogisticRegression

#criando um objeto LogistcRegression
lr = LogisticRegression()

#para treinar o modelo, usamos o método fit
columns = ['Pclass_2', 'Pclass_3', 'Sex_male']
lr.fit(train[columns], train['Survived'])

#aplicar o treinamento para todas as colunas que foram criadas dummy
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
'Age_categories_Missing','Age_categories_Infant',
'Age_categories_Child', 'Age_categories_Teenager',
'Age_categories_Young Adult', 'Age_categories_Adult',
'Age_categories_Senior']
 
lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])

#calibragem do modelo
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
verbose=0, warm_start=False)


#avaliando o modelo e criando a variável holdout
holdout = test

#importa a train_test_split do scikit
from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
all_X, all_y, test_size=0.20,random_state=0)

#utilizacao do metodo predict() 
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

#importa a bilbioteca com a funcao de acuracia
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions) #verifica a acuracia do modelo
#imprime na tela
print(accuracy)

#mportou o cross validation do scikit learn
from sklearn.model_selection import cross_val_score
#rodou as funções
lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()
 
print(scores)
print(accuracy)

#fazendo previsões usando novos dados
lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])

### Segunda tentativa

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv") 

#retirando os dados irrelevantes
#comando drop para deletar as colunas completamente do dataframe e substituir no csv
train.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)
test.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)


#fazendo uso dos dummies de novo
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

#Verificação de valores nulos, de ordem descrescente com os 10 primeiros
new_data_train.isnull().sum().sort_values(ascending = False).head(10)


#tratando valores nulos encontrados pela média
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace = True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace = True)

#Verificação de valores nulos, de ordem descrescente com os 10 primeiros
new_data_test.isnull().sum().sort_values(ascending = False).head(10)

#tratamento do valor nulo coluna fare pela média
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace = True)

#separado as features para a criação do modelo
X = new_data_train.drop("Survived", axis = 1) #tirando apenas a coluna target 
y = new_data_train["Survived"] # colocando somente a coluna target

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#parametrizando o tamanho a arvore de decisao
tree = DecisionTreeClassifier(max_depth = 3, random_state = 0)
tree.fit(X,y)

#avaliando o modelo
tree.score(X,y)

#Enviando a previsão para o Kaggle
previsao = pd.DataFrame()
previsao["PassengerId"] = new_data_test["PassengerId"]
previsao["Survived"] = tree.predict(new_data_test)

#importando para CSV
previsao.to_csv('previsao.csv',index = False)










