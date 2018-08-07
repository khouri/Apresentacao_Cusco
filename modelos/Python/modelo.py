#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 06:22:42 2018

@author: adilsonlopeskhouri
Código obtido em: https://blog.paperspace.com/getting-started-with-scikit-learn/
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#carrega o dataset em memoria
iris_dataset = load_iris()
X, y = iris_dataset.data, iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=31)


#cria uma estrutura de dados do tipo: pandas dataframe
df = pd.DataFrame(data=np.c_[X,y], columns=iris_dataset['feature_names'] + ['target'])
df.sample(frac=0.1)


#pre processamento de dados    
#Remove a média e escala para variancia unitaria
#z = (dados - media) / desvio padrao 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#treina classificador com parametros tipicos do SVM
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X_train, y_train)


#predicao de treino e teste
y_pred_train = clf.predict(X_train) 
y_pred_test = clf.predict(X_test)


#calculo de acuracia treino e teste
# Acuracia = (Verdadeiro Positivos + Verdadeiro Negativos)/Total
acc_train = accuracy_score(y_train, y_pred_train) 
acc_test = accuracy_score(y_test, y_pred_test)


#Criamos a matriz de confusao baseado no conjunto de testes
confusion_matrix = confusion_matrix(y_test,y_pred_test) # Returns a ndarray
cm_df = pd.DataFrame(confusion_matrix, 
                     index = [idx for idx in ['setosa', 'versicolor', 'virginica']],
                     columns = [col for col in ['setosa', 'versicolor', 'virginica']])


#plotamos uma matriz de confusao ternaria
plt.figure(figsize = (10,7))
sns.heatmap(cm_df, annot=True)

