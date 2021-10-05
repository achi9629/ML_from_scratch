# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 17:39:10 2021

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('headbrain.csv')
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size = 0.2, 
                                                    random_state = 1234,
                                                    shuffle=True)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
obj = LinearRegression()
obj.fit(X_train, Y_train)
Y_hat = obj.predict(X_test)
print(np.mean(Y_test - Y_hat))

plt.scatter(X_train, Y_train, marker = 'o', c = 'c')
plt.scatter(X_test, Y_test, marker = 'o', c = 'm')
plt.plot(X_test, Y_hat, 'k')
plt.show()
