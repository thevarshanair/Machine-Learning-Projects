# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 10:46:14 2018

@author: VarshaNair
"""

# Importing Keras Sequential Model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# Initializing the seed value to a integer.
seed = 7

np.random.seed(seed)

# Loading the data set (PIMA Diabetes Dataset)
dataset = pd.read_csv('diabetes.csv')

# Loading the input values to X and Label values Y using slicing.
X = dataset.iloc[:,0:8].values
Y = dataset.iloc[:,8].values

# Initializing the Sequential model from KERAS.
model = Sequential()

# Creating a 16 neuron hidden layer with Linear Rectified activation function.
model.add(Dense(16, input_dim=8, kernel_initializer='uniform', activation='relu'))

# Creating a 8 neuron hidden layer.
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

# Adding a output layer.
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



# Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fitting the model
model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))