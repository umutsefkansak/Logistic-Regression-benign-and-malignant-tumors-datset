

import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

data.info()

data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
data.info()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]


y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%% normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#%%

def initialize(dimension):
    
    w =np.full((dimension,1),0.01)
    b = 0.0
    return w,b


def sigmoid(z):
    y_head = 1/(1+ np.exp(-z))
    return y_head

#%%

def forward_backward_propagation(w,b,x_train,y_train):
    
    z = np.dot(w.T, x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] 
    
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients

#%%
def update(w,b,x_train,y_train,learning_rate,number_of_iterations):
    
    for i in range(number_of_iterations):
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
        if i % 20 == 0:
            print(i,". iteration cost: ",cost)
    
    parameters = {"weight":w,"bias":b}
    return parameters
#%%

def prediction(w,b,x_test):
    
    y_head = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    
    return y_prediction

#%%



def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,number_of_iterations):
    
    
    dimension = x_train.shape[0]
    w,b = initialize(dimension)
    
    
    parameters = update(w, b, x_train, y_train, learning_rate, number_of_iterations)
    
    y_prediction = prediction(parameters["weight"],parameters["bias"], x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - y_test)) * 100))
    
#%%

logistic_regression(x_train, x_test, y_train, y_test, 1, 1000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

