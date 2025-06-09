# Importing necessary libraries 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SimpleLinearRegression:
    def __init__(self):
        self.m = None  # slope
        self.b = None # intercept

    def fit(self,X_train,y_train):
        num = 0;
        den = 0;
    
        for i in range(X_train.shape[0]):
            num += (X_train[i] - np.mean(X_train)) * (y_train[i] - np.mean(y_train))
            den += (X_train[i] - np.mean(X_train)) ** 2
        self.m = num / den
        self.b = np.mean(y_train) - self.m * np.mean(X_train)

    def predict(self, X_test):
        return self.m * X_test + self.b


# Load the dataset
df = pd.read_csv('data/linear_regression_data.csv') 

X = df.iloc[:,0].values
y = df.iloc[:,1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lf = SimpleLinearRegression()
lf.fit(X_train, y_train)
y_pred = lf.predict(X_test)

