import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/kaggle/input/linear-regression-dataset/Linear Regression - Sheet1.csv")
df.head()
plt.plot(df)

plt.scatter(df["X"], df["Y"], color='red', label='Data points')

X = df.iloc[:,0:1]
Y = df.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


lr = LinearRegression()

lr.fit(X_train,Y_train)

print(lr.predict(X_test.iloc[3].values.reshape(1,1)))

plt.scatter(df["X"], df["Y"], color='red', label='Data points')
plt.plot(X_test,lr.predict(X_test), color='black')