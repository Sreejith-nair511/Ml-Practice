import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#load the datset
df=pd.read_csv("BostonHousing.csv").dropna()

x=df.drop("medv",axis=1)
y=df["medv"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

m=LinearRegression()
m.fit(x_train,y_train)

p=m.predict(x_test)

print("mae",np.mean(abs(y_test-p)))
print("mse",np.mean((y_test-p)**2))
print("rmse",np.sqrt(np.mean((y_test-p)**2)))
print("mape",np.mean(abs((y_test-p)/y_test))*100)
