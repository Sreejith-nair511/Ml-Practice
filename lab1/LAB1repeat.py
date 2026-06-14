import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


#load the dataset
x,y=load_diabetes(return_X_y=True)

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#col and rows 
m,n=x_train.shape

#set weight and bias 
w=np.zeros(n)
b=0

#learning rate 
lr=0.001

for i in range(3000):
    p=x_train@w+b
    #update the weights
    w-=0.01*(2/m)*(x_train.T@(p-y_train))
    b-=0.01*(2/m)*sum(p-y_train)

#trraining loop
    p=x_test@w+b

print("mse",mean_squared_error(y_test,p))

print("r2",r2_score(y_test,p))

plt.scatter(y_test,p)
plt.plot(y_test,y_test,'--r')
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
