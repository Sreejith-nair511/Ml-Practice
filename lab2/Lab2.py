import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay

#load the datset
x,y=load_breast_cancer(return_X_y=True)

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#col,row
m,n=x_train.shape

#set thge weight and bais 
w=np.zeros(n)
b=0
#set the sigmoid funtion 
def sigmoid(z):
    return 1/(1+np.exp(-z))#formula for prob conversion
#the learning rate 
lr=0.001

#the trianing loop
for i in range(1000):
    p=sigmoid(x_train@w+b)

    w-=lr*(1/m)*(x_train.T@(p-y_train))
    b-=lr*(1/m)*(sum(p-y_train))
#training loop set 

p=sigmoid(x_test@w+b)

p=np.where(p>0.5,1,0)

print("accuracy",accuracy_score(y_test,p))



ConfusionMatrixDisplay.from_predictions(y_test,p)
plt.show()


    


