import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix


#load the dataset
x,y=load_breast_cancer(return_X_y=True)

#split the datset 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#col row
m,n=x_train.shape

#set up the leanrinng rate
lr=0.01

#setup the weight and bias
w=np.zeros(n)
b=0


#sigmoiod defiotnon for prob
def sigmoid(z):
    return (1/1+np.exp(-z))


#training loop
for i in range(3000):
    p=sigmoid(x_train@w+b)

    w=lr*(1/m)*(x_train.T@(p-y_train))
    
    b=lr*(1/m)*(sum(p-y_train))

#trainig loop done 

p=sigmoid(x_test@w+b)

#fix the range 
p=np.where(p>0.5,1,0)

#priont rthe things 
print("acc",accuracy_score(y_test,p))
print(confusion_matrix(y_test,p))

ConfusionMatrixDisplay.from_predictions(y_test,p)

plt.show()
