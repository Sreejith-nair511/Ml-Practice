import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score 


#linear regression 
#it is for cont value ,the line formular 
#y=mx+c slope ,intercept and gradient 
#mse formula 

x,y=load_diabetes(return_X_y= True)

#u will load the dataset

#split the dataset

x_train,x_test,y_train,y_test=train_test_split( x,y,test_size=.2)
    


#number of row and col
m,n=x_train.shape

#weight bais 
w=np.zeros(n);
b=0

#learning rate
lr=0.001

#the training loop
for i in range(3000):
    p=x_train@w+b

    w -=0.01*(2/m)*(x_train.T@(p-y_train))
    b -=0.01*(2/m)*sum(p-y_train)
    
    
   
 #final 
p=x_test@w+b

print("mse",mean_squared_error(y_test,p))
print("r2 score",r2_score(y_test,p))


plt.scatter(y_test,p)
plt.plot(y_test,y_test,'--r')
plt.show()




