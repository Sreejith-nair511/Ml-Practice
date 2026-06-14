import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

#load the dataset
x,y=load_iris(return_X_y=True)

#run the loop 
from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X,y=load_iris(return_X_y=True)

for k in [1,3,5]:

    acc=cross_val_score(
        KNeighborsClassifier(k),
        X,
        y,
        cv=LeaveOneOut()
    ).mean()*100

    print("K =",k,"Accuracy =",acc)