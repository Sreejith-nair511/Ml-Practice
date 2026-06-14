#evalution of the svm 


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


#load the dataset
x,y=load_iris(return_X_y=True)

#split the dataset 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#setup the svc 
m=SVC(kernel='linear')

m.fit(x_train,y_train)

p=m.predict(x_test)


#evalution of everything 
print("acc",accuracy_score(y_test,p))

print("recall",recall_score(y_test,p,average="macro"))

print("f1 score",f1_score(y_test,p,average="macro"))

print(classification_report(y_test,p))

print(confusion_matrix(y_test,p))

ConfusionMatrixDisplay.from_predictions(y_test,p)

plt.show()
