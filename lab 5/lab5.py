from sklearn.datasets import  load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#load dataset
x,y=load_iris(return_X_y=True)

#split the datset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#dtc
m=DecisionTreeClassifier(criterion='entropy')

#fit this 
m.fit(x_train,y_train)

p=m.predict(x_test)

print("acc",accuracy_score(y_test,p))

plot_tree(m)
plt.show()