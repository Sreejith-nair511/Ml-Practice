import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


#load the dataset
x,y=load_iris(return_X_y=True)

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#the dtc 
m=DecisionTreeClassifier(criterion='entropy')

m.fit(x_train,y_train)

p=m.predict(x_test)

print("acc",accuracy_score(y_test,p))

plot_tree(m)
plt.show()