from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#load the dataset
x,y=load_iris(return_X_y=True)

#run the loop 
for k in [1,3,5]:
    
    print
    (
        k,
        cross_val_score(
            KNeighborsClassifier(k),
            x,
            y,
            cv=LeaveOneOut()
        ).mean()*100)
    
print("k",k)

print("new acc",accuracy_score)