import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#load the datset
x,y=load_iris(return_X_y=True)

#set up the feautre 100 row ,2 feats
x=x[:100,:2]
y=y[:100]

#set up the svc 
m=SVC(kernel='linear')

m.fit(x,y)

p=m.predict(x)

#weight and bias 
w=m.coef_[0]
b=m.intercept_[0]

#linespacwe 
x_line=np.linspace(
    x[:,0].min(),
    x[:,0].max(),
    100
    #print till the hundredth one 
)

#hyperplane equn
y1=-(w[0]*x_line+b)/w[1]

#plt 
plt.scatter(
    x[:,0],
    x[:,1],
    c=y
)

plt.plot(
    x_line,
    y1
)

plt.scatter(
    m.support_vectors_[:,0],
    m.support_vectors_[:,1],
    s=100,
    facecolor='none',
    edgecolors='r'

)

plt.xlabel("sepal_length")
plt.ylabel("sepal width")
plt.title("hyperplane and svm")

plt.show()

print("acc",accuracy_score(y,p))
print("report",classification_report(y,p))
print("confusion",confusion_matrix(y,p))