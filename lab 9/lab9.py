import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris


#load the dataset
x,y=load_iris(return_X_y= True)

#setup the kmeans 
k=KMeans(n_clusters=3,random_state=0)
k1=k.fit_predict(x)

#setup the gmm
g=GaussianMixture(n_components=3,random_state=0)
g1=g.fit_predict(x)


#label
print("kmeans")
print(k1)

print("gmm")
print(g1)


plt.scatter (x[:,0],
    x[:,1],
    c=k1)

plt.title("kmeans")
plt.show()

plt.scatter(
    x[:,0],
    x[:,1],
    c=g1
)

plt.title("gmm")
plt.show()