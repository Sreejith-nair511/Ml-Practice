import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

#load the dataset
x,y=load_iris(return_X_y=True)

#set up kmeans
k=KMeans(n_clusters=3,random_state=0)
k1=k.fit_predict(x)


#set up gmm
g=GaussianMixture(n_components=2,random_state=0)
g1=g.fit_predict(x)


#label print 
print("k label")
print(k1)

print("g label")
print(g1)


#plot the graph 
plt.scatter(x[:,0],
    x[:,1],
    c=k1)

plt.title("k means ")
plt.show()

plt.scatter(x[:,0],
    x[:,1],
    c=g1)

plt.title("gmm ")
plt.show()