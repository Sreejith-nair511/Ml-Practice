from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 


#load the  dataset
x,y=load_iris(return_X_y=True)

#std it 
x=StandardScaler().fit_transform(x)

#pca componet 
pca=PCA(n_components=2)

#fir that pca 
xp=pca.fit_transform(x)

#print the pca vairence ratio
print(pca.explained_variance_ratio_)


#graph
plt.scatter(xp[:,0],xp[:,1],c=y)
plt.title("pca")
plt.ylabel("pc2")
plt.ylabel("pc1")

plt.show()