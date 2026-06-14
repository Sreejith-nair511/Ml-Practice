from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 


#load the dataset
x,y=load_iris(return_X_y=True)

#standarsize the data 
x=StandardScaler().fit_transform(x)

#pca one 
pca=PCA(n_components=2)

#fior the pca one 
xp=pca.fit_transform(x)

#vaience roatio
print(pca.explained_variance_ratio_)

#graph 
plt.scatter(xp[:,0],xp[:,1],c=y)
plt.ylabel("pc2")
plt.xlabel("pc1")
plt.title("pca")

plt.show()
