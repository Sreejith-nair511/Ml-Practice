from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

x,_=make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)

m=DBSCAN(eps=0.3,min_samples=5)

p=m.fit_predict(x)

#scatter
plt.scatter(
    x[:,0],
    x[:,1],
    c=p
)

plt.title("dscan")

plt.show()

print("Unique Clusters =",set(p))