import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes


dataset = load_diabetes()

X = dataset.data
Y = dataset.target

print(X.shape) # (442,10)
print(Y.shape) # (442,)

# pca = PCA(n_components=5)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)
# print(sum(pca_evr))

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)


a = np.argmax(cumsum>=0.94)
print(cumsum>=0.94)
print(a)