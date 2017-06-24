# SVC with linear Kernel
# using the iris data set

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import data
iris = datasets.load_iris()
X = iris.data[:, :2] # we'll only  use 2 features

y = iris.target

# Create an instance of SVM and fit our data.
# We do not scale our data since we want to plot the support vectors
# we remain in a 2D view

C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1).fit(X, y)

# create a mesh to plot in
if __name__ == '__main__':
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal lenght')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear Kernel')

plt.show()