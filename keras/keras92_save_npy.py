from sklearn.datasets import load_iris
import numpy as np

# iris = load_iris()

# print(iris.__class__)

# x_data = iris.data
# y_data = iris.target

# print(x_data.__class__)
# print(y_data.__class__)


# np.save('./data/iris_x.npy',arr=x_data)
# np.save('./data/iris_y.npy',arr=y_data)

x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(x_data_load.__class__)
print(y_data_load.__class__)
print(x_data_load.shape)
print(y_data_load.shape)
