import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
def mish(x):
    return x*K.tanh(K.softplus(x))

x = np.arange(-5,5,0.1)
y = mish(x)

plt.plot(x,y)
plt.grid()
plt.show()