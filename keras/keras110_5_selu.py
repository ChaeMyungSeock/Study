import numpy as np
import matplotlib.pyplot as plt

def selu(x):
    y_list =[]
    a = 1.6732632423543772848170429916717
    b = 1.0507009873554804934193349852946
    for x in x:
        if(x>0):
            y = b*x
        else:
            y = a*(np.exp(x)-a)
        y_list.append(y)
    return y_list


x = np.arange(-5,5,0.1)
y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()