import numpy as np
import matplotlib.pyplot as plt

def Leak_Relu(x):
    y_list=[]
    for x in x:
        if(x>0):
            y=x
        else:
            y =  0.2*x
        y_list.append(y)
    return y_list

x = np.arange(-5,5,0.1)
y = Leak_Relu(x)

plt.plot(x,y)
plt.grid()
plt.show()