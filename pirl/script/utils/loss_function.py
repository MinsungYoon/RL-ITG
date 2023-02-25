import matplotlib.pyplot as plt
import numpy as np



def custom_reward_function(x, w, w2, th):
    if abs(x) < th:
        return -w*((x/th)**4-1)
    else:
        return -w2*np.log(abs(x)/th)

def tanh_reward_function(x,w,c):
    return w*(np.exp(2*x/c)-1)/(np.exp(2*x/c)+1)

x = np.linspace(-0.5,0.5,1001)
w = 10
w2 = 5
th = 0.1

# x = np.linspace(0,3.14,1001)
# w = 5
# w2 = 5
# th = np.pi/6

y = []
# y = -w*((x/th)**4-1)
for x_i in x:
    # y.append(custom_reward_function(x_i, w, w2, th))
    y.append(tanh_reward_function(x_i,5,0.1))

plt.plot(x,y)
plt.show()

