import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,2,1000)
# params = [2.0, -65.6, 30.0, 1.0]
params = [1.0, -15, 0.5, 1.0]

y = params[0]*np.exp(params[1]*x) - params[2]*(params[3]*x)**2

plt.plot(x,y)
plt.show()
