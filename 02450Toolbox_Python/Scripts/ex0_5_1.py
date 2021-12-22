## exercise 0.5.1
import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(0, 1, 0.1)
# f = np.exp(x)

x = np.arange(0, 2*np.pi, 0.1)
f = np.cos(x)

plt.figure(1)
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)=exp(x)')
plt.title('The exponential function')
plt.show()