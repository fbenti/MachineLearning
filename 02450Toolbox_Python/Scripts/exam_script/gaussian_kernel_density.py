''' GAUSSIAN KERNEL DENSITY ESTIMATION '''
import numpy as np
import math

# Number of attribute (x_n or centroid)
N = 2
sigma = 20
# Value (x-x_n) or (x-mu_k) :::: coeff exponetial
vec = [24.2,39.4]
x = 0 # value at which calculate the density
# Density function definition ->> to change every accordingly
def p(x,vec,N,sigma):
    # DON'T USE N/2 but USE N*2 !!!!!!!!!!!!!
    # fact = 1 / (N * ((2*math.pi)**(N*0.5)) * (sigma**2))
    fact = 1/(N*((6.28*(sigma**2))**4))
    p = 0
    for i in range(N):
        p += math.exp(-0.5*(np.square(x-vec[i]))/np.square(sigma))
    print(p)
    p *= fact
    return p

# Density value
density = p(x,vec,N,sigma)
print("\n--- Density value at x = {} ----".format(density))

print("\n")