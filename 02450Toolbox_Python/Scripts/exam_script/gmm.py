''' GMM '''
print("\n")

# Give

import numpy as np

# Weights
w_k = np.array((0.19,0.34,0.48))
# Parameters multivariate normal densities
mu_k = np.array((3.177,3.181,3.184))
sigma = np.array((0.0062,0.0076,0.0075))

# According to the GMM, what's the probability an observation x0 = ..
# is assigned to cluser k

# Normal distribution for each cluster
x0 = 3.19

prob = []
for i in range(len(w_k)):
    fact = 1/np.sqrt(6.28*(sigma[i]**2))
    num = np.exp(- ((x0 - mu_k[i])**2) / (2*(sigma[i]**2))) *    fact
    print("Probabilities {}  :  {}".format(i+1,num))
    prob.append(num)

# Cluster assigned 
k = 2

# Posterior prob 
gamma = prob[k-1]*w_k[k-1]
den = 0
for i in range(len(w_k)):
    den += prob[i] * w_k[i]
print("\n--- Probability of x0 being in kluster k = {} --> {}".format(k,gamma/den)) 


print("\n")