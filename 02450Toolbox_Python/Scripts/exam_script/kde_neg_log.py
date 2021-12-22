import numpy as np

"""Suppose we apply a Kernel Density Estimator (KDE)
to the dataset with kernel width sigma (i.e., sigma is the
standard deviation of the Gaussian kernels), and we
wish to find sigma by using leave-one-out (LOO) cross-validation 
using the average (per observation) negativelog-likelihood """

def normal_distr(x,mean,sigma):
    p = np.exp(-np.square(x-mean)/ (2*sigma**2)) * (1/ np.sqrt(2*3.14*sigma**2))
    return p

# Observations
x = np.array((3.918,-6.35,-2.677,-3.003))
N = len(x)
# Take the a point where to estimate the density
# choose index
y = 4

# Choose sigma from the graph
sigma = 2

# The density at the observation y, when KDE is fitte with N -1 obs

# ----------------- # 
# #In case of a single test:
# N_test = 1
# p_y = 0
# idx = y-1
# for i in range(N):
#     if i != idx:
#         p_y += normal_distr(x[idx],x[i],sigma)
# p_y /= (N-1)

# -----------------
# # In the case of leave-one-out
N_test = N
p_y = []
for i in range(N):
    p = 0
    for j in range(N):
        if j != i:
            p += normal_distr(x[i],x[j],sigma)
    p_y.append(round(p/(N-1),4))
print("\n--- Density at each observation : \n{}".format(p_y))

# Estimate gen E(sigma)
# To change accordingly
E = 0
for i in range(N_test):
    E += np.log(p_y[i])
E /= -(N_test)

print("\n--- Estimated gen Error at sigma={} --> E = {} ---".format(sigma,E)) 


print("\n")
