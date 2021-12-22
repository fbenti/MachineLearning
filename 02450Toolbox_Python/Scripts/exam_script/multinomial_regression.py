import numpy as np

""" MULTINOMIAL REGRESSION"""

# We consider a multinomial regression model
# applied to the dataset projected onto the first two principal directions,
# giving the two coordinates b1 and b2 for each observation.

# number of class
k = 3

# choose point in the graph
b1 = -5.52
b2 = -4.69

# Possible weights
w1 = np.array((
    (0.04,1.32,-1.48),
    (0.51,1.65,0.01),
    (-0.9,-4.39,0),
    (-1.22,-9.88,-0.01)))
print("\n Possible w1: \n{}".format(w1))

w2 = np.array((
    (-0.03,0.7,-0.85),
    (0.1,3.8,0.04),
    (-0.09,-2.45,-0.04),
    (-0.28,-2.9,-0.01)))
print("\n Possible w2: \n{}".format(w2))


p = [[] for i in range(len(w1))]
for i in range(len(w1)):
    y_hat1 = np.array((1,b1,b2)).T @ w1[i]
    y_hat2 = np.array((1,b1,b2)).T @ w2[i]
    y_hat = [y_hat1, y_hat2, 0]
    # print("\n---Predicted class for weights-{}: \n{}".format(i+1,y_hat))
    den = 1
    for c in range(k-1):
        den += np.exp(y_hat[c])
    for c in range(k):
        if c < k-1:
            p[i].append(np.exp(y_hat[c])/den)
        else:
            p[i].append(1/den)

for i in range(len(w1)):
    print("\n---Predicted class for weights-{}: \n{}".format(i+1,p[i]))
print("\n")
