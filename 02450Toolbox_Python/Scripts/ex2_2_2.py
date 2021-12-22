# exercise 2.2.2

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as linalg
import numpy as np

# Digits to include in analysis (to include all, n = range(10) )
n = [0,1]
# Number of principal components for reconstruction
K = 16
# Digits to visualize
nD = range(6);


# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat('../Data/zipdata.mat')['traindata']
X = traindata[:,1:]
y = traindata[:,0]

N,M = X.shape
C = len(n)

classValues = n
classNames = [str(num) for num in n]
classDict = dict(zip(classNames,classValues))


# Select subset of digits classes to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = (y == v)
    class_mask = class_mask | cmsk
X = X[class_mask,:]
y = y[class_mask]
N=X.shape[0]

# Center the data (subtract mean column values)
Xc = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U, S, Vh = linalg.svd(Xc,full_matrices=False)
#U = mat(U)
V = Vh.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Xc @ V

# Plot variance explained
figure()
plot(rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained value');


# Plot PCA of the data
f = figure()
title('pixel vectors of handwr. digits projected on PCs')
for c in n:
    # select indices belonging to class c:
    class_mask = (y == c)
    plot(Z[class_mask,0], Z[class_mask,1], 'o')
legend(classNames)
xlabel('PC1')
ylabel('PC2')


# Visualize the reconstructed data from the first K principal components
# Select randomly D digits.
figure(figsize=(10,3))
W = Z[:,range(K)] @ V[:,range(K)].T
D = len(nD)
for d in range(D):
    digit_ix = np.random.randint(0,N)
    subplot(2, D, d+1)
    I = np.reshape(X[digit_ix,:], (16,16))
    imshow(I, cmap=cm.gray_r)
    title('Original')
    subplot(2, D, D+d+1)
    I = np.reshape(W[digit_ix,:]+X.mean(0), (16,16))
    imshow(I, cmap=cm.gray_r)
    title('Reconstr.');
    

# Visualize the pricipal components
figure(figsize=(8,6))
for k in range(K):
    N1 = np.ceil(np.sqrt(K)); N2 = np.ceil(K/N1)
    subplot(N2, N1, k+1)
    I = np.reshape(V[:,k], (16,16))
    imshow(I, cmap=cm.hot)
    title('PC{0}'.format(k+1));

# output to screen
show()

print('Ran Exercise 2.2.2')


# Plot variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# # Show that it requires 22 PCA components to account for more than 90%
k = 22
num, den = 0, 0
for j in range(k):
    num = num + S[j]**2
# for j in range(S.shape[0]):
    # den = den + S[j][j]**2

den = (S**2).sum()

weight = num/den
print(weight)
