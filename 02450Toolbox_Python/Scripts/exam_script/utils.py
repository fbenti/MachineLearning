#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
""" MEAN,MEDIAN,MODE """
import numpy as np
from scipy import stats
# Given an array, what is the sum of the mean,meadian and mode
x = np.array([0,1,1,1,2,3,4,4,5,14])

print("Mean = {}".format(np.mean(x,axis=0)))
print("Median = {}".format(np.median(x,axis=0)))
print("Mode = {}".format(stats.mode(x)[0][0]))

print()

#%%
""" PCA ANALYSIS - VARIANCE EXPLAINED"""
import numpy as np

# After PCA we obtain the following singular values
sigma1 = 43.67
sigma2 = 33.47
sigma3 = 31.15
sigma4 = 30.36
sigma5 = 27.77
sigma6 = 13.86
# sigma7 = 0.7
sigma = np.square(np.array((sigma1,sigma2,sigma3,sigma4,sigma5,sigma6)))

# First k principal components to consider
k = 5
def cum_var_explained(sigma,k):
    tot = sigma.sum()
    var_exp = sigma[:k].sum() / tot
    return var_exp

idx = 4 # BE CAREFULL WITH THE INDEX
def var_explained(sigma,idx):
    tot = sigma.sum()
    var_exp = sigma[idx -1] / tot
    return var_exp

print("\n--- Cumulative variance explained first {} principal components: {} ---".format(k,cum_var_explained(sigma,k)))
print("\n--- Variance explained by the {}-principal components: {} ---".format(idx,var_explained(sigma,idx)))

# %% 

""" PCA ANALYSIS - PROJECTION ONTO THE SUBSBACE """
import numpy as np
# X = np.array(((3,2,1),
#              (4,1,2),
#              (0,1,2)))
# X_tilda = X - np.mean(X,axis=0)
# print(X)

# U = np.array(((-0.26,0.77,0.58),
#              (-0.54,-0.61,-0.58),
#              (0.80,-0.16,0.58)))
# print("U: \n",U)

S = np.zeros((5,5))
S[0][0],S[1][1],S[2][2],S[3][3],S[4][4] = 126.15, 104.44, 92.19, 75.07, 53.48
print("S: \n",S)

V = np.array(((0.11,-0.8,0.3,-0.17,-0.48),
             (-0.58,-0.31,0.01,-0.5,0.56),
             (0.49,0.08,-0.49,-0.72,-0.07),
             (0.6,-0.36,0.04,0.27,0.66),
             (-0.23,-0.36,-0.82,0.37,-0.09)))
print("V: \n",V)
# Reprojection onto the first k-principal components
k = 2
# Observation to project
i = 1
X = np.array((15.5,59.2,1.4,1438,5.3))
mean = np.array((12.9,58.2,1.7,1436.8,4.1))
X_tilda = X-mean
b = X_tilda.T @ V
print(b)


# b = X_tilda[i-1,:].T @ (V[:,:k])
# print("\n--- Projection obs-{} into the first {}-pricipal components ---\n{}".format(i,k,b))

'''------'''

## Concatenate needed column of V for the reprojection
# subV = np.concatenate(((V[:,0]).reshape(-1,1),(V[:,3]).reshape(-1,1)),axis=1)
# X_tilda = np.array((-1,-1,-1,1))
# b = X_tilda @ subV
# print("\n--- Projection obs into subspace ---\n\t{}".format(b))

#%%

''' SIMILARITY MEASURES '''
import numpy as np
# Vector to check 


x = np.array((0, 0, 0, 1, 0, 0, 0, 0, 0))
y = np.array((0, 1, 1, 1, 1, 1, 0, 0, 0))

def f_11(x,y,i):
    if x[i] == 1 and y[i] == 1:
        return 1
    return 0

def f_00(x,y,i):
    if x[i] == 0 and y[i] == 0:
        return 1
    return 0

def SMC(x,y):
    sim = 0
    for i in range(len(x)):
        sim += f_11(x,y,i) + f_00(x,y,i)
    return sim/len(x)

def jaccard(x,y):
    pos = 0
    neg = 0
    for i in range(len(x)):
        pos += f_11(x,y,i)
        neg += f_00(x,y,i)
    return pos / (len(x) - neg)

def cos_sim(x,y):
    return (x.T @ y) / (np.sqrt(x.sum()) * np.sqrt(y.sum()))

def extended_j(x,y):
    return (x.T @ y) / (x.sum() + y.sum() - (x.T @ y))

print("\n--- SMC: {}".format(SMC(x,y)))
print("\n--- Jaccard: {}".format(jaccard(x,y)))
print("\n--- Cos similarity: {}".format(cos_sim(x,y)))
print("\n--- Extended Jaccard: {}".format(extended_j(x,y)))


#%%
'''BAYES THEOREM'''

import numpy as np
# Calculate the p(A|BC) where B or C is usually 1
def bayes(p_B_AC, p_A_C, p_B_nAC, p_nA_C):
    
    p_A_BC = p_B_AC * p_A_C / (p_B_AC * p_A_C + p_B_nAC * p_nA_C)
    return p_A_BC

p_B_AC = 0.9
p_A_C = 0.8
p_B_nAC = 0.4
p_nA_C = 0.2

p_A_BC = bayes(p_B_AC,p_A_C,p_B_nAC,p_nA_C)
print("\nThe probability p(A|BC) = ", np.round(p_A_BC,3)*100, "%")

#%%

''' CONFUSION MATRIX '''
# Calculate the precision(p), recall(r), true negative rate(TNR) and
# false negative rate(FNR)
# _______
# |a | b|
# |c | d|

def confusion_value (a,b,c,d):
    acc = (a+d) / (a+b+c+d)
    p = a / (a+c)
    r = a / (a+b)
    TNR = d / (d+c)
    FNR = a / (d+c)
    return acc,p,r,TNR,FNR

a,b,c,d = 34,11,7,39
# a,b,c,d = 12,69,10,215,

acc,p,r,TNR,FNR = confusion_value(a,b,c,d)
print("\nAccuracy : {:.3}".format(acc))
print("\nPrecision p = {:.3}".format(p))
print("\nError rate = {:.3}".format((b+c)/sum((a,b,c,d))))
print("\nRecall r = {:.3} ".format(r))
print("\nF-measure F = {} ".format(2*p*r/(p+r)))
print("\nTrue Negative Rate TNR = {:.3}".format(TNR))
print("\nFalse Negative Rate = {:.3} ".format(FNR))


#%%


''' LINEAR MODEL '''
import numpy as np

# We observe three points (x,y) = (1,2),(3,5),(4,6)
# We wish to obtain a linear model y = ax + b -> so we construct X as
X_tilda = np.array([[1,1,1],[1,3,4]]).T
y = np.array([2,5,6]).T

# Find parameter weight (@ matrix product)
w = np.linalg.inv(X_tilda.T @ X_tilda) @ X_tilda.T @ y
print("Parameter weight w = [b a]")
print("w = ",w)

# Compute the prediction of the model at x = 5 -> y = x_tilda * w
# Construct a new x_tilda for the prediction
X_tilda = np.array([1,5])

y = X_tilda @ w

print("\nThe prediction is:\ny = ", y)


# %%

""""CALCULATE SIGMOID FUNCTION"""

import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# print(sigmoid(-0.51))

print(sigmoid(1.41 + 0.76*(-0.06) + 1.76 * (-0.28) - 0.43 * 0.43 + 0.3*0.96 + 6.64*(-0.036)
              - 2.74))





# %% 
''' LOGISTIC REGRESSION - FIND WEIGHTS '''
import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Given a logistic regression, and one variable x8, what are the weights?
# -> pick up a point and find the weight which gives the closest value
# pick up value x8 = 0
x_tilda = np.array([1, 1])
w1 = np.array([423.49,48.16])
w2 = np.array([0,-46.21])
w3 = np.array([0,-27.89])
w4 = np.array([418.94,-26.12])
w = [w1, w2, w3, w4]

for i in range(0,4):
    print('The value for w%d' %(i+1) , 'is : %f' %(sigmoid(x_tilda @ w[i])))

# %%


from scipy.stats import beta

""" CONFIDENCE INTERVAL FOR A SINGLE CLASSIFIER """

# Given N observation of which m are true, find the lower and upper bound
# of the confidence interval CI

n = 14
m = 8

alpha = 0.05
a = m + 0.5
b = n - m + 0.5

print("a = {:.3}".format(a))
print("b = {:.3}".format(b))

# plug in a,b into the inverse cdf function
print("CI = [{:.2} ; {:.2}]".format(beta.ppf(alpha/2,a,b), beta.ppf(1-alpha/2,a,b)))



# %% 

""" McNemar Test"""

# Given two classifier, and n12 and n21, find the P-VALUE

from scipy.stats import binom
from scipy.stats import beta

#
n12 = 124
n21 = 100
m = min(n12,n21)
print("p-value = {:.2}".format(2*binom.cdf(m,n12+n21,0.5)))


# Given two classifier calculate the CONFIDENCE INTERVAL

n = 981

E0 = (n12 -n21)/(n12+n21)
Q = (n**2 * (n+1) * (E0+1) * (1-E0)) / (n*(n12-n21) - (n12-n21)**2)
f = (E0 + 1)*(Q-1)/2
g = (1-E0)*(Q-1)/2

a = f
b = g

print("a = {:.3}".format(a))
print("b = {:.3}".format(b))

# plug in a,b into the inverse cdf function
print("CI = [{:.2} ; {:.2}]".format(2*beta.ppf(alpha/2,a,b) -1 , 2*beta.ppf(1-alpha/2,a,b)) -1)

# %% 
""" Jeffrey Test"""

# Given two classifier, and n12 and n21, find the P-VALUE

from scipy.stats import binom
from scipy.stats import beta

# m : numer of times classifier is correct
m = 638

# Calculate the JEFFREY alpha = 0.05  CONFIDENCE INTERVAL

n = 981
alpha = 0.05

a = m + 0.5
b = n - m + 0.5

print("a = {:.4}".format(a))
print("b = {:.4}".format(b))

## !!! BE CAREFULL WITH m > 0 or m < n 

# plug in a,b into the inverse cdf function

if m > 0 and m < n:
    print("CI = [{:.2} ; {:.2}]".format((2*beta.ppf(alpha/2,a,b) -1 ), (2*beta.ppf(1-alpha/2,a,b)) -1))
elif m <=0:
    print("\n--- m is <= than 0 ---")
    print("\nTheta_low =  {}".format(0))
    print("\nTheta_upper = {:.4}".format((2*beta.ppf(1-alpha/2,a,b)) -1))
elif m >= n:
    print("\n--- m in >= than n ---")
    print("\nTheta_low = {:.4}".format((2*beta.ppf(alpha/2,a,b) -1 )))
    print("\nTheta_upper: {}".format(1))

# %%

''' ANN - predicted score given an observation '''

import numpy as np

a = np.array((1,6.8,255,0.44,0.68))
b = np.array((21.78,-1.65,0,-13.26,-8.46))
print(a@b)
c = np.array((1,6.8,255,0.44,0.68))
d = np.array((-9.6,-0.44,0.01,14.54,9.5))
print(c@d.T) 

# %%

''' ADA BOOSTING'''
import numpy as np
import math

# Initial number of element in the dataset
nInitElem = 4
# Intiali coeff. of probability: equals at the first iteration
w_i = np.array([1/nInitElem]*nInitElem)

# Error rate err: num of misclassified elements
missclassified = 1
eps_t = w_i[0] * missclassified
# print(eps_t)
# Penalization coeff alpha
alpha_t = 0.5 * np.log((1-eps_t)/eps_t)
# print(alpha_t)

# Update prob coeff
w_corr = w_i[0] * math.exp(-alpha_t)
# print(w_corr)
w_wrong = w_i[0] * math.exp(alpha_t)
# print(w_wrong)

# denominator = sum of weight
den = w_corr * (nInitElem-missclassified) + w_wrong * missclassified
# print(den)
new_corr_weight = np.array([w_corr] * (nInitElem-missclassified))/den
new_wrong_weight = np.array([w_wrong] * missclassified)/den
if (new_wrong_weight[0] < new_wrong_weight):
    print("\n--- SOMETHING WRONG ---")
else:
    print("\n --- Weight of correct guess : \n")
    print(new_corr_weight)
    print("\n --- Weight of wrong guess : \n")
    print(new_wrong_weight)


# %% 

''' CLUSTER OVERLAPP - COMPARING PARTITIONS '''
import numpy as np

# Number of observations
N = 10
# We have two partitions Z and Q of 3 and 4 clusters respectively
# m = 3
k_z = [1,2,3]
# n = 4
k_q = [1,2,3]
Z = np.array((1,1,3,1,1,1,1,3,3,2))
Q = np.array((1,1,2,2,2,3,3,3,3,3))
print("Z: {}".format(Z))
print("Q: {}".format(Q))


# func: delta(h,k): 1 if observation h belongs to cluster k
def delta(vec,h,k):
    if vec[h] == k:
        return 1
    else:
        return 0
sum = 0

# Define joint count matrix n 
# n[1,2]: how many points belong to cluster 1 in Z and to cluster 2 in Q
# rows: correspond to Z cluster
# cols: correspond to Q cluster
n = np.zeros((len(k_z),len(k_q)))
for k in k_z:   
    for m in k_q:
        sum = 0
        for i in range(N):
            sum += delta(Z,i,k)*delta(Q,i,m)
        n[k-1][m-1] = sum
# print(n)
n_z = np.sum(n,axis = 1).T
n_q = np.sum(n,axis = 0)

# S: number of pairs i,j in the same cluster in Z,Q
S = 0
for k in range(len(k_z)):
    for m in range(len(k_q)):
        S += (n[k][m]*(n[k][m]-1))/2
print("\n--- S = {} ---".format(S))
# D: Numbers of pair i,j in different cluster Z,Q
D = N*(N-1)/2 + S
sum = 0
for k in range(len(k_z)):
    sum += n_z[k]*(n_z[k]-1)/2
D -= sum
sum = 0
for m in range(len(k_q)):
    sum += n_q[m]*(n_q[m]-1)/2
D -= sum
print("\n--- D = {} ---".format(D))
R_index = 2 * (S+D) / (N*(N-1))
Jaccard = S / (0.5*(N*(N-1)) - D)
print("\n--- Rand Index = {} ---".format(R_index))
print("\n--- Jaccard = {} ---".format(Jaccard))

#%%
''' GAUSSIAN KERNEL DENSITY ESTIMATION '''
import numpy as np
import math

# Number of attribute (x_n or centroid)
N = 8
# Value (x-x_n) or (x-mu_k) :::: coeff exponetial
vec = [5.11,4.79,4.9,4.74,2.96,5.16,2.88]
vec1 = [0,4,7,9,5,5,5,6]
x = 0 # value at which calculate the density
sigma = 1
# Density function definition ->> to change every accordingly
def p(x,vec,N,sigma):
    # DON'T USE N/2 but USE N*2 !!!!!!!!!!!!!
    fact = 1 / (N* (2*math.pi)**(N/2) * sigma**2)
    p = 0
    for i in range(N):
        p += math.exp(-0.5*(x-vec[i])**2)
    p *= fact
    return p

def p1(x,vec,N):
    # DON'T USE N/2 but USE N*2 !!!!!!!!!!!!!
    fact = 1 / 64
    p = 0
    for i in range(N):
        p += math.exp(-vec[i]/4)
    p *= fact
    return p

# Density value
# density = p(x,vec,N,sigma)
density = p1(x,vec1,N)

print("\n--- Density value at x = {} ---".format(density))
# %%

