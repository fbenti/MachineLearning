""" PCA ANALYSIS - VARIANCE EXPLAINED"""
import numpy as np

# After PCA we obtain the following singular values
sigma1 = 43.4
sigma2 = 23.29
sigma3 = 18.26
sigma4 = 9.34
sigma5 = 2.14
# sigma6 = 13.816
# sigma7 = 0.7
sigma = np.square(np.array((sigma1,sigma2,sigma3,sigma4,sigma5)))

# First k principal components to consider
k = 3
def cum_var_explained(sigma,k):
    tot = sigma.sum()
    var_exp = sigma[:k].sum() / tot
    return var_exp

idx = 1 # BE CAREFULL WITH THE INDEX
def var_explained(sigma,idx):
    tot = sigma.sum()
    var_exp = sigma[idx -1] / tot
    return var_exp

print("\n--- Cumulative variance explained first {} principal components: {} ---".format(k,cum_var_explained(sigma,k)))
print("\n--- Variance explained by the {}-principal components: {} ---".format(idx,var_explained(sigma,idx)))
