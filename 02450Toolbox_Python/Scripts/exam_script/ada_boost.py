# import os
# os.system('cls' if os.name == 'nt' else 'clear')
print("\n")
''' ADA BOOSTING'''
import numpy as np
import math

# Initial number of element in the dataset
nInitElem = 572
# Intiali coeff. of probability: equals at the first iteration
w_i = np.array([1/nInitElem]*nInitElem)
# print("\n--- Initial weights ---")
# print(w_i)
# Error rate err: num of misclassified elements
missclassified = 143
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

print(w_corr)

# denominator = sum of weight
den = w_corr * (nInitElem-missclassified) + w_wrong * missclassified
print(den)
print(w_corr/den)
new_corr_weight = np.array([w_corr] * (nInitElem-missclassified))/den
new_wrong_weight = np.array([w_wrong] * missclassified)/den
if (new_wrong_weight[0] < new_wrong_weight[0]):
    print("\n--- SOMETHING WRONG ---")
else:
    print("\n --- Weight of correct guess :")
    print(new_corr_weight[0])
    print("\n --- Weight of wrong guess :")
    print(new_wrong_weight[0])

print("\n")
