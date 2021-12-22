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
alpha = 0.05

E0 = (n12 -n21)/(n12+n21)
Q = (n**2 * (n+1) * (E0+1) * (1-E0)) / (n*(n12-n21) - (n12-n21)**2)
f = (E0 + 1)*(Q-1)/2
g = (1-E0)*(Q-1)/2

a = f
b = g

print("a = {:.4}".format(a))
print("b = {:.4}".format(b))

# plug in a,b into the inverse cdf function
print("CI = [{:.2} ; {:.2}]".format((2*beta.ppf(alpha/2,a,b) -1 ), (2*beta.ppf(1-alpha/2,a,b)) -1))

print("\n")