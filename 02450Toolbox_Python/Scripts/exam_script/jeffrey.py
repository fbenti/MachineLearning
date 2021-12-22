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