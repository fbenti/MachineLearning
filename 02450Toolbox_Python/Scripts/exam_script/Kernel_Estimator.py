# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:52:58 2021

@author: jacop
"""
import numpy as np
import scipy.stats as stats 

# p = (1/np.sqrt(2*3.14*sigma)*np.exp((x-mu)**2)/(2*sigma**2))

sigma=2
y=[]
y1=[]
N = 3
x = [4.5, -0.5, 1.2]

prob = []

for i in range(len(x)):
    s = np.delete(x, i)
    p = 0
    for j in range(len(s)):
        p += stats.norm.pdf(x[i],s[j],sigma)
    prob.append(p/(N-1))
    
print(prob)

# ***** if old-out: *****
N = int(N/2) 
prob = prob[-1]
e = (-1/N)*np.log(prob)

# ***** this should be for leave-one-out *****
# e = 0
# for pr in prob:
#     e += (-1/N)*np.log(pr)
    
print(e)