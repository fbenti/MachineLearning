# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:34:44 2020

@author: Maciek
"""
import numpy as np
################################################################

M = 10000
sentence1 = "the bag of words representation should not give you a hard time"
sentence2 = "remember the representation should be a vector"


##############################################################
#to array
x1 = np.asarray(sentence1.split())
x2 = np.asarray(sentence2.split())


#n11,n01,n11,n00
n11 = 0
for element in x1:
    if(element in x2):
        n11 += 1
n10 = len(x1) - n11
n01 = len(x2) - n11
n00 = M - n10 - n01 - n11

SMC = (n11+n00)/M
J=n11/(n11+n10+n01)
cos = n11/(len(x1)**(1/2)*len(x2)**(1/2))


print('SMC: {0}'.format(SMC))
print('Jacart: {0}'.format(J))
print('cos: {0}'.format(cos))
