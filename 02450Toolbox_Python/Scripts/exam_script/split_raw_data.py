# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:34:18 2020

@author: Maciek
from split_raw_data import split_data <-how to use that

"""
import numpy as np


input_str ="""o1 0.0 2.91 0.63 1.88 1.02 1.82 1.92 1.58 1.08 1.43
o2 2.91 0.0 3.23 3.9 2.88 3.27 3.48 4.02 3.08 3.47
o3 0.63 3.23 0.0 2.03 1.06 2.15 2.11 1.15 1.09 1.65
o4 1.88 3.9 2.03 0.0 2.52 1.04 2.25 2.42 2.18 2.17
o5 1.02 2.88 1.06 2.52 0.0 2.44 2.38 1.53 1.71 1.94
o6 1.82 3.27 2.15 1.04 2.44 0.0 1.93 2.72 1.98 1.8
o7 1.92 3.48 2.11 2.25 2.38 1.93 0.0 2.53 2.09 1.66
o8 1.58 4.02 1.15 2.42 1.53 2.72 2.53 0.0 1.68 2.06
o9 1.08 3.08 1.09 2.18 1.71 1.98 2.09 1.68 0.0 1.48
o10 1.43 3.47 1.65 2.17 1.94 1.8 1.66 2.06 1.48 0.0"""

def split_data(input_str):
    arr = input_str.split("\n")
    result = []
    for element in arr:
        result.append(element.split()[1:])
    result = np.asarray(result).astype(np.float)
    return result   
