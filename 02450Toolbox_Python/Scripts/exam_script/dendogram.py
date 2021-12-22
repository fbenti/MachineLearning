# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:35:10 2021

@author: kubin
"""


import numpy as np
from split_raw_data import split_data 
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as ssd

#input
#==================================================================
# DON'T CHANGE THE FORMAT - >> 4 """" at the beginning and 3 at the end
input_str =""""o1 0.0 53.8 87.0 67.4 67.5 71.2 65.2 117.9 56.1 90.3 109.8
o2 53.8 0.0 69.9 75.5 62.9 58.0 63.0 135.0 84.1 107.9 131.5
o3 87.0 69.9 0.0 49.7 38.5 19.3 35.5 91.8 76.9 78.7 89.1
o4 67.4 75.5 49.7 0.0 24.2 47.2 47.0 62.3 33.4 37.2 60.0
o5 67.5 62.9 38.5 24.2 0.0 37.7 41.7 79.5 52.4 60.2 78.9
o6 71.2 58.0 19.3 47.2 37.7 0.0 21.5 95.6 68.3 78.4 91.0
o7 65.2 63.0 35.5 47.0 41.7 21.5 0.0 96.0 64.3 75.5 89.4
o8 117.9 135.0 91.8 62.3 79.5 95.6 96.0 0.0 66.9 44.3 24.2
o9 56.1 84.1 76.9 33.4 52.4 68.3 64.3 66.9 0.0 39.2 60.7
o10 90.3 107.9 78.7 37.2 60.2 78.4 75.5 44.3 39.2 0.0 39.4
o11 109.8 131.5 89.1 60.0 78.9 91.0 89.4 24.2 60.7 39.4 0.0"""

X = split_data(input_str)
#X=X[5:,5:]

method='complete'

# ''' option availabe: 
#     'complete' -> max
#     'single' -> min
#     'average' -> average
#     other:
#     ’weighted’
#     ’centroid’
#     ’median’ '''

metric= 'euclidean'
#metric='cityblock'

#The distance metric to use. The distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, 
#‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
# ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
# ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

# 
# #calculation


#=================================================================

#==================================================================
# # for matrix without distance 
# arr = np.array([5.7,6,6.2,6.3,6.4,6.6,6.7,6.9,7,7.4])

# #========================= calculations ===========================

# near=[]
# for i in range(len(arr)):
#     near.append(arr[i]-arr[:])
    
# X=np.absolute(np.asarray(near))
#===================================================================

#creating labels
labels = {}
for i in range(len(X)):
    labels[i] = 'O'+str(i+1)   
    
#coverting to dataframe
X=pd.DataFrame(X)
X.rename(columns = labels, inplace = True)

#generatin distance array
distArray = ssd.squareform(X)

#plot
Z = hierarchy.linkage(distArray, method=method,metric=metric)
plt.figure(figsize=(6,5))
dn = hierarchy.dendrogram(Z,distance_sort='ascending',count_sort='descending',labels=X.columns,color_threshold=3)
plt.show()