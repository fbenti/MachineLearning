''' LOGISTIC REGRESSION - FIND WEIGHTS '''
import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Given a logistic regression, and one variable x8, what are the weights?
# -> pick up a point and find the weight which gives the closest value
# pick up value x8 = 0

x_tilda = np.array([1, 17])
w1 = np.array([423.49, 48.16])
w2 = np.array([0, -46.21])
w3 = np.array([0, -27.89])
w4 = np.array([418.94, -26.12])
w = [w1, w2, w3, w4]

for i in range(0,4):
    # print(np.exp(-x_tilda@w[i]))
    print('The value for w%d' %(i+1) , 'is : %f' %(sigmoid(x_tilda @ w[i])))
