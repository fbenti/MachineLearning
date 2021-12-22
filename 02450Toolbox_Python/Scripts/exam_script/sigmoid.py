""""CALCULATE SIGMOID FUNCTION"""

import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# print(sigmoid(-0.51))

print(sigmoid(1.7))