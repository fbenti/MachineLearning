import os
os.system('cls' if os.name == 'nt' else 'clear')
''' Density Normal Distribution '''
import numpy as np
mean = 4.5
sigma = 2

x = 1.2

p = np.exp(-np.square(x-mean)/ (2*sigma**2)) * (1/ np.sqrt(2*3.14*sigma**2))

print("\n--- Density : {} ---".format(p))