import numpy as np 

N = 1000
s= 3
d_par = 800
d_test = 200
k2 = 10
d_train = 800* (k2-1) / k2 
d_val =d_par -d_train

print(k2*s*(d_train*np.log2(d_train) + d_val) + d_par*np.log2(d_par) + d_test )

