''' CLUSTER OVERLAPP - COMPARING PARTITIONS '''
import numpy as np

# Number of observations
N = 10
# We have two partitions Z and Q of 3 and 4 clusters respectively
# m = 3
k_z = [1,2,3]
# n = 4
k_q = [1,2,3]
Z = np.array((1,1,2,2,2,3,3,3,3,3))
Q = np.array((1,1,3,1,1,1,1,3,3,2))
print("\nZ: {}".format(Z))
print("Q: {}".format(Q))


# func: delta(h,k): 1 if observation h belongs to cluster k
def delta(vec,h,k):
    if vec[h] == k:
        return 1
    else:
        return 0
sum = 0

# Define joint count matrix n 
# n[1,2]: how many points belong to cluster 1 in Z and to cluster 2 in Q
# rows: correspond to Z cluster
# cols: correspond to Q cluster
n = np.zeros((len(k_z),len(k_q)))
for k in k_z:   
    for m in k_q:
        sum = 0
        for i in range(N):
            sum += delta(Z,i,k)*delta(Q,i,m)
        n[k-1][m-1] = sum
# print(n)
n_z = np.sum(n,axis = 1).T
n_q = np.sum(n,axis = 0)

# S: number of pairs i,j in the same cluster in Z,Q
S = 0
for k in range(len(k_z)):
    for m in range(len(k_q)):
        S += (n[k][m]*(n[k][m]-1))/2
print("\n--- S = {} ---".format(S))
# D: Numbers of pair i,j in different cluster Z,Q
D = N*(N-1)/2 + S
sum = 0
for k in range(len(k_z)):
    sum += n_z[k]*(n_z[k]-1)/2
D -= sum
sum = 0
for m in range(len(k_q)):
    sum += n_q[m]*(n_q[m]-1)/2
D -= sum
print("\n--- D = {} ---".format(D))
R_index = 2 * (S+D) / (N*(N-1))
Jaccard = S / (0.5*(N*(N-1)) - D)
print("\n--- Rand Index = {:.4} ---".format(R_index))
print("\n--- Jaccard = {:.4} ---".format(Jaccard))