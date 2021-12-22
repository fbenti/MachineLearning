''' PCA ANALYSIS - PROJECTION ONTO THE SUBSBACE'''
import numpy as np
# X = np.array(((3,2,1),
#              (4,1,2),
#              (0,1,2)))
# X_tilda = X - np.mean(X,axis=0)
# print(X)

# U = np.array(((-0.26,0.77,0.58),
#              (-0.54,-0.61,-0.58),
#              (0.80,-0.16,0.58)))
# print("U: \n",U)

# S = np.zeros((3,3))
# S[0][0],S[1][1] = 2.96,1.10
# print("S: \n",S)

V = np.array(((0.04,-0.12, -0.14,0.35,0.92),
              (0.06, 0.13, 0.05,-0.92,0.37),
              (-0.03,-0.98,0.08,-0.16,-0.05),
              (-0.99,0.03,0.06,-0.02, 0.07),
              (-0.07,-0.05, 0.98,-0.11, -0.11)))
print("V: \n",V)




# Reprojection onto the first k-principal components
# k = 5
# # Observation to project
# i = 1
# X_tilda = np.array((0,0,0,-1.4,0,0))
# b = X_tilda.T @ (V[:,:])
# print("\n--- Projection obs-{} into the first {}-pricipal components ---\n{}".format(i,k,b))


# # Concatenate needed column of V for the reprojection
subV = np.concatenate(((V[:,0]).reshape(-1,1),(V[:,1]).reshape(-1,1)),axis=1)
X_tilda = np.array((0,0,-1.4,0,0))
b = X_tilda @ subV
print("\n--- Projection obs into subspace ---\n\t{}".format(b))
