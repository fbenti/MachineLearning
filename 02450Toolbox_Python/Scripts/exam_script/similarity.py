
''' SIMILARITY MEASURES '''
import numpy as np
# Vector to check 


x = np.array((0, 0, 1, 0, 0, 1, 0, 1))
y = np.array((0, 0, 0, 1, 0, 0, 0, 1))

def f_11(x,y,i):
    if x[i] == 1 and y[i] == 1:
        return 1
    return 0

def f_00(x,y,i):
    if x[i] == 0 and y[i] == 0:
        return 1
    return 0

def SMC(x,y):
    sim = 0
    for i in range(len(x)):
        sim += f_11(x,y,i) + f_00(x,y,i)
    return sim/len(x)

def jaccard(x,y):
    pos = 0
    neg = 0
    for i in range(len(x)):
        pos += f_11(x,y,i)
        neg += f_00(x,y,i)
    return pos / (len(x) - neg)

def cos_sim(x,y):
    return (x.T @ y) / (np.sqrt(x.sum()) * np.sqrt(y.sum()))

def extended_j(x,y):
    return (x.T @ y) / (x.sum() + y.sum() - (x.T @ y))

print("\n--- SMC: {}".format(SMC(x,y)))
print("\n--- Jaccard: {}".format(jaccard(x,y)))
print("\n--- Cos similarity: {}".format(cos_sim(x,y)))
print("\n--- Extended Jaccard: {}".format(extended_j(x,y)))

