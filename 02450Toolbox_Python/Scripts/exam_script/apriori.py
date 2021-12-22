import numpy as np
from split_raw_data import split_data 
from apyori import apriori

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T


# This function print the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y
def print_apriori_rules(rules):
    frules = []
    unique = set()
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            # print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
            if x != '':
                unique.add(x)
            else:
                unique.add(y)
    return frules,unique



input_str =""""o1 1 1 0 0 0 1 0 0 0 1
o2 1 0 0 0 0 0 0 0 0 0
o3 1 1 0 0 0 1 0 0 0 1
o4 0 1 1 1 0 0 0 1 1 0
o5 1 1 0 0 0 1 0 0 0 1
o6 0 1 1 1 0 0 1 1 1 0
o7 1 1 1 0 0 1 1 1 1 0
o8 0 1 1 1 0 1 1 0 0 1
o9 0 0 0 0 1 1 1 0 1 1
o10 1 0 0 0 0 1 1 1 1 0"""

X = split_data(input_str)


# Threshold
support = 0.515


#creating labels
labels = {}
for i in range(len(X)):
    labels[i] = 'F'+str(i+1) 



# apyori requires data to be in a transactions format, forunately we just wrote a helper function to do that.
T = mat2transactions(X,labels)
rules = apriori( T, min_support=support)
a, unique = print_apriori_rules(rules)

print("\n--- Length of support = {} ---".format(len(unique)))
print("\n--- Itemset with support greater than {} ---\n".format(support))
for el in unique:
    print("\t{}".format(el))

print("\n")