
import torch
import numpy as np


a = np.array([3,0,5,10,1])
print a

index = np.argsort(a)
print index

b=[a[i] for i in index]
print b

c=np.zeros(len(a))
for i, idx in enumerate(index):
    c[idx] = b[i]
print c

k = np.array([[3,0,5,10,1],[1,1,1,1,1]])
print k.shape