import numpy as np
import torch

n = 6
m = 6
all_codes = {}

def bitfield(n,length):
    repr = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    padding = [0]*(length - len(repr))
    return padding+repr

for i in range(1,n+1):
  for j in range(1,m+1):
    all_codes["n{}_m{}_r{}".format(i,j,0)] = np.array([np.zeros((1,n**m)),\
                                              np.ones((1,n**m))])
    all_nums = np.array(list(range(2**(n**m))))
    # .reshape((2**(n**m)),1)
    all_codes["n{}_m{}_r{}".format(i,j,j)] = np.array([bitfield(n,n**m) for n in all_nums])
    print("n{}_m{}_r{}".format(i,j,j),np.shape(all_codes["n{}_m{}_r{}".format(i,j,j)]))

torch.save(all_codes,"data_{}_{}.pth".format(n,m))