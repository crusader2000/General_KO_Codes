import numpy as np
import torch

n = 4
m = 3
all_codes = {}

def bitfield(n,length):
    repr = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    padding = [0]*(length - len(repr))
    return padding+repr

for j in range(1,m+1):
  all_codes["n{}_m{}_r{}".format(n,j,0)] = np.array([np.zeros((1,n**j)),\
                                            np.ones((1,n**j))])
  print(2**(n**m))
  print(list(range(2**(n**j))))
  all_nums = np.array(list(range(2**(n**j)))).reshape((2**(n**m)),1)
  all_codes["n{}_m{}_r{}".format(n,j,j)] = np.array([bitfield(n,n**j) for n in all_nums])
  print("n{}_m{}_r{}".format(n,j,j),np.shape(all_codes["n{}_m{}_r{}".format(n,j,j)]))

torch.save(all_codes,"data/data_{}_{}.pth".format(n,m))