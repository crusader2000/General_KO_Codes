import numpy as np
import torch

n = 3
m = 2
all_codes = {}

def bitfield(n,length):
    
# print(bin(n)[2:])
    repr = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    padding = [0]*(length - len(repr))
    # print(padding+repr)
    return np.array(padding+repr)

if __name__ == "__main__":

  for j in range(1,m+1):
    all_codes["n{}_m{}_r{}".format(n,j,0)] = np.hstack([np.zeros((n**j,1)),\
                                              np.ones((n**j,1))])

    all_nums = list(range(2**(n**j)))
    all_codes["n{}_m{}_r{}".format(n,j,j)] = np.array([bitfield(i,n**j) for i in all_nums]).T
    print("n{}_m{}_r{}".format(n,j,0),np.shape(all_codes["n{}_m{}_r{}".format(n,j,0)]))
    for k in all_codes["n{}_m{}_r{}".format(n,j,0)]:
      print(k)
    print()
    print("n{}_m{}_r{}".format(n,j,j),np.shape(all_codes["n{}_m{}_r{}".format(n,j,j)]))
    for k in all_codes["n{}_m{}_r{}".format(n,j,j)]:
      print(k)
    print()

  torch.save(all_codes,"data/data_{}_{}.pth".format(n,m))
