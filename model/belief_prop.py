import torch
import torch.nn as nn
import numpy as np


class BeliefProp(nn.Module):
    def __init__(self, device, matrix):
        super(BeliefProp, self).__init__()
        self.clip_tanh = 10
        self.num_edges = 0
        self.matrix = matrix
        self.k,self.n = len(matrix),len(matrix[0])
        self.device = device
        ############################
        # Matrix : 
        #   v1 v2 ..... vn
        # c1
        # c2
        # c3
        # .
        # .
        # .
        # c(n-k)
        #

        self.edges = {}
        self.adjancency_list = {}

        for i in range(self.k):
            for j in range(self.n):
                if self.matrix[i,j] == 1:

                    if "v"+str(j) not in self.adjancency_list:
                        self.adjancency_list["v"+str(j)] = [("c"+str(i),self.num_edges)]
                    else:
                        self.adjancency_list["v"+str(j)].append(("c"+str(i),self.num_edges))
                    
                    if "c"+str(i) not in self.adjancency_list:
                        self.adjancency_list["c"+str(i)] = [("v"+str(j),self.num_edges)]
                    else:
                        self.adjancency_list["c"+str(i)].append(("v"+str(j),self.num_edges))
                    
                    self.edges[self.num_edges] = ("v"+str(j),"c"+str(i))
                    self.num_edges = self.num_edges + 1

        self.input_layer_mask = torch.zeros(self.n, self.num_edges).to(device)
        self.output_layer_mask = torch.zeros(self.num_edges, self.n).to(device)
        self.odd_to_even_layer_mask = torch.zeros(self.num_edges, self.num_edges).to(device)
        self.even_to_odd_layer_mask = torch.zeros(self.num_edges, self.num_edges).to(device)

        for i in range(self.num_edges):
            self.input_layer_mask[int(self.edges[i][0][1]),i] = 1

        for i in range(self.n):
            for _,e_num in self.adjancency_list["v"+str(j)]:
                self.output_layer_mask[e_num,i] = 1
    
        for i in range(self.num_edges):
            for _,e_num in self.adjancency_list[self.edges[i][1]]: 
                self.odd_to_even_layer_mask[e_num,i] = 1
            for _,e_num in self.adjancency_list[self.edges[i][0]]: 
                self.even_to_odd_layer_mask[i,e_num] = 1

    def odd_layer(self, inputs_v, inputs_e):
        
        # inputs_v = inputs_v.to(torch.float)
        # inputs_e = inputs_e.to(torch.float)
        e_out = torch.matmul(inputs_e, self.odd_to_even_layer_mask).to(torch.float)

        odd = inputs_v + e_out
        odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
        odd = torch.tanh(odd)
        return odd

    def even_layer(self, odd, flag_clip):
        
        num_m,_ = odd.size()
        out = []
        for i in range(num_m):
            temp = torch.reshape(odd[i].repeat(1,self.num_edges),(self.num_edges,self.num_edges))
            # print(temp.size())
            # print(temp[:15,:15])
            temp = torch.mul(temp,self.even_to_odd_layer_mask)
            # print(temp.size())
            # print(temp[:15,:15])
           
            temp = torch.add(temp,1-self.even_to_odd_layer_mask)
            # print(temp.size())
            # print(temp[:15,:15])
           
            temp = torch.prod(temp,dim = 1,keepdim=False).to(torch.float)
            # print(temp.size())
            # print(temp[:15])
           
            out.append(torch.unsqueeze(temp,dim=0))

        out = torch.cat(out,dim=0)
        # print(out.size())

        if flag_clip:
            out = torch.clamp(out, min=-self.clip_tanh, max=self.clip_tanh)
        
        # print(out.size())
        

        out = torch.log(torch.div(1 + out, 1 - out))
        
        return out

    def output_layer(self, inputs_e):
        inputs_e = inputs_e.to(torch.float)
        e_out = torch.matmul(inputs_e, self.output_layer_mask).to(torch.float)
        
        return e_out

    def forward(self, x):
        x = x.to(torch.float)
        flag_clip = 0
        lv = torch.matmul(x,self.input_layer_mask).to(torch.float)
        odd_result = self.odd_layer(lv, lv)
        even_result = self.even_layer(odd_result, flag_clip)

        # flag_clip = 0
        # odd_result = self.odd_layer(lv, even_result)
        # even_result = self.even_layer(odd_result, flag_clip)

        # odd_result = self.odd_layer(lv, even_result)
        # even_result = self.even_layer(odd_result, flag_clip)

        output = self.output_layer(odd_result)
        # print(output[:5,:])
        return output
