import torch

from util.utils import *
from model.models import *
import app.sharedstuff as sharedstuff

from util.conf_util import *
from util.log_util import *
from util.utils import *
import random

def log_sum_exp(LLR_vector):
	# print(LLR_vector.size())
	sum_vector = LLR_vector.sum(dim=1, keepdim=True)
	sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)
	# print(torch.logsumexp(sum_concat, dim=1).size())
	return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1) 

def encoding(n,r,m,msg_bits):
	if r==0:
		#temp=torch.ones(1,n**m).to(sharedstuff.device)
		return msg_bits.repeat(1,n**m)
	if r==m:
		return msg_bits
	
	lefts = []
	lcd = get_dimen(n,r-1,m-1)
	
	for i in range(n-1):
		lefts.append(torch.unsqueeze(encoding(n,r-1,m-1,msg_bits[:,i*lcd:(i+1)*lcd]),2))


	rights = torch.unsqueeze(encoding(n,r,m-1,msg_bits[:,(n-1)*lcd:]),2)

	code = []
	for i in range(n-1):
		temp_tensor=torch.cat([rights,lefts[i]],dim=2)
		code.append(torch.squeeze(sharedstuff.gnet_dict["G_{}_{}".format(r,m)](temp_tensor)))

	code.append(torch.squeeze(rights))
	code = torch.hstack(code)

	return code

def decoding(n,r,m,code):
	if r==0:
		return torch.tanh(torch.sum(code,dim=1)/2)
	
	if r==m:
		return torch.tanh(torch.squeeze(code)/2)

	msg_bits = torch.tensor([]).to(sharedstuff.device)
	sub_code_len = np.power(n,m-1)
	sub_codes = []
	sub_codes_est = []
	codewords = []
	L_us = []
	L_vs = []

	y_v = code[:,(n-1)*sub_code_len:,:]
	for i in range(n-1):
		y_ui = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
		sub_codes.append(y_ui)
		
		temp = torch.cat([y_v,y_ui],dim=2)
		L_ui = sharedstuff.fnet_dict["F_{}_{}_l".format(r,m)](temp) + log_sum_exp(temp.permute(0, 2, 1)).unsqueeze(2)
		L_us.append(L_ui)

		m_i = decoding(n,r-1,m-1,L_ui)
		msg_bits = torch.hstack([msg_bits,m_i])

		c_i = torch.unsqueeze(encoding(n,r-1,m-1,m_i),dim=2)
		codewords.append(c_i)

		L_vs.append(y_v + c_i*y_ui)

	
	# sub_codes.insert(0,y_v)
	sub_codes.append(y_v)
	#sub_codes_est.append(torch.zeros(sub_codes[-1].size()).to(sharedstuff.device))
	#codewords.append(torch.zeros(sub_codes[-1].size()).to(sharedstuff.device))
	
	# print(sub_codes[0].size())
	# print(sub_codes_est[0].size())
	# print(codewords[0].size())

	
	sub_codes	=	torch.cat(sub_codes,dim=2)
	# sub_codes_est	=	torch.cat(sub_codes_est,dim=2)
	codewords	=	torch.cat(codewords,dim=2)
	L_vs	=	torch.cat(L_vs,dim=2)
	L_us	=	torch.cat(L_us,dim=2)

	# final_tensor	=	torch.cat((sub_codes,sub_codes_est,codewords),dim=2)
	# print("final_tensor.size()")
	# print(final_tensor.size())
	# print(c_i.size())
	if r == 1:
		final_tensor	=	torch.cat((sub_codes,codewords),dim=2)
		if n==2:
			last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](final_tensor) + y_v + c_i*y_ui
		else:
			# last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](final_tensor) + sharedstuff.fnet_dict["F_{}_{}_c".format(r,m)](torch.cat(L_vs,dim=2))
			# randidx = random.randint(0,n-2)
			# temp = torch.cat((y_v,torch.unsqueeze(sub_codes[:,:,randidx],2),torch.unsqueeze(codewords[:,:,randidx],2)),2)
			# last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](temp) + 10*(y_v + torch.unsqueeze(codewords[:,:,randidx]*sub_codes[:,:,randidx],2))


			
			# last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](torch.cat((y_v,y_ui,c_i),2)) 
			last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](final_tensor) 
			for i in range(n-1):
				last_bits +=  10*(y_v + torch.unsqueeze(codewords[:,:,i]*sub_codes[:,:,i],2))

			# last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](final_tensor) + y_v + c_i*y_ui
	else:
		final_tensor	=	torch.cat((sub_codes,L_us,codewords),dim=2)
		last_bits = sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)](final_tensor) 
		for i in range(n-1):
			last_bits +=  10*(y_v + torch.unsqueeze(codewords[:,:,i]*sub_codes[:,:,i],2))

	msg_bits = torch.hstack([msg_bits,decoding(n,r,m-1,last_bits)])
	
	return msg_bits

def initialize(n,r,m,hidden_size):
	if r==0 or r==m:
		return
	if not sharedstuff.gnet_dict.__contains__("G_{}_{}".format(r,m)):
		sharedstuff.gnet_dict["G_{}_{}".format(r,m)] = g_Full(2, hidden_size, 1, sharedstuff.device)
		sharedstuff.fnet_dict["F_{}_{}_l".format(r,m)] = f_Full(2, hidden_size, 1, sharedstuff.device)
		# sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(2*n, hidden_size, 1, sharedstuff.device)
		# sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(n+2*(n-1), hidden_size, 1, sharedstuff.device)
		if r == 1:
			sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(n+n-1, hidden_size, 1, sharedstuff.device)
		else:
			sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(n+2*(n-1), hidden_size, 1, sharedstuff.device)
  			
		# sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(3, hidden_size, 1, sharedstuff.device)
		#sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(3*n, hidden_size, 1, sharedstuff.device)

		# sharedstuff.fnet_dict["F_{}_{}_c".format(r,m)] = f_Half(n-1, 16, 1, sharedstuff.device)
		# sharedstuff.fnet_dict["F_{}_{}_c".format(r,m)] = f_Half(n-1, 5, 1, sharedstuff.device)

		sharedstuff.gnet_dict["G_{}_{}".format(r,m)].apply(weights_init)
		sharedstuff.fnet_dict["F_{}_{}_l".format(r,m)].apply(weights_init)
		sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)].apply(weights_init)
		# sharedstuff.fnet_dict["F_{}_{}_c".format(r,m)].apply(weights_init)

		sharedstuff.enc_params += list(sharedstuff.gnet_dict["G_{}_{}".format(r,m)].parameters())
		sharedstuff.dec_params += list(sharedstuff.fnet_dict["F_{}_{}_l".format(r,m)].parameters()) + \
		              list(sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)].parameters())
		
		# sharedstuff.dec_params += list(sharedstuff.fnet_dict["F_{}_{}_l".format(r,m)].parameters()) + \
		#               list(sharedstuff.fnet_dict["F_{}_{}_r".format(r,m)].parameters()) + \
		#               list(sharedstuff.fnet_dict["F_{}_{}_c".format(r,m)].parameters())

		initialize(n,r-1,m-1,hidden_size)
		initialize(n,r,m-1,hidden_size)
	return
