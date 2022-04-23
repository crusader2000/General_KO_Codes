import torch

from util.utils import *

from util.conf_util import *
from util.log_util import *
from util.utils import *
import random
import app.sharedstuff as sharedstuff

def log_sum_exp(LLR_vector):
	# print(LLR_vector.size())
	sum_vector = LLR_vector.sum(dim=1, keepdim=True)
	sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)
	# print(torch.logsumexp(sum_concat, dim=1).size())
	return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1) 

def encode_burman(n,r,m,msg_bits):
	# print("msg_bits")
	# print(msg_bits[:5,:])
	
	if r==0:
		#temp=torch.ones(1,n**m).to(sharedstuff.device)
		return msg_bits.repeat(1,n**m)
	if r==m:
		return msg_bits
	
	lefts = []
	lcd = get_dimen(n,r-1,m-1)
	
	for i in range(n-1):
		lefts.append(encode_burman(n,r-1,m-1,msg_bits[:,i*lcd:(i+1)*lcd]))

	rights = encode_burman(n,r,m-1,msg_bits[:,(n-1)*lcd:])

	code = []
	for i in range(n-1):
		code.append((rights+lefts[i])%2)

	code.append(rights)
	code = torch.hstack(code)
	# print("code")
	# print(code[:5,:])
	return code

def decode_burman(n,r,m,code):
	# print("code")
	# print(code[:1,:,:])
	if r==0:
		return (torch.sum(code,dim=1) > (pow(n,m)/2)).to(torch.int64)
	
	if r==m:
		return torch.squeeze(code,2)
	# print("code.size()")
	# print(code.size())
	msg_bits = torch.tensor([]).to(sharedstuff.device)
	sub_code_len = np.power(n,m-1)
	
	y_n_1 = code[:,(n-1)*sub_code_len:,:]
	y_i_dash = []
	for i in range(n-1):
		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
		y_i_tilda = (y_n_1 + y_i)%2
		m_i = decode_burman(n,r-1,m-1,y_i_tilda)
		msg_bits = torch.hstack([msg_bits,m_i])

		# print("m_i.size()")
		# print(m_i.size())
		c_i = torch.unsqueeze(encode_burman(n,r-1,m-1,m_i),dim=2)
		# print("c_i.size()")
		# print(c_i.size())
		# print("y_i.size()")
		# print(y_i.size())
		
		y_i_dash.append((y_i+c_i)%2)

	y_i_dash.append(y_n_1)

	num_messages = code.size()[0]
	final_msg_bits = []
	for idx in range(num_messages):
		for i in range(n):
			m_i = decode_burman(n,r,m-1,torch.unsqueeze(y_i_dash[i][idx,:],0))
			u = encode_burman(n,r,m-1,m_i)
			sum = 0
			# print(n,r,m-1)
			# print(u.T.size())
			for j in range(n):
				# print(y_i_dash[j][idx,:].size())
				sum += torch.sum(torch.abs(u.T-y_i_dash[j][idx,:]))
			if sum < (pow(n,(m-r))/2):
				break
		final_msg_bits.append(m_i)
	
	final_msg_bits = torch.cat(final_msg_bits,dim=0)
	# if num_messages > 1:
	# 	print("msg_bits.size()")
	# 	print(msg_bits.size())
	# 	print("final_msg_bits.size()")
	# 	print(final_msg_bits.size())
	msg_bits = torch.hstack([msg_bits,final_msg_bits])
	
	# print("msg_bits")
	# print(msg_bits[:1,:])
	
	return msg_bits
