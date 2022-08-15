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

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(sharedstuff.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def perform_ML(n,r,m,code):
	code = (code > 0).to(torch.int64)
	k = get_dimen(n,r,m)
	num_messages = code.size()[0]
	final_msg_bits = []

	for idx in range(num_messages):
		max = 10000000
		result = None
		
		for i in range(2**k):
			curr_codeword = sharedstuff.codebook["{}_{}".format(r,m)][i]
			diff = torch.sum(torch.abs(code[idx,:,:].T-curr_codeword))

			if diff < max:
				result = sharedstuff.codebook_msg_bits["{}_{}".format(r,m)][i]
				max = diff

		final_msg_bits.append(result)

	final_msg_bits = torch.cat(final_msg_bits,dim=0)
	return final_msg_bits				

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

# def decode_burman(n,r,m,code):
# 	if r==1:
# 		return perform_ML(n,r,m,code)
	
# 	if r==0:
# 		return (torch.sum(code,dim=1) > 0).to(torch.int64)
	
# 	if r==m:
# 		return (torch.squeeze(code,2) > 0).to(torch.int64)
	
# 	msg_bits = torch.tensor([]).to(sharedstuff.device)
# 	sub_code_len = np.power(n,m-1)
	
# 	y_n_1 = code[:,(n-1)*sub_code_len:,:]
# 	for i in range(n-1):
# 		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]

# 		temp = torch.cat([y_n_1,y_i],dim=2)
# 		y_i_tilda = torch.unsqueeze(log_sum_exp(temp.permute(0, 2, 1)),2)
		
# 		m_i = decode_burman(n,r-1,m-1,y_i_tilda)
# 		msg_bits = torch.hstack([msg_bits,m_i])

# 		c_i = torch.unsqueeze(encode_burman(n,r-1,m-1,m_i),dim=2)
# 		# print(c_i.size())
# 		# print(y_i.size())
# 		# print(y_n_1.size())
# 		y_i_dash = y_n_1+torch.pow(-1,c_i)*y_i

# 	m_i = decode_burman(n,r,m-1,y_i_dash)
	
# 	msg_bits = torch.hstack([msg_bits,m_i]) 
# 	return msg_bits

### USING MEDIAN F2
# def decode_burman(n,r,m,code):
# 	if r==0:
# 		return (torch.sum(code,dim=1) > (pow(n,m)/2)).to(torch.int64)
	
# 	if r==m:
# 		return torch.squeeze(code,2)
# 	msg_bits = torch.tensor([]).to(sharedstuff.device)
# 	sub_code_len = np.power(n,m-1)
	
# 	y_n_1 = code[:,(n-1)*sub_code_len:,:]
# 	y_i_dash = []
# 	for i in range(n-1):
# 		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
# 		y_i_tilda = (y_n_1 + y_i)%2
# 		m_i = decode_burman(n,r-1,m-1,y_i_tilda)
# 		msg_bits = torch.hstack([msg_bits,m_i])

# 		c_i = torch.unsqueeze(encode_burman(n,r-1,m-1,m_i),dim=2)
		
# 		y_i_dash.append((y_i+c_i)%2)

# 	y_i_dash.append(y_n_1)

# 	num_messages = code.size()[0]
# 	final_msg_bits = []

# 	for idx in range(num_messages):
# 		m_i = None 
# 		cs = []
# 		for i in range(n):
# 			cs.append(torch.unsqueeze(y_i_dash[i][idx,:],dim=0))
# 		cs = torch.cat(cs,dim=0)
# 		m_i = decode_burman(n,r,m-1,torch.unsqueeze(torch.median(cs,dim=0)[0],dim=0))
# 		final_msg_bits.append(m_i)

	
# 	final_msg_bits = torch.cat(final_msg_bits,dim=0)
# 	msg_bits = torch.hstack([msg_bits,final_msg_bits])
		
# 	return msg_bits

### USING MEDIAN R
# def decode_burman(n,r,m,code):
# 	if r==0:
# 		return (torch.sum(code,dim=1) > 0).to(torch.int64)
	
# 	if r==m:
# 		return torch.squeeze(code,2)
# 	msg_bits = torch.tensor([]).to(sharedstuff.device)
# 	sub_code_len = np.power(n,m-1)
	
# 	y_n_1 = code[:,(n-1)*sub_code_len:,:]
# 	y_i_dash = []
# 	for i in range(n-1):
# 		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
# 		y_i_tilda = (y_n_1 + y_i)%2
# 		m_i = decode_burman(n,r-1,m-1,y_i_tilda)
# 		msg_bits = torch.hstack([msg_bits,m_i])

# 		c_i = torch.unsqueeze(encode_burman(n,r-1,m-1,m_i),dim=2)
		
# 		y_i_dash.append(y_n_1+torch.pow(-1,c_i)*y_i)

# 	y_i_dash.append(y_n_1)

# 	num_messages = code.size()[0]
# 	final_msg_bits = []

# 	for idx in range(num_messages):
# 		m_i = None 
# 		cs = []
# 		for i in range(n):
# 			cs.append(torch.unsqueeze(y_i_dash[i][idx,:],dim=0))
# 		cs = torch.cat(cs,dim=0)
# 		# print(torch.quantile(cs,q=0.5,dim=0).size())
# 		m_i = perform_ML(n,r,m-1,torch.unsqueeze(torch.quantile(cs,q=0.5,dim=0),dim=0))
# 		final_msg_bits.append(m_i)

	
# 	final_msg_bits = torch.cat(final_msg_bits,dim=0)
# 	msg_bits = torch.hstack([msg_bits,final_msg_bits])
		
# 	return msg_bits


#### AS GIVEN IN THE PAPER
# def decode_burman(n,r,m,code):
# 	if r==0:
# 		return (torch.sum(code,dim=1) > (pow(n,m)/2)).to(torch.int64)
	
# 	if r==m:
# 		return torch.squeeze(code,2)
	
# 	msg_bits = torch.tensor([]).to(sharedstuff.device)
# 	sub_code_len = np.power(n,m-1)
	
# 	y_n_1 = code[:,(n-1)*sub_code_len:,:]
# 	y_i_dash = []
# 	for i in range(n-1):
# 		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
# 		y_i_tilda = (y_n_1 + y_i)%2
# 		m_i = decode_burman(n,r-1,m-1,y_i_tilda)
# 		msg_bits = torch.hstack([msg_bits,m_i])

# 		c_i = torch.unsqueeze(encode_burman(n,r-1,m-1,m_i),dim=2)
		
# 		y_i_dash.append((y_i+c_i)%2)

# 	y_i_dash.append(y_n_1)

# 	num_messages = code.size()[0]
# 	final_msg_bits = []

# 	for idx in range(num_messages):
# 		m_i = None 
# 		for i in range(n):
# 			m_i = decode_burman(n,r,m-1,torch.unsqueeze(y_i_dash[i][idx,:,:],0))
# 			u = encode_burman(n,r,m-1,m_i)
# 			sum = 0
# 			for j in range(n):
# 				sum += torch.sum(torch.abs(u.T-y_i_dash[j][idx,:]))
# 			if sum < (pow(n,(m-r))/2):
# 				break
# 		final_msg_bits.append(m_i)

	
# 	final_msg_bits = torch.cat(final_msg_bits,dim=0)
# 	msg_bits = torch.hstack([msg_bits,final_msg_bits])
		
# 	return msg_bits

#### AS GIVEN IN THE PAPER - output codewords
def decode_burman(n,r,m,code):
	if r==0:
		return torch.unsqueeze(((torch.sum(code,dim=1) > (pow(n,m)/2)).to(torch.int64)).repeat(1,n**m),dim=2)
	
	if r==m:
		return code
	
	sub_code_len = np.power(n,m-1)
	codewords = torch.tensor([]).to(sharedstuff.device)

	y_n_1 = code[:,(n-1)*sub_code_len:,:]
	y_i_dash = []
	for i in range(n-1):
		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
		y_i_tilda = (y_n_1 + y_i)%2
		c_i = decode_burman(n,r-1,m-1,y_i_tilda)

		y_i_dash.append((y_i+c_i)%2)
		codewords = torch.hstack([codewords,c_i])
	
	y_i_dash.append(y_n_1)

	num_messages = code.size()[0]

	final_codeword_bits = []
	for idx in range(num_messages):
		m_i = None 
		for i in range(n):
			u = decode_burman(n,r,m-1,torch.unsqueeze(y_i_dash[i][idx,:,:],0))
			# print(u.size())
			sum = 0
			for j in range(n):
				# print(torch.unsqueeze(y_i_dash[j][idx,:],dim=0).size())
				sum += torch.sum(torch.abs(u.T-torch.unsqueeze(y_i_dash[j][idx,:],dim=0)))
			if sum < (pow(n,(m-r))/2):
				break
		final_codeword_bits.append(u)

	
	final_codeword_bits = torch.cat(final_codeword_bits,dim=0)
	codewords = torch.hstack([codewords,final_codeword_bits])
		
	return codewords


### ML RULE
# def decode_burman(n,r,m,code):
# 	return perform_ML(n,r,m,code)


### ML RULE 2
# def decode_burman(n,r,m,code):
# 	if r==0:
# 		return (torch.sum(code,dim=1) > (pow(n,m)/2)).to(torch.int64)
	
# 	if r==m:
# 		return torch.squeeze(code,2)
	
# 	msg_bits = torch.tensor([]).to(sharedstuff.device)
# 	sub_code_len = np.power(n,m-1)
	
# 	y_n_1 = code[:,(n-1)*sub_code_len:,:]
# 	y_i_dash = []
# 	for i in range(n-1):
# 		y_i = code[:,i*sub_code_len:(i+1)*sub_code_len,:]
# 		y_i_tilda = (y_n_1 + y_i)%2
# 		m_i = decode_burman(n,r-1,m-1,y_i_tilda)
# 		msg_bits = torch.hstack([msg_bits,m_i])

# 		c_i = torch.unsqueeze(encode_burman(n,r-1,m-1,m_i),dim=2)
		
# 		y_i_dash.append((y_i+c_i)%2)

# 	y_i_dash.append(y_n_1)

# 	num_messages = code.size()[0]
# 	final_msg_bits = []
# 	final_codewords = []

# 	for idx in range(num_messages):
# 		m_i = None 
# 		for i in range(n):
# 			m_i = decode_burman(n,r,m-1,torch.unsqueeze(y_i_dash[i][idx,:,:],0))
# 			u = encode_burman(n,r,m-1,m_i)
# 			# print(u.size())
# 			sum = 0
# 			for j in range(n):
# 				sum += torch.sum(torch.abs(u.T-y_i_dash[j][idx,:]))
# 			if sum < (pow(n,(m-r))/2):
# 				break
# 		final_msg_bits.append(m_i)
# 		final_codewords.append(u)

# 	final_codewords = torch.cat(final_codewords,dim=0)
	
# 	# print(final_codewords.size())	
# 	if num_messages == 1 :
# 		final_msg_bits = perform_ML(n,r,m-1,torch.unsqueeze(final_codewords,2))
# 	else:
# 		final_msg_bits = torch.cat(final_msg_bits,dim=0)
  		
# 	msg_bits = torch.hstack([msg_bits,final_msg_bits])
		
# 	return msg_bits


### Automorphism Decoder
def permute(code,permute_order):
	new_codeword = []
	for idx in permute_order:
		new_codeword.append(code[:,idx,:])
	# 	print(new_codeword[-1].size())
	# print(torch.cat(new_codeword,dim=1).size())
	return torch.unsqueeze(torch.cat(new_codeword,dim=1),2)

def unpermute(permute_code,permute_order):
	orig_codeword = list(range(len(permute_order)))
	for i in range(len(permute_order)):
		orig_codeword[permute_order[i]] = permute_code[:,i,:]
	# print(torch.cat(orig_codeword,dim=1).size())
	return torch.unsqueeze(torch.cat(orig_codeword,dim=1),2)

def decode_burman_automorphism(n,r,m,code):
  	
	decoder_used = decode_burman_beliefprop2

	permutations = []

	for i in range(len(sharedstuff.am_list)):
		permutations.append(decoder_used(n,r,m,permute(code,sharedstuff.am_list[i])))

	unpermute_codewords = []
	for i in range(len(permutations)):
		unpermute_codewords.append(unpermute(permutations[i],sharedstuff.am_list[i]))

	min = 999999999
	final_codeword = None
	for i in range(len(permutations)):
		dist = torch.sum(torch.abs(code-unpermute_codewords[i]))
		if dist < min:
			min = dist
			final_codeword = unpermute_codewords[i]
	
	# final_msg_bits = perform_ML(n,r,m,final_codeword)

	return final_codeword

def decode_burman_beliefprop(n,r,m,code):
	code = torch.squeeze(code,dim=2)
	y_out = sharedstuff.belief_prop.forward(code)
	# y_out = torch.unsqueeze(y_out,dim=2)
	
	return y_out

def decode_burman_beliefprop2(n,r,m,code):
	code = torch.squeeze(code,dim=2)
	# y_out = torch.unsqueeze(y_out,dim=2)
	m,_ = code.size()

	decoded = []

	for i in range(m):
		temp, _, _ = sharedstuff.belief_prop.decode(code[i,:])
		# print(temp.size())
		decoded.append(torch.unsqueeze(temp,dim=0))

	y_out = torch.unsqueeze(torch.cat(decoded,dim=0).to(torch.float).to(sharedstuff.device),dim=2)
	# print(y_out.size())
	return y_out

	