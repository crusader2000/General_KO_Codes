import torch
import numpy as np
from math import sqrt


# def FHT(n,R):
#     """
#     Implementation of Recursive Fast Hadamard Transform
#     Returns:
#         P - Hadamard Transform
#     """
#     if n == 2:
#         return torch.cat([R[:,0]+R[:,1],R[:,0]-R[:,1]])
    
#     a = R[:,0::2]
#     b = R[:,1::2]
    
#     P1,c1 = FHT(n//2,a)
#     P2,c2 = FHT(n//2,b)
    
#     return torch.cat((P1+P2,P1-P2))


# # In[39]:


# def decode_hadamard(code_length,rx):
#     """
#     Function to decode a receive codeword using Fast Hadamard Transform
#     """
    
#     P = FHT(code_length,rx)
    
#     ti = np.argmax(abs(P))
    
#     decode_msg = np.zeros(m+1,dtype=int)
    
#     if P[ti] < 0:
#         decode_msg[0] = 1
    
#     pos = 1
    
#     while ti:
#         if ti % 2:
#             decode_msg[pos] = 1
#         ti //= 2
#         pos = pos + 1
        
#     return decode_msg,count

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.)

# Calculating BER
def errors_ber(y_true, y_pred):
	y_true = y_true.view(y_true.shape[0], -1, 1)
	y_pred = y_pred.view(y_pred.shape[0], -1, 1)

	myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
	res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
	return res


# Calculating BLER
def errors_bler(y_true, y_pred):
	y_true = y_true.view(y_true.shape[0], -1, 1)
	y_pred = y_pred.view(y_pred.shape[0], -1, 1)

	decoded_bits = torch.round(y_pred).cpu()
	X_test       = torch.round(y_true).cpu()
	tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
	tp0 = tp0.detach().cpu().numpy()
	bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
	return bler_err_rate

# Channel Type 1
# def snr_db2sigma(train_snr):
#   	return 10**(train_snr*1.0/10)

# def awgn_channel(codewords, snr, rate):
# 	snr_sigma = snr_db2sigma(snr)
# 	standard_Gaussian = torch.randn_like(codewords)
# 	corrupted_codewords = codewords+ sqrt(1/(2*snr_sigma*rate))*standard_Gaussian
# 	return corrupted_codewords

# def get_LLR(received_codewords,snr,rate,channel = 'awgn'):
# 	if channel == 'awgn':
# 		snr_sigma = snr_db2sigma(snr)
# 		sigma = sqrt(1/(2*snr_sigma*rate))
# 		L = (2/(sigma**2))*received_codewords
		
# 	return L

# Channel Type 2
def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def awgn_channel(codewords, snr,rate):
    noise_sigma = snr_db2sigma(snr)
    standard_Gaussian = torch.randn_like(codewords)
    corrupted_codewords = codewords+noise_sigma * standard_Gaussian
    return corrupted_codewords


def get_LLR(received_codewords,snr,rate,channel = 'awgn'):
	if channel == 'awgn':
		sigma = snr_db2sigma(snr)
		L = (2/(sigma**2))*received_codewords
		
	return L