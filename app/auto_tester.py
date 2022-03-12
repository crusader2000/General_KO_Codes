#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.utils.data
from IPython import display

from util.conf_util import *
from util.log_util import *
from util.utils import *
from model.models import *

import sys
import pickle
import glob
import os
import logging
import time
from datetime import datetime
from datetime import date
import random
from data.generate_data import *

from math import sqrt

import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt


def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

# data = {}

bers = []
losses = []

gnet_dict = {}
fnet_dict = {}

enc_params = []
dec_params = []

device = None


def encoding(n,r,m,msg_bits):
	global device
	# device = torch.device("cuda")
	if r==0:
		# print("r=0")
		# print(msg_bits.size())
		# print((msg_bits*np.ones((1,np.power(n,m)))).size())
		#print('msg_bits is a tensor:  ',torch.is_tensor(msg_bits))
		# temp=np.ones((1,np.power(n,m)))
		temp_tensor=torch.ones(1,np.power(n,m)).to(device)
		msg_bits = msg_bits.to(device)
		# temp_tensor_gpu=temp_tensor.to(device)
		# print("r==0")
		# print(device)
		# print(msg_bits.is_cuda)
		# print(temp_tensor.is_cuda)
		return msg_bits*temp_tensor
	if r==m:
		# print("r=m")
		# print(msg_bits.size())
		return msg_bits
	
	lefts = []
	lcd = get_dimen(n,r-1,m-1)
	
	for i in range(n-1):
		# print("i*lcd,(i+1)*lcd,msg_bits.size()")
		# print(i*lcd,(i+1)*lcd,msg_bits.size())
		# print("msg_bits[:,i*lcd:(i+1)*lcd].size()",msg_bits[:,i*lcd:(i+1)*lcd].size())
		lefts.append(torch.unsqueeze(encoding(n,r-1,m-1,msg_bits[:,i*lcd:(i+1)*lcd]),2))
		# print("lefts",lefts[-1].size())


	rights = torch.unsqueeze(encoding(n,r,m-1,msg_bits[:,(n-1)*lcd:]),2)

	# print("rights",rights.size())
	code = []
	#lefts=lefts.to(device)
	rights=rights.to(device)
	for i in range(n-1):
		# temp = torch.cat([lefts[i],rights],dim=2)
		# print(np.shape(temp))
		temp_tensor=torch.cat([lefts[i],rights],dim=2).to(device)
		code.append(torch.squeeze(gnet_dict["G_{}_{}".format(r,m)](temp_tensor)))
		# print(code[-1].size())
	code.append(torch.squeeze(rights))
	code = torch.hstack(code)
	# print("code.size()",code.size())
	# print()

	return code

def decoding(n,r,m,code):
  
	if r==0:
		s = torch.sum(torch.squeeze(code),dim=1) > 0
		s = torch.unsqueeze(s.int(),1)
		return 2*s-1
	
	if r==m:
		msg = torch.squeeze(code) > 0
		msg = msg.int().to(device)
		return 2*msg-1

	msg_bits = torch.tensor([]).to(device)
	sub_code_len = np.power(n,m-1)
	sub_codes = []
	sub_codes_est = []
	
	y = code[:,(n-1)*sub_code_len:]
	for i in range(n-1):
		sub_codes.append(code[:,i*sub_code_len:(i+1)*sub_code_len])
		# print(code[:,i*sub_code_len:(i+1)*sub_code_len].size())
		# print(code[:,(n-1)*sub_code_len:].size())
		x = code[:,i*sub_code_len:(i+1)*sub_code_len]
		# print(torch.stack([x,y],dim=2).size())
		# print(torch.cat([x,y],dim=2).size())
		sub_codes_est.append(fnet_dict["F_{}_{}_l".format(r,m)](torch.cat( \
				[x,y],dim=2)))
		# print("msg_bits.is_cuda")
		# print(msg_bits.is_cuda)
		# print(msg_bits.is_cuda)
		msg_bits = torch.hstack([msg_bits,decoding(n,r-1,m-1,sub_codes_est[-1])])
		# print("msg_bits.size()",msg_bits.size())

	
	sub_codes.append(code[:,(n-1)*sub_code_len:])
	sub_codes_est.append(torch.zeros(sub_codes[-1].size()))
	# print(sub_codes[-1].size())
	#print(device)
	#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
	#for tens in sub_codes_est:
	#	print(tens.is_cuda)
	#rint('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
	for i in range(len(sub_codes)):
		sub_codes[i]=sub_codes[i].to(device)
	for i in range(len(sub_codes_est)):
                sub_codes_est[i]=sub_codes_est[i].to(device)

	#print('xxxxxxxxxxxxxxxxxx>xxxxxxxxxx')
	#for tens in sub_codes_est:
	#	print(tens.is_cuda)
	#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
	sub_codes=torch.cat(sub_codes,dim=2)
	sub_codes_est=torch.cat(sub_codes_est,dim=2)
	final_tensor=torch.cat((sub_codes,sub_codes_est),dim=2).to(device)
	
	# print('lol')

	# print(sub_codes_est.size())
	# print(sub_codes.size())
	# print("final_tensor.size()",final_tensor.size())

	last_bits = fnet_dict["F_{}_{}_r".format(r,m)](final_tensor)
	#last_bits = fnet_dict["F_{}_{}_r".format(r,m)](torch.hstack( \
         #               [sub_codes_est[0],sub_codes[0]]))

	# print('lol')
	msg_bits = torch.hstack([msg_bits,decoding(n,r,m-1,last_bits)])
	# print()
	return msg_bits

def get_LLR(received_codewords,snr,rate,channel = 'awgn'):
	# print("snr",snr)
	if channel == 'awgn':
		snr_sigma = snr_db2sigma(snr)
		sigma = sqrt(1/(2*snr_sigma*rate))
		L = (2/(sigma**2))*received_codewords
		
	return L

def initialize(n,r,m,hidden_size,device):
	global enc_params
	global dec_params
	if r==0 or r==m:
		return
	if not gnet_dict.__contains__("G_{}_{}".format(r,m)):
		gnet_dict["G_{}_{}".format(r,m)] = g_Full(2, hidden_size, 1, device)
		fnet_dict["F_{}_{}_l".format(r,m)] = f_Full(2, hidden_size, 1, device)
		fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(2*n, hidden_size, 1, device)
		enc_params += list(gnet_dict["G_{}_{}".format(r,m)].parameters())
		dec_params += list(fnet_dict["F_{}_{}_l".format(r,m)].parameters()) + \
		              list(fnet_dict["F_{}_{}_r".format(r,m)].parameters()) 	
		initialize(n,r-1,m-1,hidden_size,device)
		initialize(n,r,m-1,hidden_size,device)
	return

def test(snr, rate, code_dimension):
	global device
	BER_total = []
	Test_msg_bits = 2*torch.randint(0,2,(test_size, code_dimension)).to(torch.float) -1
	Test_Data_Generator = DataLoader(Test_msg_bits, batch_size=100 , shuffle=False)

	num_test_batches = len(Test_Data_Generator)
	ber = 0
	start_time = time.time()

	with torch.no_grad():
			for msg_bits in Test_Data_Generator:
					msg_bits.to(device)
					codewords = encoding(n,r,m,msg_bits)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*np.sqrt(code_length)
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, snr, rate)
					L = get_LLR(corrupted_codewords, snr,rate)
					decoded_bits = decoding(n,r,m,L)

					decoded_bits=decoded_bits.to('cuda')
					msg_bits=msg_bits.to('cuda')
					ber += errors_ber(msg_bits, decoded_bits.sign()).item()
			
			ber /= num_test_batches
			logger.warning(f"[Testing Block] SNR={snr} : BER={ber:.7f}")
			logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

	return ber

if __name__ == "__main__":
	if len(sys.argv) == 2:
		conf_name = sys.argv[1]
		print("train conf_name:", conf_name)
		conf = get_default_conf(f"./config/{conf_name}.yaml")
	else:
		print("default")
		conf = get_default_conf()

	if torch.cuda.is_available():
		device = torch.device("cuda")
		os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
		print(device,os.environ["CUDA_VISIBLE_DEVICES"])
	else:
		device = torch.device("cpu")
		print(device)

	para = conf["para"]
	seed = para["seed"]
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	test_conf = conf["test"]

	today = date.today().strftime("%b-%d-%Y")

	logger = get_logger(test_conf["logger_name"])
	logger.info("test_conf_name : "+conf_name)
	logger.info("Device : "+str(device))

	data_type = para["data_type"]

	code_dimension = np.int64(para["code_dimension"])
	code_length = np.int64(para["code_length"])

	n = para["n"]
	r = para["r"]
	m = para["m"]

	print("code_dimension",code_dimension)
	print("code_length",code_length)

	rate = 1.0*(code_dimension/code_length)
	hidden_size = para["hidden_size"]

	# data = torch.load(para["data_file"])

	initialize(n,r,m,hidden_size,device)

	test_size = para["test_size"]
	test_model_path_encoder = test_conf["test_model_path_encoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])
	test_model_path_decoder = test_conf["test_model_path_decoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])
	
	test_save_dirpath = para["train_save_path_dir"].format(test_conf["day"], para["data_type"])
	if not os.path.exists(test_save_dirpath):
			os.makedirs(test_save_dirpath)

	saved_model_enc = torch.load(test_model_path_encoder)
	saved_model_dec = torch.load(test_model_path_decoder)

	for key,val in saved_model_enc.items():
		gnet_dict[key].load_state_dict(val)

	for key,val in saved_model_dec.items():
		fnet_dict[key].load_state_dict(val)

	bers = []
	snrs = []
	logger.info("Testing {} trained till epoch_num {}".format(conf_name,test_conf["epoch_num"]))
	logger.info("Model trained on {}".format(test_conf["day"]))
	logger.info("Less go!")
	for snr in test_conf["snr_list"].split(","):
			
			ber = test(int(snr), rate, code_dimension)
			bers.append(ber)
			snrs.append(int(snr))
	
	plt.plot(snrs, bers, label=" ",linewidth=2, color='blue')

	plt.xlabel("SNRs")
	plt.ylabel("BERs (Testing)")
	plt.title("Testing BERs")
	plt.savefig(test_save_dirpath+ "/ber_testing_epoch_"+str(test_conf["epoch_num"])+".png")
	plt.close()
