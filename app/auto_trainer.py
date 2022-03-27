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


def log_sum_exp(LLR_vector):
	# print(LLR_vector.size())
	sum_vector = LLR_vector.sum(dim=1, keepdim=True)
	sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)
	# print(torch.logsumexp(sum_concat, dim=1).size())
	return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1) 

def encoding(n,r,m,msg_bits):
	if r==0:
		temp=torch.ones(1,n**m).to(device)
		return msg_bits*temp
	if r==m:
		return msg_bits
	
	lefts = []
	lcd = get_dimen(n,r-1,m-1)
	
	for i in range(n-1):
		lefts.append(torch.unsqueeze(encoding(n,r-1,m-1,msg_bits[:,i*lcd:(i+1)*lcd]),2))


	rights = torch.unsqueeze(encoding(n,r,m-1,msg_bits[:,(n-1)*lcd:]),2)

	code = []
	for i in range(n-1):
		temp_tensor=torch.cat([lefts[i],rights],dim=2)
		code.append(torch.squeeze(gnet_dict["G_{}_{}".format(r,m)](temp_tensor)))

	code.append(torch.squeeze(rights))
	code = torch.hstack(code)

	return code

def decoding(n,r,m,code):
  
	if r==0:
		s = torch.sum(torch.squeeze(code),dim=1) > 0
		s = torch.unsqueeze(s.int(),1)
		return 2*s-1
	
	if r==m:
		msg = torch.squeeze(code) > 0
		msg = msg.int()
		return 2*msg-1

	msg_bits = torch.tensor([]).to(device)
	sub_code_len = np.power(n,m-1)
	sub_codes = []
	sub_codes_est = []
	codewords = []
	
	y = code[:,(n-1)*sub_code_len:]
	for i in range(n-1):
		sub_codes.append(code[:,i*sub_code_len:(i+1)*sub_code_len])
		x = code[:,i*sub_code_len:(i+1)*sub_code_len]
		L_v = fnet_dict["F_{}_{}_l".format(r,m)](torch.cat( \
				[x,y],dim=2)) + torch.unsqueeze(log_sum_exp(torch.cat([x,y],dim=2).permute(0, 2, 1)),2)
		# print("Inside Decoding")
		# print(L_v.size())
		# print(LSE.size())
		# v_hat = torch.tanh(L_v/2)
		sub_codes_est.append(L_v)
		m_i = decoding(n,r-1,m-1,sub_codes_est[-1])
		msg_bits = torch.hstack([msg_bits,m_i])
		c_i = torch.unsqueeze(encoding(n,r-1,m-1,m_i),dim=2)
		codewords.append(c_i)

	
	sub_codes.append(code[:,(n-1)*sub_code_len:])
	#sub_codes_est.append(torch.zeros(sub_codes[-1].size()).to(device))
	#codewords.append(torch.zeros(sub_codes[-1].size()).to(device))
	
	sub_codes	=	torch.cat(sub_codes,dim=2)
	sub_codes_est	=	torch.cat(sub_codes_est,dim=2)
	codewords	=	torch.cat(codewords,dim=2)

	final_tensor	=	torch.cat((sub_codes,sub_codes_est,codewords),dim=2)
	# final_tensor	=	torch.cat((sub_codes,codewords),dim=2)
	
	last_bits = fnet_dict["F_{}_{}_r".format(r,m)](final_tensor)

	msg_bits = torch.hstack([msg_bits,decoding(n,r,m-1,last_bits)])
	return msg_bits

def initialize(n,r,m,hidden_size,device):
	global enc_params
	global dec_params
	if r==0 or r==m:
		return
	if not gnet_dict.__contains__("G_{}_{}".format(r,m)):
		gnet_dict["G_{}_{}".format(r,m)] = g_Full(2, hidden_size, 1, device)
		fnet_dict["F_{}_{}_l".format(r,m)] = f_Full(2, hidden_size, 1, device)
		# fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(2*n, hidden_size, 1, device)
		fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(n+2*(n-1), hidden_size, 1, device)
		# fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(n+(n-1), hidden_size, 1, device)
		#fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(3*n, hidden_size, 1, device)

		gnet_dict["G_{}_{}".format(r,m)].apply(weights_init)
		fnet_dict["F_{}_{}_l".format(r,m)].apply(weights_init)
		fnet_dict["F_{}_{}_r".format(r,m)].apply(weights_init)

		enc_params += list(gnet_dict["G_{}_{}".format(r,m)].parameters())
		dec_params += list(fnet_dict["F_{}_{}_l".format(r,m)].parameters()) + \
		              list(fnet_dict["F_{}_{}_r".format(r,m)].parameters()) 	
		initialize(n,r-1,m-1,hidden_size,device)
		initialize(n,r,m-1,hidden_size,device)
	return

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
	
	data = {}

	bers = []
	losses = []

	gnet_dict = {}
	fnet_dict = {}

	enc_params = []
	dec_params = []

	para = conf["para"]
	seed = para["seed"]
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	today = date.today().strftime("%b-%d-%Y")
	data_type = para["data_type"]

	logger = get_logger(para["logger_name"])
	logger.info("train_conf_name : "+conf_name)
	logger.info("Device : "+str(device))
	logger.info("conf file")
	logger.info(conf)
	logger.info("We are on!!!")

	code_dimension = np.int64(para["code_dimension"])
	code_length = np.int64(para["code_length"])

	n = para["n"]
	r = para["r"]
	m = para["m"]


	print("code_dimension",code_dimension)
	print("code_length",code_length)

	rate = 1.0*(code_dimension/code_length)

	hidden_size = para["hidden_size"]

	initialize(n,r,m,hidden_size,device)

	criterion = BCEWithLogitsLoss()
		
	enc_optimizer = optim.Adam(enc_params, lr=para["lr"])
	dec_optimizer = optim.Adam(dec_params, lr=para["lr"])

	bers = []
	losses = []
	
	start_epoch = 0

	train_save_dirpath = para["train_save_path_dir"].format(today, data_type)
	if not os.path.exists(train_save_dirpath):
		os.makedirs(train_save_dirpath)
	
	torch.autograd.set_detect_anomaly(True)


	if para["retrain"]:
		train_model_path_encoder = para["train_save_path_encoder"].format(para["retrain_day"],para["data_type"],para["retrain_epoch_num"])
		train_model_path_decoder = para["train_save_path_decoder"].format(para["retrain_day"],para["data_type"],para["retrain_epoch_num"])
		start_epoch = int(para["retrain_epoch_num"])
		
		saved_model_enc = torch.load(model_path_encoder)
		saved_model_dec = torch.load(model_path_decoder)

		for key,val in saved_model_enc.items():
			gnet_dict[key].load_state_dict(val)

		for key,val in saved_model_dec.items():
			fnet_dict[key].load_state_dict(val)

		logger.info("Retraining Model " + conf_name + " : " +str(para["retrain_day"]) +" Epoch: "+str(para["retrain_epoch_num"]))


	# Training Algorithm
	try:
		for k in range(start_epoch, para["full_iterations"]):
			start_time = time.time()
			msg_bits_large_batch = 2*torch.randint(0,2,(para["train_batch_size"], code_dimension)).to(torch.float32) -1
			msg_bits_large_batch = msg_bits_large_batch.to(device)

			num_small_batches = int(para["train_batch_size"]/para["train_small_batch_size"])

			# Train decoder  
			for iter_num in range(para["dec_train_iters"]):
				dec_optimizer.zero_grad()        
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_input = msg_bits_large_batch[start:end,:]

					codewords = encoding(n,r,m,msg_input)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*sqrt(code_length)
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, para["dec_train_snr"],rate)
					L = get_LLR(corrupted_codewords, para["dec_train_snr"],rate)
					decoded_bits = decoding(n,r,m,L)
					# print(msg_input[:10,:])
					# print(decoded_bits[:10,:])
					decoded_bits.requires_grad=True

					loss = criterion(decoded_bits, msg_input)					
					loss.backward()
				dec_optimizer.step()

				print("Decoder",iter_num)
					
			# Train Encoder
			for iter_num in range(para["enc_train_iters"]):
				ber = 0
				enc_optimizer.zero_grad()        
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_input = msg_bits_large_batch[start:end].to(device)						
					codewords = encoding(n,r,m,msg_input)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*sqrt(code_length)
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, para["enc_train_snr"],rate)
					L = get_LLR(corrupted_codewords, para["enc_train_snr"],rate)
					decoded_bits = decoding(n,r,m,L)
					decoded_bits.requires_grad=True

					loss = criterion(decoded_bits, msg_input)
					loss.backward()
					
					ber = errors_ber(msg_input, decoded_bits).item()
				enc_optimizer.step()
				print("Encoder",iter_num)
				#ber /= num_small_batches	
				
			bers.append(ber)
			logger.info('[%d/%d] At ENC SNR %f dB DEC SNR %f dB, Loss: %.10f BER: %.10f' 
							% (k+1, para["full_iterations"], para["enc_train_snr"], para["dec_train_snr"], loss.item(), ber))
			logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

			losses.append(loss.item())
			if k % 10 == 0:
				torch.save(dict(zip(list(gnet_dict.keys()), [v.state_dict() for v in gnet_dict.values()])),para["train_save_path_encoder"].format(today, data_type, k+1))
				torch.save(dict(zip(list(fnet_dict.keys()), [v.state_dict() for v in fnet_dict.values()])),para["train_save_path_decoder"].format(today, data_type, k+1))

				plt.figure()
				plt.plot(bers)
				plt.plot(moving_average(bers, n=10))
				plt.legend(("bers","moving_average"))
				plt.xlabel("Iterations")
				plt.savefig(train_save_dirpath +'/training_ber.png')
				plt.close()

				plt.figure()
				plt.plot(losses)
				plt.plot(moving_average(losses, n=10))
				plt.legend(("bers","moving_average"))
				plt.xlabel("Iterations")
				plt.savefig(train_save_dirpath +'/training_losses.png')
				plt.close()

	except KeyboardInterrupt:
		logger.warning('Graceful Exit')
		exit()
	else:
		logger.warning('Finished')

	plt.figure()
	plt.plot(bers)
	plt.plot(moving_average(bers, n=5))
	plt.legend(("bers","moving_average"))
	plt.xlabel("Iterations")

	plt.savefig(train_save_dirpath +'/training_ber.png')
	plt.close()

	plt.figure()
	plt.plot(losses)
	plt.plot(moving_average(losses, n=5))
	plt.legend(("bers","moving_average"))
	plt.xlabel("Iterations")
	plt.savefig(train_save_dirpath +'/training_losses.png')
	plt.close()
	torch.save(dict(zip(list(gnet_dict.keys()), [v.state_dict() for v in gnet_dict.values()])),para["train_save_path_encoder"].format(today, data_type, para["full_iterations"]))

	torch.save(dict(zip(list(fnet_dict.keys()), [v.state_dict() for v in fnet_dict.values()])),para["train_save_path_decoder"].format(today, data_type, para["full_iterations"]))
