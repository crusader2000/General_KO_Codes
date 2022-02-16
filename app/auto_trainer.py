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


import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt


def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


bers = []
losses = []

gnet_dict = {}
fnet_dict = {}

enc_params = []
dec_params = []

def encoding(n,r,m,msg_bits):
	if r==0:
		return msg_bits[0]*np.ones((1,np.power(n,m)))
	if r==m:
		return np.array(msg_bits)

	lefts = []
	lcd = get_dimen(n,r-1,m-1)
	for i in range(n-1):
		lefts.append(encoding(n,r-1,m-1,msg_bits[i*lcd:(i+1)*lcd]))
	
	right = encoding(n,r,m-1,msg_bits[(n-1)*lcd:])
 
	code = np.array([])

	for i in range(n-1):
		np.append(code,gnet_dict["G_{}_{}".format(r,m)](torch.cat([lefts[i],right],dim=2)))
	np.append(code,right)

	return code

def perform_ML(n,m,code):
	return

def perform_ML_repeatition(n,m,code):
	return

def decoding(n,r,m,code):
  	
	if r==0:
		return perform_ML_repeatition(n,m,code)
	
	if r==m:
		return perform_ML(n,m,code)
	
	msg_bits = np.array([])
	sub_code_len = np.power(n,m-1)
	sub_codes = []
	sub_codes_est = []
	for i in range(n-1):
  	sub_codes.append(code[i*sub_code_len:(i+1)*sub_code_len])
		sub_codes_est.append(fnet_dict["F_{}_{}_l".format(r,m)](torch.cat( \
				[code[i*sub_code_len:(i+1)*sub_code_len],code[(n-1)*sub_code_len:]],dim=2)))
		np.append(msg_bits,decoding(n,r-1,m-1,sub_codes_est[-1]))
	
	sub_codes.append(code[(n-1)*sub_code_len:])
	sub_codes_est.append(np.zeros((1,n**m)))

	last_bits = fnet_dict["F_{}_{}_r".format(r,m)](torch.hstack( \
			[np.array(sub_codes_est),np.array(sub_codes)],dim=2))

	np.append(msg_bits,decoding(n,r,m-1,last_bits))

	return msg_bits

def initialize(n,r,m,hidden_size):
	if r==0 or r==m:
		return
	if not gnet_dict.__contains__("G_{}_{}".format(r,m)):
		gnet_dict["G_{}_{}".format(r,m)] = g_Full(2, hidden_size, 1)
		fnet_dict["F_{}_{}_l".format(r,m)] = f_Full(2, hidden_size, 1)
		fnet_dict["F_{}_{}_r".format(r,m)] = f_Full(2*n, hidden_size, 1)
		enc_params += list(gnet_dict["G_{}_{}".format(r,m)].parameters())
		dec_params += list(fnet_dict["F_{}_{}_l".format(r,m)].parameters()) + \
		              list(fnet_dict["F_{}_{}_r".format(r,m)].parameters()) 	
		initialize(n,r-1,m-1,hidden_size)
		initialize(n,r,m-1,hidden_size)
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
	logger.info("We are on!!!")

	code_dimension = para["code_dimension"]
	code_length = para["code_length"]

	n = para["n"]
	r = para["r"]
	m = para["m"]

	hidden_size = para["hidden_size"]

	initialize(n,r,m,hidden_size)

	criterion = BCEWithLogitsLoss()
	enc_optimizer = optim.Adam(enc_params, lr=para["lr"])
	dec_optimizer = optim.Adam(dec_params, lr=para["lr"])
	enc_scheduler = ReduceLROnPlateau(enc_optimizer, 'min')
	dec_scheduler = ReduceLROnPlateau(dec_optimizer, 'min')

	
	bers = []
	losses = []
	
	train_save_dirpath = para["train_save_path_dir"].format(today, data_type)
	if not os.path.exists(train_save_dirpath):
		os.makedirs(train_save_dirpath)
	
	torch.autograd.set_detect_anomaly(True)

	# Training Algorithm
	try:
		for k in range(start_epoch, para["full_iterations"]):
			start_time = time.time()
			msg_bits_large_batch = 2*torch.randint(0,2,(para["train_batch_size"], code_dimension)).to(torch.float) -1

			num_small_batches = int(para["train_batch_size"]/para["train_small_batch_size"])

			# Train decoder  
			for iter_num in range(para["dec_train_iters"]):
				dec_optimizer.zero_grad()        
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_bits = msg_bits_large_batch[start:end].to(device)
					codewords = encoding(n,r,m,msg_bits)      
					# print("codewords")
					# print(codewords)
					transmit_codewords = F.normalize(torch.hstack((msg_bits,codewords)), p=2, dim=1)*np.sqrt(code_length)
					# print("transmit_codewords")
					# print(transmit_codewords)
					corrupted_codewords = awgn_channel(transmit_codewords, para["dec_train_snr"])
					# print("corrupted_codewords")
					# print(corrupted_codewords)
					
					decoded_bits = decoding(n,r,m,corrupted_codewords)
					# print("decoded_bits")
					# decoded_bits = torch.nan_to_num(decoded_bits,0.0)
					# print(decoded_bits)
					loss = criterion(decoded_bits, msg_bits)/num_small_batches
					
					# print(loss)
					loss.backward()
				dec_scheduler.step(loss)
				print("Decoder",iter_num)
				dec_optimizer.step()
					
			# Train Encoder
			for iter_num in range(para["enc_train_iters"]):
				enc_optimizer.zero_grad()        
				ber = 0
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_bits = msg_bits_large_batch[start:end].to(device)						
					codewords = encoding(n,r,m,msg_bits)      
					transmit_codewords = F.normalize(torch.hstack((msg_bits,codewords)), p=2, dim=1)*np.sqrt(code_length)
					corrupted_codewords = awgn_channel(transmit_codewords, para["enc_train_snr"])
					decoded_bits = decoding(n,r,m,corrupted_codewords)

					loss = criterion(decoded_bits, msg_bits )/num_small_batches
					
					loss.backward()
					ber += errors_ber(msg_bits, decoded_bits.sign()).item()

				enc_scheduler.step(loss)
				print("Encoder",iter_num)
				enc_optimizer.step()
				ber /= num_small_batches	
				
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