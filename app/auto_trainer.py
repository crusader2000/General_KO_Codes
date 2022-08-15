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
from app.operations import * 
import app.sharedstuff as sharedstuff

import sys
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

# def automorphism(n ,codeword, num_samples) :
# 	# codeword is a individual tensor
# 	temp_list = np.array(torch.split(codeword,n))
# 	perm_list = temp_list[:-1]
# 	last_ele = temp_list[-1]
# 	automorphism_list = []
# 	for i in range(0,num_samples):
# 		np.random.shuffle(perm_list)
# 		temp_tensor = torch.cat(tuple(perm_list))
# 		temp_tensor = torch.cat(temp_tensor , last_ele)
# 		automorphism_list.append(temp_tensor)
# 	return automorphism_list
# data = {}

bers = []
losses = []

if __name__ == "__main__":
	sharedstuff.init() 
	
	if len(sys.argv) == 2:
		conf_name = sys.argv[1]
		print("train conf_name:", conf_name)
		conf = get_default_conf(f"./config/{conf_name}.yaml")
	else:
		print("default")
		conf = get_default_conf()

	if torch.cuda.is_available():
		sharedstuff.device = torch.device("cuda")
		os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
		print(sharedstuff.device,os.environ["CUDA_VISIBLE_DEVICES"])
	else:
		sharedstuff.device = torch.device("cpu")
		print(sharedstuff.device)
	
	data = {}

	bers = []
	losses = []

	sharedstuff.gnet_dict = {}
	sharedstuff.fnet_dict = {}

	sharedstuff.enc_params = []
	sharedstuff.dec_params = []

	para = conf["para"]
	seed = para["seed"]
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	today = date.today().strftime("%b-%d-%Y")
	data_type = para["data_type"]

	logger = get_logger(para["logger_name"])
	logger.info("train_conf_name : "+conf_name)
	logger.info("Device : "+str(sharedstuff.device))
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

	initialize(n,r,m,hidden_size)

	criterion = BCEWithLogitsLoss()
		
	enc_optimizer = optim.Adam(sharedstuff.enc_params, lr=para["lr"])
	dec_optimizer = optim.Adam(sharedstuff.dec_params, lr=para["lr"])

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
			sharedstuff.gnet_dict[key].load_state_dict(val)

		for key,val in saved_model_dec.items():
			sharedstuff.fnet_dict[key].load_state_dict(val)

		logger.info("Retraining Model " + conf_name + " : " +str(para["retrain_day"]) +" Epoch: "+str(para["retrain_epoch_num"]))


	# Training Algorithm
	try:
		for k in range(start_epoch, para["full_iterations"]):
			start_time = time.time()
			msg_bits_large_batch = 2*torch.randint(0,2,(para["train_batch_size"], code_dimension)).to(torch.float32) -1
			msg_bits_large_batch = msg_bits_large_batch.to(sharedstuff.device)

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
					# L = get_LLR(corrupted_codewords, para["enc_train_snr"],rate)
					# decoded_bits = decoding(n,r,m,L)
					decoded_bits = decoding(n,r,m,corrupted_codewords)

					# print(msg_input[:10,:])
					# print(decoded_bits[:10,:])
					# decoded_bits.requires_grad=True

					loss = criterion(decoded_bits, 0.5*msg_input+0.5)/num_small_batches					
					loss.backward()
				dec_optimizer.step()

				print("Decoder",iter_num)
					
			# Train Encoder
			for iter_num in range(para["enc_train_iters"]):
				ber = 0
				enc_optimizer.zero_grad()        
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_input = msg_bits_large_batch[start:end].to(sharedstuff.device)						
					codewords = encoding(n,r,m,msg_input)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*sqrt(code_length)
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, para["enc_train_snr"],rate)
					# L = get_LLR(corrupted_codewords, para["enc_train_snr"],rate)
					# decoded_bits = decoding(n,r,m,L)
					decoded_bits = decoding(n,r,m,corrupted_codewords)

					loss = criterion(decoded_bits, 0.5*msg_input+0.5)/num_small_batches					
					loss.backward()
					
				ber = errors_ber(msg_input, decoded_bits.sign()).item()
				
				enc_optimizer.step()
				print("Encoder",iter_num)
				#ber /= num_small_batches	
				
			bers.append(ber)
			logger.info('[%d/%d] At ENC SNR %f dB DEC SNR %f dB, Loss: %.10f BER: %.10f' 
							% (k+1, para["full_iterations"], para["enc_train_snr"], para["dec_train_snr"], loss.item(), ber))
			logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

			losses.append(loss.item())
			if k % 10 == 0:
				torch.save(dict(zip(list(sharedstuff.gnet_dict.keys()), [v.state_dict() for v in sharedstuff.gnet_dict.values()])),para["train_save_path_encoder"].format(today, data_type, k+1))
				torch.save(dict(zip(list(sharedstuff.fnet_dict.keys()), [v.state_dict() for v in sharedstuff.fnet_dict.values()])),para["train_save_path_decoder"].format(today, data_type, k+1))

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
	plt.legend(("bers","moving_average_bers"))
	plt.xlabel("Iterations")

	plt.savefig(train_save_dirpath +'/training_ber.png')
	plt.close()

	plt.figure()
	plt.plot(losses)
	plt.plot(moving_average(losses, n=5))
	plt.legend(("losses","moving_average_losses"))
	plt.xlabel("Iterations")
	plt.savefig(train_save_dirpath +'/training_losses.png')
	plt.close()
	torch.save(dict(zip(list(sharedstuff.gnet_dict.keys()), [v.state_dict() for v in sharedstuff.gnet_dict.values()])),para["train_save_path_encoder"].format(today, data_type, para["full_iterations"]))

	torch.save(dict(zip(list(sharedstuff.fnet_dict.keys()), [v.state_dict() for v in sharedstuff.fnet_dict.values()])),para["train_save_path_decoder"].format(today, data_type, para["full_iterations"]))
