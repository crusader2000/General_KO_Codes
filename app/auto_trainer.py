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
	# device = torch.device("cuda")
	if r==0:
		# print("r=0")
		# print(msg_bits.size())
		# print((msg_bits*np.ones((1,np.power(n,m)))).size())
		#print('msg_bits is a tensor:  ',torch.is_tensor(msg_bits))
		temp=np.ones((1,np.power(n,m)))
		temp_tensor=torch.from_numpy(temp)
		
		temp_tensor_gpu=temp_tensor.to(device)
		return msg_bits*temp_tensor_gpu
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

	# data = torch.load(para["data_file"])

	initialize(n,r,m,hidden_size,device)

	criterion = BCEWithLogitsLoss()
	enc_optimizer = optim.Adam(enc_params, lr=para["lr"])
	dec_optimizer = optim.Adam(dec_params, lr=para["lr"])
	enc_scheduler = ReduceLROnPlateau(enc_optimizer, 'min')
	dec_scheduler = ReduceLROnPlateau(dec_optimizer, 'min')

	
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

			num_small_batches = int(para["train_batch_size"]/para["train_small_batch_size"])

			# Train decoder  
			for iter_num in range(para["dec_train_iters"]):
				dec_optimizer.zero_grad()        
				for i in range(num_small_batches):
					start, end = i*para["train_small_batch_size"], (i+1)*para["train_small_batch_size"]
					msg_input = msg_bits_large_batch[start:end,:].to(device)
					# print("train msg_bits.is_cuda")
					# print(msg_input.is_cuda)

					codewords = encoding(n,r,m,msg_input)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*sqrt(code_length)
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, para["dec_train_snr"],rate)
					L = get_LLR(corrupted_codewords, para["dec_train_snr"],rate)
					decoded_bits = decoding(n,r,m,L)
					# decoded_bits=decoded_bits.to(device)
					# msg_input=msg_input.to(device)
					decoded_bits.requires_grad=True
					msg_input.requires_grad=True
					# decoded_bits = torch.squeeze(decoded_bits,dim = 2)

					# print("decoded_bits.size()")
					# print(decoded_bits)
					# print(decoded_bits.size())
					# print("msg_input.size()")
					# print(msg_input)
					# print(msg_input.size())
					loss = criterion(decoded_bits, msg_input)/num_small_batches
					
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
					msg_input = msg_bits_large_batch[start:end].to(device)						
					codewords = encoding(n,r,m,msg_input)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*sqrt(code_length)
					# Adding this because this is only thing different from decoder. TO fix shape error
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, para["enc_train_snr"],rate)
					L = get_LLR(corrupted_codewords, para["enc_train_snr"],rate)
					decoded_bits = decoding(n,r,m,L)
					# decoded_bits=decoded_bits.to(device)
					# msg_input=msg_input.to(device)
					decoded_bits.requires_grad=True
					# msg_input.requires_grad=True
					decoded_bits = torch.unsqueeze(decoded_bits,dim = 2)

					loss = criterion(decoded_bits, msg_input )/num_small_batches
					
					loss.backward()
					ber += errors_ber(msg_input, decoded_bits.sign()).item()

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
