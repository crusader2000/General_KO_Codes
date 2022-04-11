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
from app.operations import * 
from model.models import *
import app.sharedstuff as sharedstuff

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

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


# data = {}

bers = []
losses = []
	
def test(snr, rate, code_dimension):
	BER_total = []
	Test_msg_bits = 2*torch.randint(0,2,(test_size, code_dimension)).to(torch.float) -1
	Test_msg_bits = Test_msg_bits.to(sharedstuff.device)
	Test_Data_Generator = DataLoader(Test_msg_bits, batch_size=100 , shuffle=False)

	num_test_batches = len(Test_Data_Generator)
	ber = 0
	start_time = time.time()

	with torch.no_grad():
			for msg_bits in Test_Data_Generator:
					# msg_bits.to(sharedstuff.device)
					codewords = encoding(n,r,m,msg_bits)      
					transmit_codewords = F.normalize(codewords, p=2, dim=1)*np.sqrt(code_length)
					transmit_codewords = torch.unsqueeze(transmit_codewords,2)
					corrupted_codewords = awgn_channel(transmit_codewords, snr, rate)
					L = get_LLR(corrupted_codewords, snr,rate)
					decoded_bits = decoding(n,r,m,L)

					# decoded_bits=decoded_bits.to('cuda')
					# msg_bits=msg_bits.to('cuda')
					ber += errors_ber(msg_bits, decoded_bits.sign()).item()
			
			ber /= num_test_batches
			logger.warning(f"[Testing Block] SNR={snr} : BER={ber:.7f}")
			logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

	return ber

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

	para = conf["para"]
	seed = para["seed"]
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	test_conf = conf["test"]

	today = date.today().strftime("%b-%d-%Y")

	logger = get_logger(test_conf["logger_name"])
	logger.info("test_conf_name : "+conf_name)
	logger.info("Device : "+str(sharedstuff.device))

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

	initialize(n,r,m,hidden_size)

	test_size = para["test_size"]
	test_model_path_encoder = test_conf["test_model_path_encoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])
	test_model_path_decoder = test_conf["test_model_path_decoder"].format(test_conf["day"],para["data_type"],test_conf["epoch_num"])
	
	test_save_dirpath = para["train_save_path_dir"].format(test_conf["day"], para["data_type"])
	if not os.path.exists(test_save_dirpath):
			os.makedirs(test_save_dirpath)

	saved_model_enc = torch.load(test_model_path_encoder)
	saved_model_dec = torch.load(test_model_path_decoder)

	for key,val in saved_model_enc.items():
		sharedstuff.gnet_dict[key].load_state_dict(val)

	for key,val in saved_model_dec.items():
		sharedstuff.fnet_dict[key].load_state_dict(val)

	bers = []
	snrs = []
	logger.info("Testing {} trained till epoch_num {}".format(conf_name,test_conf["epoch_num"]))
	logger.info("Model trained on {}".format(test_conf["day"]))
	logger.info("Less go!")
	for snr in test_conf["snr_list"].split(","):
			
			ber = test(int(snr), rate, code_dimension)
			bers.append(ber)
			snrs.append(int(snr))
	plt.semilogy(snrs, bers, marker='o', linewidth=1.5)
	plt.grid()
	#plt.plot(snrs, bers, label=" ",linewidth=2, color='blue')
       	#ax = plt.gca()
	#ax.set_yscale([1e-6, 1])        
	for a,b in zip(snrs,bers):
		plt.text(a,b,str(format_e(b)))
        
	plt.xlabel("SNRs")
	plt.ylabel("BERs (Testing)")
	plt.title("Testing BERs")
	plt.savefig(test_save_dirpath+ "/ber_testing_epoch_"+str(test_conf["epoch_num"])+".png")
	plt.close()
