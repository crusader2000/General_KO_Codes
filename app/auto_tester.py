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
from app.burman import *

import sys
import pickle
import glob
import os
import logging
import time
from datetime import datetime
from datetime import date
import random
# from data.generate_data import *

from math import sqrt

import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt


def format_e(num):
    a = '%E' % num
    return a.split('E')[0].rstrip('0').rstrip('.')[:4] + 'E' + a.split('E')[1]

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(sharedstuff.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

# data = {}

bers = []
losses = []
	
def test(snr, rate, code_dimension):
	BER_total = []
	Test_msg_bits = 2*torch.randint(0,2,(test_size, code_dimension)).to(torch.float) -1
	Test_msg_bits = Test_msg_bits.to(sharedstuff.device)
	Test_Data_Generator = DataLoader(Test_msg_bits, batch_size=100, shuffle=False)

	num_test_batches = len(Test_Data_Generator)
	ber_burman = 0
	ber_nn = 0
	start_time = time.time()

	with torch.no_grad():
		# print(n,r,m)
		for msg_bits in Test_Data_Generator:
			# msg_bits.to(sharedstuff.device)
			codeword_burman = encode_burman(n,r,m,(torch.clone(msg_bits)+1)/2)
			codewords_nn = encoding(n,r,m,torch.clone(msg_bits))

			transmit_codewords_nn = F.normalize(codewords_nn, p=2, dim=1)*np.sqrt(code_length)
			transmit_codewords_burman = 2*codeword_burman - 1
			

			corrupted_codewords_burman,corrupted_codewords_nn = test_awgn_channel(transmit_codewords_burman,transmit_codewords_nn,snr,rate)

			corrupted_codewords_burman = torch.unsqueeze(corrupted_codewords_burman,2)
			corrupted_codewords_nn = torch.unsqueeze(corrupted_codewords_nn,2)
			
			L_burman = get_LLR(corrupted_codewords_burman, snr,rate)
			# decoded_bits_burman = decode_burman(n,r,m,L_burman)
			decoded_bits_burman = decode_burman(n,r,m,(L_burman>0).to(torch.int64))

			L_nn = get_LLR(corrupted_codewords_nn, snr,rate)
			decoded_bits_nn = decoding(n,r,m,L_nn)
			ber_burman += errors_ber((msg_bits+1)/2, decoded_bits_burman).item()
			ber_nn += errors_ber(msg_bits, decoded_bits_nn.sign()).item()
	
		ber_burman /= num_test_batches
		ber_nn /= num_test_batches
		logger.warning(f"[Testing Block] Burman SNR={snr} : BER={ber_burman:.10f}")
		logger.warning(f"[Testing Block] GKO SNR={snr} : BER={ber_nn:.10f}")
		logger.info("Time for one full iteration is {0:.4f} minutes".format((time.time() - start_time)/60))

	return ber_burman,ber_nn

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

	for i in range(2**code_dimension):
		msg_bits = dec2bin(torch.tensor([i]).to(sharedstuff.device),code_dimension)
		curr_codeword = encode_burman(n,r,m,msg_bits)
		sharedstuff.codebook_msg_bits.append(msg_bits)
		sharedstuff.codebook.append(curr_codeword)

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

	# KO(1,6) Channel Model 2
	# bers_dumer = [0.313688568174839,0.264614283293486,0.214914283305407,0.159894283115864,0.106405713111162,0.062264284864068,0.030767142493278,0.012194285602309,0.003567142811371,0.000727142849937,0.0000514285708777607]

	# KO(1,6) Channel Model 1
	# bers_dumer = [0.294888568520546,0.245595710426569,0.192464283555746,0.138125712648034,0.088437141776085,0.048258571103215,0.021572857033461,0.007621428444982,0.001862857118249,0.000275714282179,0.0000428571423981339]

	bers_burman = []
	bers_nn = []
	snrs = []
	logger.info("Testing {} trained till epoch_num {}".format(conf_name,test_conf["epoch_num"]))
	logger.info("Model trained on {}".format(test_conf["day"]))
	logger.info("Less go!")
	for snr in test_conf["snr_list"].split(","):
			
			ber_burman,ber_nn = test(int(snr), rate, code_dimension)
			bers_burman.append(ber_burman)
			bers_nn.append(ber_nn)
			snrs.append(int(snr))
	
	plt.figure(figsize = (12,12))
	plt.semilogy(snrs, bers_burman, label="Burman", marker='^', linewidth=1.5)
	for a,b in zip(snrs,bers_burman):
  		plt.text(a,b,str(format_e(b)))
			
	plt.semilogy(snrs, bers_nn, label="GKO", marker='o', linewidth=1.5)
	plt.grid()
	for a,b in zip(snrs,bers_nn):
		plt.text(a,b,str(format_e(b)))
        
	plt.xlabel("SNRs")
	plt.ylabel("BERs (Testing)")
	plt.title("Testing BERs")
	plt.legend(prop={'size': 15})
	plt.savefig(test_save_dirpath+ "/ber_testing_epoch_"+str(test_conf["epoch_num"])+".png")
	plt.close()
