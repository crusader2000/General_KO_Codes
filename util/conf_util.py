import yaml
import numpy as np
from math import comb

def get_dimen(n,r,m):
	code_dimension = 0
	for w in range(0,r+1):
		code_dimension += comb(n,w)*np.power((n-1),w)
	return code_dimension

def get_default_conf(conf_path=None):
	if conf_path is None:
		conf_path = "./config/default.yaml"
	with open(conf_path, "r") as f_conf:
		conf = yaml.load(f_conf.read(), Loader=yaml.FullLoader)

	data_type = conf["para"]["data_type"]

	n = conf["para"]["n"]
	m = conf["para"]["m"]
	r = conf["para"]["r"]

	conf["para"]["logger_name"] = conf["para"]["logger_name"].format(data_type)
	conf["test"]["logger_name"] = conf["test"]["logger_name"].format(data_type)
	

	conf["para"]["code_length"] = np.power(n,m)
	conf["para"]["code_dimension"] = get_dimen(n,r,m)

	return conf
