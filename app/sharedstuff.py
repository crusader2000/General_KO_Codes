def init():
  global gnet_dict
  global fnet_dict

  global enc_params
  global dec_params

  global device

  global codebook
  global codebook_msg_bits

  gnet_dict = {}
  fnet_dict = {}

  enc_params = []
  dec_params = []

  device = None

  codebook = {}
  codebook_msg_bits = {}

  am_list = []

  belief_prop = None