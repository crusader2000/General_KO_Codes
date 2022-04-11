def init():
  global gnet_dict
  global fnet_dict

  global enc_params
  global dec_params

  global device

  gnet_dict = {}
  fnet_dict = {}

  enc_params = []
  dec_params = []

  device = None