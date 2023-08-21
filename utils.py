import torch

import numpy as np
import random
import requests
from io import StringIO

def seed_everything(seed=429):
  """Function to set reproducibility of results"""
  random.seed(seed)
  #os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def load_data(url):
    response = requests.get(url)
    data_content = response.text
    data_file = StringIO(data_content)
    data = np.loadtxt(data_file)
    return data
