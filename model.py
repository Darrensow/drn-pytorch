import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

class JSLoss(nn.Module):
  def __init__(self):
    super(JSLoss, self).__init__()

  def forward(self, P, Q):
    M = 0.5 * (P + Q)
    JS = 0.5 * (self.KL_div(P, M) + self.KL_div(Q, M)) / torch.log(torch.tensor(2.0))
    scalar_JS = torch.mean(JS)
    # print(scalar_JS)
    return scalar_JS

  def KL_div(self, P, Q):
    P = torch.clamp(P, min = 1e-15)  # Clip values to prevent log(0)
    Q = torch.clamp(Q, min = 1e-15)
    return torch.sum(P * torch.log(P / Q), dim = 2)  # Adjust dimensions as needed

class CustomNet(nn.Module):
  def __init__(self, in_dim : int, num_hidden: int, hidden_dim : int,
               out_dim : int, bin : int, hidden_bin : int):
    super(CustomNet, self).__init__()
    self.in_dim = in_dim
    self.num_hidden = num_hidden
    self.hidden_dim = hidden_dim
    self.out_dim = out_dim
    self.bin = bin
    self.hidden_bin = hidden_bin

    if self.num_hidden == 0:
      self.in_layer = CustomLayer(self.in_dim, self.out_dim, self.bin, self.bin)
    else:
      self.in_layer = CustomLayer(self.in_dim, self.hidden_dim, self.bin, self.hidden_bin)
      for layer in range(num_hidden - 1):
        setattr(self, f'hidden_{layer}', CustomLayer(self.hidden_dim, self.hidden_dim, self.hidden_bin, self.hidden_bin))
      self.out_layer = CustomLayer(self.hidden_dim, self.out_dim, self.hidden_bin, self.bin)


  def forward(self, x):
    for i, layer in enumerate(self.children()):
      # print(i)
      x = layer(x)
    return x

class CustomLayer(nn.Module):
  def __init__(self, n_lower : int, n_upper : int, bin_lower : int, bin_upper : int,
               weight_init = 0.1, weight_init_method = 'uniform'):
    super(CustomLayer, self).__init__()
    self.n_lower = n_lower
    self.n_upper = n_upper
    self.bin_lower = bin_lower
    self.bin_upper = bin_upper
    self.weight_init = weight_init
    self.weight_init_method = weight_init_method

    self.init_weights()
    self.D = self.initD()


  def init_weights(self):
    initializers = {
              'uniform': init.uniform_,
              'normal': init.normal_,
              'glorot_normal': init.xavier_normal_
          }
    initializer = initializers.get(self.weight_init_method, init.uniform_)

    self.weights = nn.Parameter(torch.empty((self.n_upper, self.n_lower)))
    self.abs_bias = nn.Parameter(torch.empty(self.n_upper, 1))
    self.quad_bias = nn.Parameter(torch.empty(self.n_upper, 1))
    self.abs_lambda = nn.Parameter(torch.empty(self.n_upper, 1).uniform_(0.0, 1.0))
    self.quad_lambda = nn.Parameter(torch.empty(self.n_upper, 1).uniform_(0.0, 1.0))

    if self.weight_init_method == 'uniform':
      for param in [self.weights, self.abs_bias, self.quad_bias]:
        initializer(param, a = -self.weight_init, b = self.weight_init)
    else:
      for param in [self.weights, self.abs_bias, self.quad_bias]:
        initializer(param)


  def initD(self):
    si = np.arange(self.bin_lower, dtype = float)
    sk = np.arange(self.bin_upper, dtype = float)

    si_mat = np.tile(si, (self.bin_upper, 1))
    sk_mat = np.tile(sk[:, np.newaxis], (1, self.bin_lower))

    D_np = np.exp(-((sk_mat / self.bin_upper - si_mat / self.bin_lower) ** 2))
    D_tensor = torch.tensor(D_np[..., np.newaxis, np.newaxis], dtype=torch.float32)
    D = D_tensor.expand(-1, -1, self.n_upper, self.n_lower)
    return D


  def cal_logexp_bias(self):
    s0 = torch.arange(self.bin_upper, dtype=torch.float32).reshape(1, self.bin_upper)
    B = -self.quad_bias * (s0 / self.bin_upper - self.quad_lambda) ** 2 - self.abs_bias * torch.abs(s0 / self.bin_upper - self.abs_lambda)  # nu x qu
    return B


  def forward(self, x):
    P_reshaped = torch.reshape(x, [-1, 1, self.n_lower, self.bin_lower, 1])
    Ptile = torch.tile(P_reshaped, [1, self.n_upper, 1, 1, 1])  # bs x nu x nl x ql x 1
    T = torch.permute(torch.pow(self.D, self.weights), [2, 3, 0, 1]) # nu x nl x qu x ql

    # Einstein summation : sum(P(s) * e^{wD})
    Pw_unclipped = torch.squeeze(torch.einsum('jklm,ijkmn->ijkln', T, Ptile), dim = 4)
    Pw = torch.clamp(Pw_unclipped, 1e-15, 1e+15)

    # unnormalized p(sk) = e^{B_sk} * sum
    # unnormalized P(sk) = B_sk + logsum
    logPw = torch.log(Pw)
    logsum = torch.sum(logPw, dim = 2)
    B_sk = self.cal_logexp_bias()
    logsumB = logsum + B_sk

    # Normalization
    max_logsum = torch.max(logsumB, dim = 2, keepdim = True).values
    expm_P = torch.exp(logsumB - max_logsum)
    Z = torch.sum(expm_P, dim=2, keepdim = True)
    ynorm = expm_P / Z

    return ynorm