import torch
from torch import nn

class Transformer(nn.Module):
  def __init__(
      self,
      input_size,
      output_size,
      kernel_size,
      seq_len,
      dropout,
  )