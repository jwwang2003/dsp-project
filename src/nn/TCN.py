"""
This file contains code authored by another student.
The purpose is just for reference, testing, and experimentation.
We will build upon this foundation and try to make it better.
"""

import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Chomp1D(nn.Module):
  """
  Removes a certain number of elements from the end of the tensor, ensuring that the output of the convolution is the same length as the input (for causal convolutions).
  """
  def __init__(self, chomp_size):
    super(Chomp1D, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    """
    Purpose: trims the tensor along the last dimension.
    """
    return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TemporalBlock, self).__init__()
    self.conv1 = weight_norm(
      nn.Conv1d(
        n_inputs, 
        n_outputs, 
        kernel_size,
        stride=stride,
        padding=padding, 
        dilation=dilation
      )
    )
    
    self.chomp1 = Chomp1D(padding)
    self.relu1 = nn.PReLU()
    self.dropout1 = nn.Dropout(dropout)

    self.conv2 = weight_norm(
      nn.Conv1d(
        n_outputs,
        n_outputs,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
      )
    )
    
    self.chomp2 = Chomp1D(padding)
    self.relu2 = nn.PReLU()
    self.dropout2 = nn.Dropout(dropout)

    self.net = nn.Sequential(
      self.conv1,
      self.chomp1,
      self.relu1,
      self.dropout1,
      self.conv2,
      self.chomp2,
      self.relu2,
      self.dropout2
    )
    
    self.downsample = nn.Conv1d(
      n_inputs,
      n_outputs,
      1
    ) if n_inputs != n_outputs else None
    
    self.relu = nn.PReLU()
    self.init_weights()

  def init_weights(self):
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv2.weight.data.normal_(0, 0.01)
    if self.downsample is not None:
      self.downsample.weight.data.normal_(0, 0.01)

  def forward(self, x):
    out = self.net(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.relu(out + res)


class TemporalConvNet(nn.Module):
  """
  Wrapper for Temporal Convolution Network
  """
  def __init__(self, 
    num_inputs, num_channels, kernel_size=2, dropout=0.2
  ):
    super(TemporalConvNet, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2 ** i
      in_channels = num_inputs if i == 0 else num_channels[i-1]
      out_channels = num_channels[i]
      layers += [
        TemporalBlock(
          in_channels,
          out_channels,
          kernel_size,
          stride=1,
          dilation=dilation_size,
          padding=(kernel_size-1) * dilation_size, 
          dropout=dropout
        )
      ]
      
      self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)