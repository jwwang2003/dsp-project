import torch
from torch import nn
from nn.TCN1 import TemporalConvNet
import matplotlib.pyplot as plt

class BiTCN(nn.Module):
  def __init__(self, 
    input_size,
    output_size,
    num_channels,
    kernel_size,
    seq_len,
    dropout
  ):
    super(BiTCN, self).__init__()
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    # num_channels[-1]正是TCN最后一个时间卷积层输出的特征数。
    self.linear = nn.Linear(num_channels[-1] * 2 * seq_len, output_size)

  def forward(self, inputs):
    """Inputs have to have dimension (N, C_in, L_in)"""
    inputs = inputs.permute(0,2,1)
    y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
    y2 = self.tcn(inputs.flip(dims=[2]))
    y = torch.cat((y1, y2), dim=1)
    y_flat = y.reshape(y.shape[0],-1)
    # 考虑到模型处理的是批量数据（即一次处理多个样本），PyTorch模型的最后一个线性层不需要显式地处理批量这一维度。
    # PyTorch会自动处理批量维度，你只需要定义每个样本的输入和输出特征数量即可。
    o = self.linear(y_flat)
    return o

class TCN_BiGRU(nn.Module):
  def __init__(self,
    input_size,
    num_channels,
    kernel_size,
    hidden_size,
    dropout,
    num_layers,
    output_size,
    seq_len
  ):
    super(TCN_BiGRU, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.seq_len = seq_len
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    self.gru = nn.GRU(num_channels[-1], hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_size*2*seq_len, output_size)  # 2 for bidirection

  def forward(self, x):
    x = x.permute(0,2,1)
    x = self.tcn(x)
    x = x.permute(0,2,1)
    out, _ = self.gru(x)
    out = out.reshape(out.shape[0],-1)
    out = self.fc(out)
    return out


def decreasing_list_by_division(start):
  result = []
  while start >= 50:
      result.append(int(start))
      start /= 8
  result.append(1)
  return result

def plot_weight_distribution(model):
  # 对每个模块进行循环，检查是否为线性层（或任何具有权重属性的层）
  for name, module in model.named_modules():
    if hasattr(module, 'weight'):
      # 获取权重数据
      weights = module.weight.data.cpu().numpy().flatten()  # 确保数据在CPU上，并转化为一维数组
      plt.figure(figsize=(10, 7))
      plt.hist(weights, bins=50)
      plt.title(f'Weight Distribution in {name}')
      plt.xlabel('Weight Values')
      plt.ylabel('Frequency')
      plt.grid(True)
      plt.show()