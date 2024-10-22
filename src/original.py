import numpy as np
import torch
import torch.nn as nn

from src.nn.TCNN1 import BiTCN, decreasing_list_by_division

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 创建一个文件处理器，将日志写入文件
file_handler = logging.FileHandler('training.log', mode='w')
file_handler.setLevel(logging.INFO)
# 创建一个流处理器，将日志输出到控制台
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
# 将处理器添加到日志记录器
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_file(file_path):
  data = []
  try:
    with open(file_path, 'r') as file:
      for line in file:
        # line.strip()是一个字符串方法，它用于去除字符串首尾的空白字符
        line_data = line.strip().split()
        data.append([float(x) for x in line_data])
    return data
  except Exception as e:
    print(f"读取标签文件 {file_path} 时出错: {e}")
    return None

def preprocess_and_label_data(folder_path,data_file,label_file,win_len):
  dataset = []
  labels = []
  data_path = os.path.join(folder_path,data_file)
  label_path = os.path.join(folder_path,label_file)
  dataset = read_file(data_path)
  labels = read_file(label_path)

  dataset,labels = split_sequence(dataset,labels,win_len=win_len)
  
  return dataset,labels

def split_sequence(dataset,labels,win_len):
    new_dataset = []
    new_labels = []
    for j in range( len(dataset) - win_len + 1 ):
      # 设置win_len为45，则dataset里的元素按照0--44，这样排列
      start_index = j
      end_index = start_index + win_len
      label_index = int(j + win_len//2 )
      new_dataset.append(dataset[start_index:end_index])
      new_labels.append(labels[label_index])
    return np.array(new_dataset), np.array(new_labels)

# Training parameters
batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.00

clip = -1  # -1表示不剪辑梯度
num_epochs = 10
log_interval = 100
seq_length = 255
output_size = 1
kernel_size = 16

input_channels = 1
channel_sizes = [32]*4
hidden_size = 128
MLP_fir = hidden_size * 2 * seq_length# 128 * 2
MLP_layer = decreasing_list_by_division(MLP_fir)

# Import model
model = BiTCN(input_channels,output_size,channel_sizes,kernel_size,seq_len=seq_length,dropout=dropout)
model.to(device)

# Import optimizer
learning_rate = 3e-4
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
  model.parameters(),
  lr = learning_rate
)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Import data
# r前缀表示原始字符串（raw string）。原始字符串中的反斜杠字符 () 不会被解释为转义字符，而是作为普通字符对待。
data_dir = r'./dataset'
data_file = 'OSC_sync_471.txt'
label_file = 'data64QAM.txt'
dataset, labels = preprocess_and_label_data(data_dir,data_file,label_file,seq_length)
print(dataset.shape)
print(labels.shape)

X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.4, random_state=42)
#torch.long可以使其进入crossentropy
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

def train(model, device,train_loader, optimizer, criterion):
  model.train()
  total_loss = 0
  for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
    optimizer.zero_grad()
    output = model(X_batch.to(device))
    loss = criterion(output, y_batch.to(device))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  average_loss = total_loss / len(train_loader)
  return average_loss

def validate(model, device, val_loader, criterion):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for X_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False):
      output = model(X_batch.to(device))
      loss = criterion(output, y_batch.to(device))
      total_loss += loss.item()
  average_loss = total_loss / len(val_loader)
  return average_loss

def train():
  # 训练模型
  train_losses = []
  val_losses = []

  best_val_loss = float('inf')
  for epoch in range(num_epochs):
    scheduler.step()
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, device, train_loader, optimizer, criterion)
    val_loss = validate(model, device, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), 'BiTCN_best_model.pth')
      logger.info(f"Saved model with validation loss: {best_val_loss:.4f}")
  
def run():
  model = BiTCN(input_channels,output_size,channel_sizes,kernel_size,seq_len=seq_length,dropout=dropout)
  model.load_state_dict(torch.load("BiTCN_best_model.pth", map_location=torch.device('cpu')))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  # dataset, labels = preprocess_and_label_data(data_dir,data_file,label_file,seq_length)
  X_train = torch.tensor(dataset, dtype=torch.float32)
  y_train = torch.tensor(labels, dtype=torch.float32)
  batch_size = 512
  train_data = TensorDataset(X_train, y_train)
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
  predictions = []
  
  def get_txt(model, device, data_loader):
    model.eval()
    output=[]
    with torch.no_grad():
      import time
      i = 0
      for X_batch, y_batch in tqdm(data_loader, desc="Validation", leave=False):
        start_time = time.time()  # 记录开始时间
        out = model(X_batch.to(device))
        end_time = time.time()  # 记录结束时间
        inference_time = end_time - start_time  # 计算推理时间
        print(f"Inference Time: {inference_time} seconds")
        output.append(out.detach().cpu())
        outputs = torch.cat(output, dim=0)
        outputs_np = outputs.numpy()
        print(len(outputs))
        # i = i + 1
        # if i == 10:
        #   break
    return outputs_np.flatten()
  predictions = get_txt(model,device,train_loader)

  half_window = (seq_length - 1) // 2
  with open(r'./dataset/data64QAM.txt', 'r') as file:
    original_data = np.array([float(line.strip()) for line in file.readlines()])
  prefix = original_data[:half_window]
  suffix = original_data[-half_window:]
  final_predictions = np.concatenate([prefix, predictions, suffix])
  final_predictions = final_predictions.reshape(-1,1)
  # 保存补齐后的数据
  final_predictions_file = 'new_BiTCN_final_predictions_471.txt'
  np.savetxt(final_predictions_file, final_predictions, fmt='%f')
  print(f"Final predictions with padding saved to {final_predictions_file}")

def main():
  run()

if __name__ == "__main__":
  main()