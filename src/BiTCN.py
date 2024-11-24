import time
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import model
from model import BiGRU,BiTCN,MLP,decreasing_list_by_division,plot_weight_distribution
from torch.optim.lr_scheduler import StepLR
from data_pre import preprocess_and_label_data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
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
warnings.filterwarnings("ignore")
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
# channel_sizes = [8,16,24,32,40,48,56,64]
channel_sizes = [32]*4
hidden_size = 128
MLP_fir = hidden_size * 2 * seq_length# 128 * 2
MLP_layer = decreasing_list_by_division(MLP_fir)
### import model
model = BiTCN(input_channels,output_size,channel_sizes,kernel_size,seq_len=seq_length,dropout=dropout)
model.to(device)
### import optimizer
learning_rate = 3e-4
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

###import data
# r前缀表示原始字符串（raw string）。原始字符串中的反斜杠字符 () 不会被解释为转义字符，而是作为普通字符对待。
data_dir = r'D:\05_code\expLD0711 (2)\expLD0711'
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
"""
# 绘制损失曲线
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
"""


"""
data_dir = r'D:\03.Data\rxdata\rxdata'
data_file = 'OSC_sync_326.txt'
label_file = 'Superposed_original.txt'
"""
model = BiTCN(input_channels,output_size,channel_sizes,kernel_size,seq_len=seq_length,dropout=dropout)
model.load_state_dict(torch.load("BiTCN_best_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# dataset, labels = preprocess_and_label_data(data_dir,data_file,label_file,seq_length)
X_train = torch.tensor(dataset, dtype=torch.float32)
y_train = torch.tensor(labels, dtype=torch.float32)
batch_size = 256
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
predictions = []
def get_txt(model, device, data_loader):
    model.eval()
    output=[]
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Validation", leave=False):
            #start_time = time.time()  # 记录开始时间
            out = model(X_batch.to(device))
            #end_time = time.time()  # 记录结束时间
            #inference_time = end_time - start_time  # 计算推理时间
            #print(f"Inference Time: {inference_time} seconds")
            output.append(out.detach().cpu())
            outputs = torch.cat(output, dim=0)
            outputs_np = outputs.numpy()
    return outputs_np.flatten()
predictions = get_txt(model,device,train_loader)

half_window = (seq_length - 1) // 2
with open(r'D:\05_code\expLD0711 (2)\expLD0711\data64QAM.txt', 'r') as file:
    original_data = np.array([float(line.strip()) for line in file.readlines()])
prefix = original_data[:half_window]
suffix = original_data[-half_window:]
final_predictions = np.concatenate([prefix, predictions, suffix])
final_predictions = final_predictions.reshape(-1,1)
# 保存补齐后的数据
final_predictions_file = 'new_BiTCN_final_predictions_471.txt'
np.savetxt(final_predictions_file, final_predictions, fmt='%f')
print(f"Final predictions with padding saved to {final_predictions_file}")










"""

### 直接全部并行计算，但是GPU可能一下不能并行计算这么多数据
model.load_state_dict(torch.load('BiTCN_best_model.pth'))
model.to(device)
model.eval()
dataset_tensor = torch.tensor(dataset, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = model(dataset_tensor)
predictions = predictions.cpu().numpy()
predictions_file = 'predictions.txt'
np.savetxt(predictions_file, predictions, fmt='%f')
print(f"Predictions saved to {predictions_file}")

"""