import time
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from src.data_pre import preprocess_and_label_data
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

dropout = 0.1

clip = -1  # -1表示不剪辑梯度
num_epochs = 10
log_interval = 100
seq_length = 255
output_size = 1

input_channels = 1
hidden_size = 128  # For the feedforward network in the transformer
### Define the TimeSeriesTransformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # Dividing by 2 because we use half dimensions for sin, half for cos
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_heads, hidden_size, dropout, output_size):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.model_type = 'Transformer'
        self.src_mask = None

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        src = self.embedding(src)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.mean(dim=0)  # (batch_size, d_model)
        output = self.decoder(output)
        return output

# Model parameters
input_size = input_channels  # Since input_channels=1
d_model = 32  # Embedding dimension
num_layers = 4
num_heads = 4  # d_model must be divisible by num_heads
dropout = 0.1
output_size = 1  # Regression output

model = TimeSeriesTransformer(input_size, d_model, num_layers, num_heads, hidden_size, dropout, output_size)
model.to(device)

### Define optimizer and loss function
learning_rate = 3e-4
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

### Import data
# r前缀表示原始字符串（raw string）。原始字符串中的反斜杠字符 () 不会被解释为转义字符，而是作为普通字符对待。
data_dir = r'./dataset'
data_file = 'OSC_sync_471.txt'
label_file = 'data64QAM.txt'
dataset, labels = preprocess_and_label_data(data_dir,data_file,label_file,seq_length)
print(dataset.shape)
print(labels.shape)


X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.4, random_state=42)
# torch.long可以使其进入crossentropy
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
        loss = criterion(output.squeeze(), y_batch.to(device))
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
            loss = criterion(output.squeeze(), y_batch.to(device))
            total_loss += loss.item()
    average_loss = total_loss / len(val_loader)
    return average_loss

# Train the model
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
        torch.save(model.state_dict(), 'TimeSeriesTransformer_best_model.pth')
        logger.info(f"Saved model with validation loss: {best_val_loss:.4f}")

"""
# Plot loss curves (optional)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
"""

# Load the best model for inference
model = TimeSeriesTransformer(input_size, d_model, num_layers, num_heads, hidden_size, dropout, output_size)
model.load_state_dict(torch.load("TimeSeriesTransformer_best_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare data for inference
X_train = torch.tensor(dataset, dtype=torch.float32)
y_train = torch.tensor(labels, dtype=torch.float32)
batch_size = 256
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

def get_txt(model, device, data_loader):
    model.eval()
    output=[]
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Inference", leave=False):
            out = model(X_batch.to(device))
            output.append(out.detach().cpu())
        outputs = torch.cat(output, dim=0)
        outputs_np = outputs.numpy()
    return outputs_np.flatten()

predictions = get_txt(model,device,train_loader)

half_window = (seq_length - 1) // 2
with open(r'./dataset/data64QAM.txt', 'r') as file:
    original_data = np.array([float(line.strip()) for line in file.readlines()])
prefix = original_data[:half_window]
suffix = original_data[-half_window:]
final_predictions = np.concatenate([prefix, predictions, suffix])
final_predictions = final_predictions.reshape(-1,1)
# Save the padded predictions
final_predictions_file = 'new_TimeSeriesTransformer_final_predictions_471.txt'
np.savetxt(final_predictions_file, final_predictions, fmt='%f')
print(f"Final predictions with padding saved to {final_predictions_file}")
