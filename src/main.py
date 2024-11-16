import numpy as np
import torch
import torch.nn as nn

# Check for CUDA!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently training on: {device}")

################################################################################
###########                   Import Dataset(s)               ##################
################################################################################
from os import walk, path
from src.data.file_handler import filter_filenames, read_file
from src.data.processing import parse_str
from src.config import BASE, DATASET_DIR, DATA_REG, LABEL_REG
from torch.utils.data import DataLoader
from src.data.dataset import CustomDataset_1D

# Get paths
filenames = next(walk("./dataset"), (None, None, []))[2]  # extract all files from dataset folder
# file_paths = ["./dataset/data64QAM.txt", "./dataset/OSC_sync_291.txt", "./dataset/OSC_sync_292.txt", "./dataset/OSC_sync_293.txt"]
input_filenames = filter_filenames(filenames, DATA_REG) # filter the files, we only want output signals
input_filenames.sort()
label_filenames = filter_filenames(filenames, LABEL_REG)
label_filenames.sort()

array_data = []
array_labels = []

# We are specifally training on "OSC_sync_471.txt"
if input_filenames.count("OSC_sync_471.txt"):
  array_data = parse_str(
    read_file(path.join(BASE, DATASET_DIR, "OSC_sync_471.txt"))
  )
else:
  print("OSC_sync_471.txt not found!")
  exit(1)

if label_filenames.count("data64QAM.txt"):
  array_labels = parse_str(
    read_file(path.join(BASE, DATASET_DIR, "data64QAM.txt")
    )
  )
else:
  print("data64QAM.txt not found!")
  exit(1)

dataset = CustomDataset_1D(array_data, array_labels, 255)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

count = 0
flag = False
print(f"Dataloader Iterations: {len(dataloader)}")
for idx, data in enumerate(dataloader):
    datas = data[0]
    labels = data[1]
    if not flag:
      print("Data shape:", datas.shape)
      print("Label shape:", labels.shape)
      flag = True
    count += datas.shape[0]
    # break
print(f"Dataset count: {count}")

################################################################################
###########                   Neural Network                  ##################
################################################################################

def main():
  print("Test")

if __name__ == "__main__":
  main()