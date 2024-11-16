from os import path
import re

# File handling helpers & constants

BASE = "./"

DATASET_DIR = path.join(BASE, "dataset")
LABEL_REG = re.compile("data64QAM.txt")
DATA_REG = re.compile("OSC_sync_([0-9]*).txt")

# NN constants

# Dataset parameters
BATCH_SIZE = 512
WINDOW_LENGTH = 255

# Training parameters
NUM_EPOCHS = 10
DROPOUT = 0.00

# Network parameters
INPUT_CHANNELS = 1    # Note: dimm. is 1 with a length of window length
OUTPUT_CHANNELS = 1

KERNEL_SIZE = 16
CHANNEL_SIZES = [32] * 4
HIDDEN_SIZE = 128

MLP_fir = HIDDEN_SIZE * 2 * WINDOW_LENGTH# 128 * 2
# MLP_layer = decreasing_list_by_division(MLP_fir)
