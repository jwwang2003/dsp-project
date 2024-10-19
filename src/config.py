from os import path
import re

# File handling helpers & constants

BASE = "./"

DATASET_DIR = path.join(BASE, "dataset")
INPUT_REG = re.compile("data64QAM.txt")
OUTPUT_REG = re.compile("OSC_sync_([0-9]*).txt")

# NN constants
