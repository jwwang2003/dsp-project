import torch
from typing import Tuple
from os import path
from torch.utils.data import Dataset
import numpy as np

# Local
from data.processing import parse_str
from data.file_handler import read_file

class CustomDataset(Dataset):
    def __init__(
            self,
            folder_path: str,
            data_file: str,
            label_file: str,
            win_len: int=1
          ):
        """
        Args:
            folder_path (str): Path to the folder containing data and label files.
            data_file (str): File name of the data file.
            label_file (str): File name of the label file.
            win_len (int): Window length for sequence splitting.
        """
        self.dataset, self.labels = self.preprocess_and_label_data(
            folder_path, data_file, label_file, win_len
        )
        
        # Convert to tensor
        self.dataset = torch.tensor(self.dataset, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
    
    @staticmethod
    def split_sequence(
        dataset: list[list[float]], labels: list[float], win_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the dataset and labels into sequences of length `win_len`.

        Args:
            dataset (list[list[float]]): The input dataset as a list of lists, where each sublist represents a data row.
            labels (list[float]): The corresponding labels as a list of floats.
            win_len (int): The length of the sliding window to use for splitting sequences.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - `new_dataset` (np.ndarray): The resulting dataset split into sequences, shape (N, win_len, features).
                - `new_labels` (np.ndarray): The corresponding labels for the sequences, shape (N,).
        
        Raises:
            ValueError: If the dataset is shorter than the window length.
        """
        if len(dataset) < win_len:
            raise ValueError("Dataset length must be greater than or equal to the window length.")

        new_dataset = []
        new_labels = []

        # Sliding window implementation
        for j in range(len(dataset) - win_len + 1):
            start_index = j
            end_index = start_index + win_len
            # Label is the midpoint of the window
            label_index = int(j + win_len // 2)
            new_dataset.append(dataset[start_index:end_index])
            new_labels.append(labels[label_index])
          
        return np.array(new_dataset), np.array(new_labels)

    def preprocess_and_label_data(
        self, folder_path: str, data_file: str, label_file: str, win_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses the data and labels by reading from files and applying sequence splitting.

        Args:
            folder_path (str): Path to the folder containing the data and label files.
            data_file (str): The name of the data file to load.
            label_file (str): The name of the label file to load.
            win_len (int): The length of the sliding window for splitting sequences.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - `dataset` (np.ndarray): The processed dataset split into sequences, shape (N, win_len, features).
                - `labels` (np.ndarray): The corresponding labels for the sequences, shape (N,).
        
        Raises:
            ValueError: If the dataset or labels cannot be loaded from the files.
        """
        data_path = path.join(folder_path, data_file)
        label_path = path.join(folder_path, label_file)

        dataset = parse_str(read_file(data_path))
        labels = parse_str(read_file(label_path))

        if dataset is None or labels is None:
            raise ValueError("Failed to load data or labels.")

        return self.split_sequence(dataset, labels, win_len)

# # Custom Dataset class for a 1D vector of floats
# class CustomDataset_1D(Dataset):
#     def __init__(
#         self, 
#         data: list[float], 
#         labels: list[float],
#         win_len: int = 1,
#         train: bool = True, 
#         train_ratio: float = 0.9
#     ):
#         """
#         Args:
#           data (list or numpy array): A 1D array of numbers.
#           labels (list or numpy array): A 1D array of numbers (same length as data).
#           win_len (int): Length of the sliding window.
#         """
#         print(
#             f"[{CustomDataset_1D.__name__}] Data Length: {len(data)} | Label Length: {len(labels)} | Window Length: {win_len} | Ratio: {train_ratio}"
#         )
#         assert len(data) == len(labels)
#         assert len(data) >= win_len  # Ensure that the input data is at least win_len
#         assert 0 <= train_ratio <= 1

#         test_ratio = 1 - train_ratio

#         # Split data into train and test sets
#         train_datas, test_datas, train_labels, test_labels = train_test_split(
#             data, labels,
#             train_size=train_ratio,
#             random_state=42
#         )
    
#         # Assign train or test data
#         if train:
#             self.data, self.labels = (
#                 torch.tensor(train_datas, dtype=torch.float32), 
#                 torch.tensor(train_labels, dtype=torch.float32)
#             )
#         else:
#             self.data, self.labels = (
#                 torch.tensor(test_datas, dtype=torch.float32), 
#                 torch.tensor(test_labels, dtype=torch.float32)
#             )

#         self.win_len = win_len
        
#     def __len__(self):
#         # Dataset length adjusted for sliding window
#         return len(self.data) - self.win_len + 1
    
#     def __getitem__(self, idx):
#         # Sliding window implementation
#         start_idx = idx
#         end_idx = start_idx + self.win_len

#         data_window = self.data[start_idx:end_idx]

#         # Add a singleton dimension to the data window
#         data_window = data_window.unsqueeze(-1)

#         # Label is the midpoint of the window
#         label_idx = int(start_idx + self.win_len // 2)
#         label = self.labels[label_idx]

#         return data_window, label

# def demo():
#   # Example 1D array of numbers
#   array_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#   # Create the dataset
#   dataset = CustomDataset_1D(array_data, array_data, 1)   # Setting win_len = 1, custom dataset acts normally

#   # Create DataLoader
#   dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#   # Iterate over the DataLoader
#   for batch in dataloader:
#     print(batch)
  
#   from src.data.file_handler import read_file
#   from src.data.processing import parse_str
  
#   array_data = parse_str(read_file("./dataset/OSC_sync_336.txt"))
#   array_labels = parse_str(read_file("./dataset/data64QAM.txt"))
#   dataset = CustomDataset_1D(array_data, array_labels, 255)
#   dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

#   itr = 0
#   for data, label in dataloader:
#     if itr > 2:
#       break
    
#     # for data, label in batch:
#     print(f"Data Shape: {data.shape} Label Shape: {label.shape}")
#     # print(data, label)
#     itr += 1

# if __name__ == "__main__":
#   # Run the demo
#   demo()

# Reference from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# import os
# import pandas as pd
# from torchvision.io import read_image

# class CustomImageDataset(Dataset):
#   def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#     self.img_labels = pd.read_csv(annotations_file)
#     self.img_dir = img_dir
#     self.transform = transform
#     self.target_transform = target_transform

#   def __len__(self):
#     return len(self.img_labels)

#   def __getitem__(self, idx):
#     img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#     image = read_image(img_path)
#     label = self.img_labels.iloc[idx, 1]
#     if self.transform:
#       image = self.transform(image)
#     if self.target_transform:
#       label = self.target_transform(label)
#     return image, label