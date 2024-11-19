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
        self.dataset, self.labels, self.size = self.preprocess_and_label_data(
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
    ) -> Tuple[np.ndarray, np.ndarray, int]:
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

        # Ensure that the lengths of the dataset and their respective labels have the same length
        assert len(dataset) == len(labels)

        if dataset is None or labels is None:
            raise ValueError("Failed to load data or labels.")

        return [*self.split_sequence(dataset, labels, win_len), len(dataset)]
