import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class for a 1D vector of floats
class CustomDataset_1D(Dataset):
  def __init__(self, data: list[float], labels: list[float]):
    """
    Args:
      data (list or numpy array): A 1D array of numbers.
    """
    assert len(data) == len(labels)
    
    # if len(data) != len(labels):
    #   raise ValueError("Length of dataset does not match the length of labels")

    # Ensure data is a tensor
    self.data = torch.tensor(data, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.float32)
    
  def __len__(self):
    # Return the total number of samples
    return len(self.data)   # data and labels are of the same length
  
  def __getitem__(self, idx):
    # Return a sample from the dataset
    return self.data[idx], self.labels[idx]

def demo():
  # Example 1D array of numbers
  array_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

  # Create the dataset
  dataset = CustomDataset_1D(array_data, array_data)

  # Create DataLoader
  dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

  # Iterate over the DataLoader
  for batch in dataloader:
    print(batch)
  
  from src.data.file_handler import read_file
  from src.data.processing import parse_str
  
  array_data = parse_str(read_file("./dataset/OSC_sync_336.txt"))
  array_labels = parse_str(read_file("./dataset/data64QAM.txt"))
  dataset = CustomDataset_1D(array_data, array_labels)
  dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
  
  for data, label in dataloader:
    # for data, label in batch:
    print(data, label)


if __name__ == "__main__":
  demo()

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