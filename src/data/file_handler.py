__author__ = "JUN WEI WANG"
__email__ = "wjw_03@outlook.com"

import threading
from os import walk
from re import Pattern

def read_file(file_path: str):
  with open(file_path, 'r') as file:
    content = file.read()
  return content

def read_files_threaded(file_paths: list[str]):
  threads = []
  results = {}

  # Define the worker function
  def read_file_thread(file_path):
    result = read_file(file_path)
    results[file_path] = result

  # Create and start threads
  for file_path in file_paths:
    thread = threading.Thread(target=read_file_thread, args=("./dataset/" + file_path,))
    threads.append(thread)
    thread.start()

  # Wait for all threads to finish
  for thread in threads:
    thread.join()

  return results

def filter_filenames(filenames: list[str], regex: Pattern) -> list[str]:
  return [ i for i in filenames if regex.match(i) ]

def demo():
  from src.config import OUTPUT_REG
  from src.data.processing import parse_str
  
  filenames = next(walk("./dataset"), (None, None, []))[2]  # extract all files from dataset folder
  # file_paths = ["./dataset/data64QAM.txt", "./dataset/OSC_sync_291.txt", "./dataset/OSC_sync_292.txt", "./dataset/OSC_sync_293.txt"]
  filenames = filter_filenames(filenames, OUTPUT_REG) # filter the files, we only want output signals
  filenames.sort()
  
  results = read_files_threaded(filenames)
  separator_size = 50
  for file_path, content in results.items():
    print(f"Reading {file_path}:")
    parse_str(content)
    # print(content)
    # print("-" * separator_size)

if __name__ == "__main__":
  demo()