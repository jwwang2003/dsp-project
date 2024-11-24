import os
from scipy.io import loadmat
import numpy as np

def read_txt_data(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # line.strip()是一个字符串方法，它用于去除字符串首尾的空白字符
                line_data = line.strip().split()
                data.append([float(x) for x in line_data])
        return data
    except Exception as e:
        print(f"读取数据文件 {file_path} 时出错: {e}")
        return None

def read_txt_labels(file_path):
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
    dataset = read_txt_data(data_path)
    labels = read_txt_labels(label_path)

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




