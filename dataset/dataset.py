#-*- coding:utf-8 _*-
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def normalization(data):
    length, height, width = data.shape
    normalized_data = np.zeros((length, height, width), dtype=float)
    min = data.min()
    max = data.max()
    for i in range(length):
        for j in range(height):
            for k in range(width):
                normalized_data[i][j][k] = (data[i][j][k] - min)/(max - min)
    return normalized_data

def make_rolling_Data_monthly(time_interval,is_train=True):
    # load data
    CMIP_data = torch.load('C:\\Users\\123\\Desktop\\github\\PycharmProjects\\ENSO_zxy\\DataProcess\\CMIP6_SST_165301-201812_4392.pt')
    CMIP_data = np.array(CMIP_data, dtype=float)
    print("Shape of origin Dataset: ", CMIP_data.shape)

    CMIP_data = normalization(CMIP_data)

    data_X = np.lib.stride_tricks.sliding_window_view(CMIP_data,(12,20,50)).reshape(-1,12,20,50,1)
    data = CMIP_data[time_interval+1:]
    data_Y = np.lib.stride_tricks.sliding_window_view(data,(12,20,50)).reshape(-1,12,20,50,1)

    train_X,test_X,train_Y,test_Y = train_test_split(data_X[:data_Y.shape[0]],data_Y,test_size=0.2)
    print('train_X shape:',train_X.shape,'train_Y shape:',train_Y.shape,
            'test_X shape:',test_X.shape,'test_Y shape:',test_Y.shape)

    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y)
    test_X = torch.from_numpy(test_X)
    test_Y = torch.from_numpy(test_Y)
    train_X = train_X.to(torch.float32)
    train_Y = train_Y.to(torch.float32)
    test_X = test_X.to(torch.float32)
    test_Y = test_Y.to(torch.float32)
    if is_train:
        return train_X, train_Y
    else:
        return test_X, test_Y

class ENSODataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, idx):
        input = self.data_tensor[-1].permute(0, 3, 1, 2)  # [len,12,20,50,1]
        output = self.target_tensor[-1].permute(0, 3, 1, 2) # [len,12,20,50,1]
        out = [idx, input, output]
        return out

    def __len__(self):
        return self.data_tensor.size(0)


def inverse_normalization(data):
    input_length, map_height, map_width = data.shape
    min = data.min()
    max = data.max()
    inverse_data = np.zeros((input_length, map_height, map_width), dtype=float)
    for i in range(input_length):
        for j in range(map_height):
            for k in range(map_width):
                inverse_data[i][j][k] = data[i][j][k]*(max - min) + min
    return inverse_data





