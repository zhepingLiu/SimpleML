import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class CSVDataset(Dataset):
    def __init__(self, x, y):
        # 将数据转换成torch.tensor格式
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)

    def getShape(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

'''
在这个function里面做需要做的数据处理
'''
def preProcess(train_data, label_name, standardize):

    redundant_columns = []

    # drop空值
    train_data.dropna(inplace=True)
    # drop掉不需要的列(feature)
    train_data = train_data.drop(redundant_columns, axis=1)
    
    # 定义你的label feature
    y = np.array(train_data[f'{label_name}'])
    train_data = train_data.drop([f'{label_name}'], axis=1)

    # one-hot, 把catagorical data转换成0和1
    train_data = pd.get_dummies(train_data)

    # order_list = []
    # # re-order the train_data columns
    # df = df[order_list]

    x1 = np.array(train_data, dtype=float)

    # 标准化预处理
    if standardize:
        X = preprocessing.StandardScaler().fit_transform(x1)
    else:
        X = x1 # without preprocessing

    # 分割训练数据集和测试数据集, 测试比例(test_size)为20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)
    all_dataset = CSVDataset(X, y)
    
    return train_dataset, test_dataset, all_dataset
