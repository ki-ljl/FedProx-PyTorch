# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:22
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import sys

import numpy as np
import pandas as pd
import torch
from args import args_parser

sys.path.append('../')
from torch.utils.data import Dataset, DataLoader

args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


def load_data(file_name):
    df = pd.read_csv('data/Wind/Task 1/Task1_W_Zone1_10/' + file_name + '.csv', encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B):
    # print('data processing...')
    data = load_data(file_name)
    columns = data.columns
    wind = data[columns[2]]
    wind = wind.tolist()
    data = data.values.tolist()
    X, Y = [], []
    seq = []
    for i in range(len(data) - 30):
        train_seq = []
        train_label = []
        for j in range(i, i + 24):
            train_seq.append(wind[j])

        for c in range(3, 7):
            train_seq.append(data[i + 24][c])
        train_label.append(wind[i + 24])
        train_seq = torch.FloatTensor(train_seq).view(-1)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))

    Dtr = seq[0:int(len(seq) * 0.8)]
    Dte = seq[int(len(seq) * 0.8):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dte


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))
