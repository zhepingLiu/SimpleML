#!/usr/bin/env python
# coding: utf-8

# 神经网络训练pricePrediction
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

import data_preprocess
import utils
from models import MyNN1, MyNN2


# 训练网络
def train(network, total_iterations, train_loader, lr, device):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    cost = torch.nn.MSELoss(reduction='mean')
    cost.to(device)

    for i in range(1, total_iterations + 1):
        batch_loss = []
        for x, y in train_loader:
            xx = torch.tensor(x, dtype=torch.float, requires_grad=True)
            yy = torch.tensor(y, dtype=torch.float, requires_grad=True)
            xx.to(device)
            yy.to(device)

            prediction = network(xx)
            prediction.to(device)

            loss = cost(prediction.squeeze(), yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        if i % 1000 == 0:
            print("Iterations: " + str(i) + ", Loss value: " + str(np.sqrt(np.mean(batch_loss))))

    return network


def main(SAVE_URL=None, TRAIN_DATA_URL=None, batch_size=4, bucket=None,
         database=None, hidden_size=64, lr=0.01,
         mse_table=None, on_td=True, pred_all_table=None,
         pred_val_table=None, region=None,
         source_table=None, standardize=False, total_iterations=10000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据集
    train_data_raw = pd.read_csv(TRAIN_DATA_URL)

    # 定义你数据集的label_name, 也就是你要预测的那个column的名字
    label_name = ''

    # train_data_raw.to_csv(SAVE_URL+'debug.csv', index=False, header=True)

    mses = pd.DataFrame(columns=['Category', 'Model No.', 'Validation_log', 'Validation', 'All_log', 'All'])
    results = pd.DataFrame()
    results_all = pd.DataFrame()

    train_dataset, test_dataset, all_dataset = data_preprocess.preProcess(
        train_data_raw, label_name, standardize)

    # 设定参数, 创建网络
    input_shape = train_dataset.getShape()
    output_shape = 1

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # 决定要用来训练的模型
    models = []
    my_nn1 = MyNN1(input_shape, output_shape, hidden_size, bias=True)
    my_nn2 = MyNN2(input_shape, output_shape, hidden_size, bias=True)
    models.append(my_nn1)
    models.append(my_nn2)

    for i in range(len(models)):
        models[i].train()  # switch to training mode
        models[i] = train(models[i], total_iterations, train_loader, lr, device)
        models[i].eval()  # switch to evaluation mode (will ignore BN layer)
        # results[f"{current}_actual"] = pd.Series(yt)
        # results_all[f"{current}_actual"] = pd.Series(y_all)
        results[f'model{i}_actual'] = pd.Series(test_dataset.y)
        results_all[f'model{i}_actual'] = pd.Series(all_dataset.y)
        with torch.no_grad():
            # results[f"{current}"] = pd.Series(models[i](xt).data.numpy().flatten())
            # results_all[f"{current}"] = pd.Series(models[i](X_all).data.numpy().flatten())
            # torch.save(models[i].state_dict(), SAVE_URL + f"{current}_model{i}.pth")
            # if on_td: utils.upload_data(s3_client, bucket, SAVE_URL + f"{current}_model{i}.pth")
            results[f'model{i}_pred'] = pd.Series(models[i](test_dataset.x).data.numpy().flatten())
            results_all[f'model{i}_pred'] = pd.Series(models[i](all_dataset.x).data.numpy().flatten())
            torch.save(models[i].state_dict(), SAVE_URL + f"model{i}.pth")
            print("MSE of validation set:")
            logMse = utils.logMSE(models[i], test_dataset.x, test_dataset.y)
            print(logMse)
            mse = utils.mse(models[i], test_dataset.x, test_dataset.y)
            print(mse)
            # mses[f"{current}_model{i}_validation"] = [logMse, mse]
            print("MSE of all dataset:")
            logMse_all = utils.logMSE(models[i], all_dataset.x, all_dataset.y)
            print(logMse_all)
            mse_all = utils.mse(models[i], all_dataset.x, all_dataset.y)
            print(mse_all)

        df = pd.DataFrame({
            # 'Category': f'{current}',
            'Model No.': [i],
            'Validation_log': [logMse],
            'Validation': [mse],
            'All_log': [logMse_all],
            'All': [mse_all]
        })
        mses = mses.append(df)

        print("-----------------Complete------------------")
        if not on_td:
            # utils.plotAll(results[f'{current}'], yt, SAVE_URL + f"{current}_validation")
            # utils.plotAll(results_all[f'{current}'], y_all, SAVE_URL + f"{current}_all")
            utils.plotAll(results[f'model{i}_pred'], results[f'model{i}_actual'], SAVE_URL + f"model{i}_validation")
            utils.plotAll(results_all[f'model{i}_pred'], results_all[f'model{i}_actual'], SAVE_URL + f"model{i}_all")

        # save results as csv
        # if not on_td:
        #     results.to_csv(SAVE_URL+f'{current}_results_validation.csv', index=False, header=True)
        #     results_all.to_csv(SAVE_URL+f'{current}_results_all.csv', index=False, header=True)

    # save mses as csv
    mses.to_csv(SAVE_URL + 'mses.csv', index=False, header=True)
    results.to_csv(SAVE_URL + 'results_validation.csv', index=False, header=True)
    results_all.to_csv(SAVE_URL + 'results_all.csv', index=False, header=True)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--SAVE_URL', type=str, default='../../saved_models/')
    parser.add_argument('--TRAIN_DATA_URL', type=str,
                        default='../../data/data.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--standardize', action='store_true', default=False)
    parser.add_argument('--total_iterations', type=int, default=10000)

    SAVE_URL = vars(parser.parse_args())['SAVE_URL']
    TRAIN_DATA_URL = vars(parser.parse_args())['TRAIN_DATA_URL']
    batch_size = vars(parser.parse_args())['batch_size']
    hidden_size = vars(parser.parse_args())['hidden_size']
    lr = vars(parser.parse_args())['lr']
    standardize = vars(parser.parse_args())['standardize']
    total_iterations = vars(parser.parse_args())['total_iterations']

    # automatically concat the SAVE_URL
    SAVE_URL = f'{SAVE_URL}'

    time = datetime.now().strftime("%Y-%m-%d")
    SAVE_URL = SAVE_URL + '_' + time + '/'
    save_path = Path(SAVE_URL)
    if not save_path.exists():
        os.mkdir(SAVE_URL)

    models = main(SAVE_URL=SAVE_URL, TRAIN_DATA_URL=TRAIN_DATA_URL,
                  batch_size=batch_size, hidden_size=hidden_size,
                  lr=lr, standardize=standardize, 
                  total_iterations=total_iterations)
