import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def logMSE(network, xt, yt):
    with torch.no_grad():
        predict = network(xt)
        score = torch.mean((torch.log(predict) - torch.log(yt)) ** 2)
    # torch.log(yt) has some NaN, since yt at some entires are negative
    return np.sqrt(score.data.numpy())

def mse(network, xt, yt):
    with torch.no_grad():
        predict = network(xt)
        score = torch.mean((predict - yt) ** 2)
    return np.sqrt(score.data.numpy())

def plotActualVsPred(predict, actual):
    true_data = pd.DataFrame(data = {'actual': actual})
    test_data = pd.DataFrame(data = {'pred': predict.reshape(-1)})

    plt.figure(figsize=(18, 8))
    plt.plot(true_data['actual'], 'b-', label='actual')
    plt.plot(test_data['pred'], 'r-', label='prediction')
    plt.legend()

    # 给图片添加名字
    plt.xlabel('Index'); plt.ylabel('Median'); plt.title('Actual and Predicted Price')
    
def plotAll(predicts, actual, figure_name):
    # true_data = pd.DataFrame(data = {'actual': actual})
    # predicts['actual'] = actual
    # test_data = {}
    # for k, v in predicts.items():
    #     test_data[k] = pd.DataFrame(data = {k: v.reshape(-1)})

    plt.figure(figsize=(18, 8))
    # plt.plot(true_data['actual'], 'r-', label='actual')
    # for k, v in test_data.items():
    #     plt.plot(v, label=k)
    plt.plot(actual, label='actual')
    plt.plot(predicts, label='predict')
    plt.legend(loc='upper left')

    # 给图片添加名字
    plt.xlabel('Index'); plt.ylabel('MeanPrice'); plt.title('Actual and Predicted Price')
    plt.savefig(f'{figure_name}.svg', format='svg')
    # plt.show()

    plt.close()