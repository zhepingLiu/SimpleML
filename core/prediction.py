import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

import data_preprocess
import utils
from models import MyNN1, MyNN2

def main(DATA_URL, MODEL_URL, SAVE_URL, bucket, hidden_size):
    # 1.读取所有数据
    predict_data_raw = pd.read_csv(DATA_URL)

    # 2.转换为nparray
    predict_data_prime = np.array(predict_data_raw, dtype=float)
    # 3.转换为tensor
    x = torch.tensor(predict_data_prime, dtype=torch.float)

    # 4.定义模型参数
    input_shape = x.shape[1]
    output_shape = 1
    my_nn1 = MyNN1(input_shape, output_shape, hidden_size, bias=True)
    my_nn2 = MyNN2(input_shape, output_shape, hidden_size, bias=True)

    # 5.分别读取最新的模型
    my_nn1.load_state_dict(torch.load(MODEL_URL+'_model0.pth'), strict=True)
    my_nn1.eval()
    my_nn2.load_state_dict(torch.load(MODEL_URL+'_model1.pth'), strict=True)
    my_nn2.eval()

    # 6.用模型预测
    with torch.no_grad():
        predict1 = my_nn1(x).data.numpy().flatten()
        predict2 = my_nn2(x).data.numpy().flatten()
        print(predict1)
        print(predict2)

    # 7.储存结果到csv/target_table
    # predicts.to_csv(SAVE_URL+'predict_results.csv', index=False, header=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--DATA_URL', type=str, default='../../../Learning/yephome-data/Domain_Aggregated0822.csv')
    parser.add_argument('--MODEL_URL', type=str, default='../../saved_models/4pastMonths_noLag1_2022-08-30/')
    parser.add_argument('--SAVE_URL', type=str, default='../../prediction_results/')
    parser.add_argument('--hidden_size', type=int, default=128)

    DATA_URL = vars(parser.parse_args())['DATA_URL']
    MODEL_URL = vars(parser.parse_args())['MODEL_URL']
    SAVE_URL = vars(parser.parse_args())['SAVE_URL']
    hidden_size = vars(parser.parse_args())['hidden_size']

    # automatically concat the SAVE_URL
    SAVE_URL = f'{SAVE_URL}'

    time = datetime.now().strftime("%Y-%m-%d")
    SAVE_URL = SAVE_URL + '_' + time + '/'
    save_path = Path(SAVE_URL)
    if not save_path.exists():
        os.mkdir(SAVE_URL)

    main(DATA_URL=DATA_URL, MODEL_URL=MODEL_URL,
         SAVE_URL=SAVE_URL, hidden_size=hidden_size)