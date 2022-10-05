# 安装环境
## 安装Anaconda
## 新建一个conda环境
- `conda create -n YOUR_ENVIRONMENT_NAME python=3.7`
## 之后使用的时候需要激活这个环境
- `conda activate YOUR_ENVIRONMENT_NAME

# 安装packages
## 在创建好的conda环境下执行
- `pip install -r requirements.txt`
- 或者 `conda install --file requirements.txt`

# 运行训练程序
- python core/train.py
## 可用的参数
-   `--TRAIN_DATA_URL: the url to the training data csv file`
    `--SAVE_URL: the saving directory for the model and plots`
    `--batch_size: batch size for neural network training`
    `--total_iterations: total iterations for training`
    `--lr: learning rate for training`
    `--hidden_size: hidden nodes size for the nueral network`
    `--standardize: whether to standardize the training data`