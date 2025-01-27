import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_iris

iris = load_iris()

x = iris["data"]    # ndarray型
t = iris["target"]

# tensor型へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)
"""
print(x.shape)    # torch.Size([150, 4])_150行あって、アヤメの花の種類を決定する入力値が4つある
print(t.shape)    # torch.Size([150])_150行
"""
# 入力値と目標値をまとめる
dataset = torch.utils.data.TensorDataset(x, t)

