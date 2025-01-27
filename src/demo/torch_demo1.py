import torch
import torch.nn as nn
import torch.nn.functional as F

# 3ノード → 2ノード
torch.manual_seed(0)
fc = nn.Linear(3, 2)

# tensor方への変更
x = torch.tensor([[1., 2., 3.,]])

# 線形変換の計算  xと重みを掛けて、、、、、、っていう計算
u = fc(x)
# ReLU関数
h = F.relu(u)
t = torch.tensor([[1.], [3.]])      # 目標値
y = torch.tensor([[2.], [4.]])      # 予測値
print(F.mse_loss(y, t))             # 平均2乗誤差
