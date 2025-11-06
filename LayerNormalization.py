import torch
import torch.nn as nn   

# 实现一个基本的层归一化层，用于归一化输入张量的每个样本的特征维度
class LayerNormalization(nn.Module):
    def __init__(self, feature_dim, eps=1e-6): # feature_dim 是输入张量的特征维度，eps 是一个小的常量，用于防止除零错误，
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_dim)) # 缩放因子，scale，用于缩放归一化后的张量，默认初始化为1，没有缩放
        self.beta = nn.Parameter(torch.zeros(feature_dim)) # 偏移因子，shfit，用于平移归一化后的张量，默认初始化为0，没有偏移

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 基本的层归一化公式：(x - mean) / (std + eps) * gamma + beta，
        # 其中 mean 是输入张量的特征维度上的均值，std 是输入张量的特征维度上的标准差，eps 是一个小的常量，用于防止除零错误，
        # gamma 是缩放因子，用于缩放归一化后的张量，默认初始化为1，没有缩放，
        # beta 是偏移因子，用于平移归一化后的张量，默认初始化为0，没有偏移。
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

# 测试
if __name__ == '__main__':
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [2.0, 3.0, 4.0, 5.0, 6.0],
                        [3.0, 4.0, 5.0, 6.0, 7.0]])
    print(input)

    layer = LayerNormalization(input.size(-1))
    output = layer(input)
    print(output)