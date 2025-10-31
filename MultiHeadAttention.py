
import torch
import torch.nn as nn

import os
from CausalAttention import CausalAttention     

class MultiHeadAttention(nn.Module):
    # d_model 是输入的维度， num_heads 是头的数量， d_head 是每个头的维度
    def __init__(self, d_model, num_heads, d_head, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head

        # 使用 CausalAttention 作为每个头的注意力机制
        # d_head 是每个头的注意力维度, 注意力头的数量就是 num_heads, 实际Transformer 的维度是 d_head * num_heads
        # num_heads个注意力头，每个注意力头的维度是 d_head
        self.heads = nn.ModuleList([CausalAttention(d_model, d_head, dropout) for _ in range(num_heads)])

    def forward(self, input):
        return torch.cat([head(input) for head in self.heads], dim=-1)  # 在最后一个维度上拼接
        #列表推导式，[ 对x 的操作，或者 x 操作别的 for x in 列表 if 条件]
        # 第一部分，对x 的操作，或者 x 操作别的
        # 第二部分，for x in 列表
        # 第三部分，if 条件，可选
        # torch.cat 是把多个张量在指定维度上拼接起来，dim=-1 表示最后一个维度,多个向量合成一个向量

        # 这个写法是在 python 运行时是顺序执行的，不是并行的，如果想并行的话，可以使用 torch.jit.fork 和 torch.jit.wait
        # 但是这样写会比较复杂，而且需要对代码进行修改，不是很方便
        # 另外一种方法是使用 torch.nn.DataParallel 或者 torch.nn.parallel.DistributedDataParallel
        # 但是这些方法也需要对代码进行修改，而且需要对模型进行包装，不是很方便
        # 所以这里先不考虑并行的问题，先实现一个简单的版本
        


# 对 MultiHeadAttention 进行测试
if __name__ == "__main__":
    batch_size = 2
    seq_length = 4
    d_model = 16
    num_heads = 4
    d_head = 8  # 每个头的维度

    mha = MultiHeadAttention(d_model, num_heads, d_head)

    x = torch.randn(batch_size, seq_length, d_model)

    output = mha(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)  # 应该是 (batch_size, seq_length, num_heads * d_head) 即 (2, 4, 32)