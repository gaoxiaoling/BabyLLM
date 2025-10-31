
import torch
import torch.nn as nn

import os
from CausalAttention import CausalAttention     

class MultiHeadAttentionV2(nn.Module):
    # d_model 是输入的维度， num_heads 是头的数量， d_head 是每个头的维度
    def __init__(self, d_model, num_heads, dropout=0):
        super(MultiHeadAttentionV2, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        #主流做法是保持 d_model = heads × d_head
        # 每个头的维度
        self.d_head = d_model // num_heads


        # 使用 CausalAttention 作为每个头的注意力机制
        # d_head 是每个头的注意力维度, 注意力头的数量就是 num_heads, 实际Transformer 的维度是 d_head * num_heads
        # num_heads个注意力头，每个注意力头的维度是 d_head
        # self.heads = nn.ModuleList([CausalAttention(d_model, d_head, dropout) for _ in range(num_heads)])

        # 根据最基本的原理，每个 head 里面都有相应的 Wq, Wk, Wv，然后在分别得到Q,K,V，在进行计算，最后再拼起来，
        # 这样的计算过程，有多少个head就要进行多少次的矩阵乘法计算
        # 实际上也可以先将各个 Wq， Wk, Wv 拼起来，变成一个大的矩阵，然后进行一次计算，得到所有头的 Q,K,V，
        # 其中Q，K 在分别计算得到 Attention，然后再拼接起来
        # 这样的话，原本的Wq, Wk, Wv 的维度是 (d_model, d_head)，现在变成 (d_model, d_head * num_heads)
        # 从数学上来讲是一样的，只是把这样一个过程描述成多头，更加让人能够理解而已，实际就是拼起来一起算的，不是优化，而是本来就这样
        self.Wq = nn.Parameter(torch.randn(d_model, d_model))#d_model = heads × d_head
        self.Wk = nn.Parameter(torch.randn(d_model, d_model))
        self.Wv = nn.Parameter(torch.randn(d_model, d_model))

        #add dropout layer
        self.dropout = nn.Dropout(p=dropout)

        #mask buffer
        MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 2048))  # 假设的最大上下文长度
        self.register_buffer("mask", torch.tril(torch.ones(MAX_CONTEXT_LENGTH, MAX_CONTEXT_LENGTH), diagonal=1))  #固定参数的缓存



    def forward(self, input):
        #取得基本参数
        batch_size, seq_length, _ = input.size()

        #一并计算 qkv
        keys = torch.matmul(input, self.Wk)  # (batch_size, seq_length, d_head * num_heads)
        queries = torch.matmul(input, self.Wq)  # (batch_size, seq_length, d_head * num_heads)
        values = torch.matmul(input, self.Wv)  # (batch_size, seq_length, d_head * num_heads)

        #将 qkv 分成多个头
        #input 的维度是 (batch_size, seq_length, d_model)，qkv 只是改变了d_model 维度,变成 d_head * num_heads
        #这时候可以理解为一整个矩阵，要竖向拆开来，在 d_head * num_heads 维度上拆成 num_heads 份，每份 d_head
        #batch还是不变的，是最外层的，
        #本来第二层是输入内容的长度seq_length,现在切开来了，每个 head 都要处理这些sequence, 每个 head 是d_head维度
        #因此num_heads 这个维度要提到第二层，
        #举个例子，2 条输入的内容，每条长度为 4个 token，每个 token 是 8 个维度的向量，
        #如果 Transformer 是 12 个 head，每个 head是 6 个维度，
        #那 本来 Wq 是 8*6，计算出来是 2批4*6，现在12个 head 拼在一起，就变成，2*4*（12*6）(batch_size, seq_length, d_head * num_heads)
        #要变成 2 批12 个 4*6，(batch_size, num_heads, seq_length, d_head)
        #
        keys = keys.view(batch_size, seq_length, self.num_heads, self.d_head).permute(0,2,1,3)  # (batch_size, num_heads, seq_length, d_head)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.d_head).permute(0,2,1,3)  # (batch_size, num_heads, seq_length, d_head)
        values = values.view(batch_size, seq_length, self.num_heads, self.d_head).permute(0,2,1,3)  # (batch_size, num_heads, seq_length, d_head)

        #如果更加激进一点的话，其实可以用一个更大的矩阵，一个 W 包含所有 heads 的 Wq, Wk, Wv，



        # 然后在分别得到Q,K,V，在进行计算，最后再拼起来，
        # 这样的计算过程，有多少个head就要进行多少次的矩阵乘法计算
        # 实际上也可以先将各个 Wq， Wk, Wv 拼起来，变成一个大的矩阵，然后进行一次计算，得到所有头的 Q,K,V，
        # 其中Q，K 在分别计算得到 Attention，然后再拼接起来
        # 这样的话，原本的Wq, Wk, Wv 的维度是 (d_model, d_head)，现在变成 (d_model, d_head * num_heads)
        # 从数学上来讲是一样的，只是把这样一个过程描述成多头，更加让人能够理解而已，实际就是拼起来一起算的，不是优化，而是本来就这么做的

        #打印 日志 keys 的 shape
        print(f"keys shape: {keys.shape}")
        print(f"queries shape: {queries.shape}")
        print(f"values shape: {values.shape}")
        
        # reshape() - 自动处理内存连续性，view 需要连续内存，多数情况下两者差距不大
        # transpose() - 只能交换两个维度， permute可以处理多维度，
        #计算注意力分数
        # batch跟 head 两个维度是不需要转置的，只要 seq_length 跟 d_head 两个维度需要转置
        # 因为 QK^T 是 seq_length 跟 seq_length 的矩阵，要和 V 做点积，所以要转置
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, seq_length, d_head )
        # 打印日志 scores 的 shape
        print(f"scores shape: {scores.shape}")

         
        #mask处理,从缓存获取合适大小的 mask，并对 sores 进行处理
        # 从缓存中获取 mask, 并根据 seq_length 截取合适的部分
        mask = self.mask[:seq_length, :seq_length]  # (seq_length, seq_length)
        # 扩展 mask 维度以匹配 scores 的维度 (batch_size, num_heads, seq_length, seq_length)
        mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_length, seq_length)
        #这里如果不进行unsqueeze，实际效果也是一样的，（seq_length, seq_length)会被当做(1, 1, seq_length, seq_length)

        # 打印mask的具体数据
        print(f"mask shape: {mask.shape}")
        print(f"mask data: {mask}")

        # 将 mask 应用到 scores 中，将 padding 位置的 scores 设为负无穷大
        # 这时候计算出的结果就是 batch size， heads， seq_length， seq_length，多个合在一起的 Attention 矩阵
        scores = scores.masked_fill(mask == 0, float('-inf'))  # (batch_size, num_heads, seq_length, seq_length)
        # 打印日志 scores 的 shape及数据
        print(f"scores shape after mask: {scores.shape}")
        print(f"scores data after mask: {scores}")


        # 对 scores 进行 softmax 归一化，得到注意力权重
        # 对全部的数据进行归一化，而不是对每个 head 单独归一化
        attention_weights = torch.softmax(scores/self.d_head**0.5, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # 打印日志 attention_weights 的 shape
        print(f"attention_weights shape: {attention_weights.shape}")
        print(f"attention_weights data: {attention_weights}")


        # 对 values 进行加权求和，得到最终的输出
        output = torch.matmul(attention_weights, values)  # (batch_size, num_heads, seq_length, d_head)

        #这时候，output 是 (batch_size, num_heads, seq_length, d_head)
        #要把 num_heads 这个维度提到第二层， seq_length 降为第三层，d_head 保持不变
        #我们看数据的角度在这个过程中是不一样的，计算 Attention 的时候是对每个 head，在seq_length 进行计算的，
        #Attention 计算完成之后，要改回数据视角，要把num_heads* d_head作为一个维度来看，
        #所以要把这两维度重新换到最后，以便拼起来，output.permute(0, 2, 1, 3)实现维度的调换，
        #
        #最后将多个头的输出拼接起来
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.num_heads * self.d_head)  # (batch_size, seq_length, d_head * num_heads)
        #如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作。
        #view操作要求数据是连续的，
        #permute可以调整维度，相当于只是修改了对这个 tensor 的描述

        #在很多模型中，d_model = num_heads * d_head，这样出来的结果可以重新回到与输入一样的大小，
        #如果不一样的，就需要一个线性层进行维度的转换，
        return output
        

# 对 MultiHeadAttention 进行测试
if __name__ == "__main__":
    batch_size = 2
    seq_length = 5
    d_model = 8
    num_heads = 2

    mha = MultiHeadAttentionV2(d_model, num_heads)

    x = torch.randn(batch_size, seq_length, d_model)

    output = mha(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)  # 应该是 (batch_size, seq_length, num_heads * d_head) 即 (2, 4, 32)