import torch
import torch.nn as nn
import os   

class CausalAttention(nn.Module):
    def __init__(self, d_model, d_attention, dropout=0):
        super(CausalAttention, self).__init__()
        self.d_model = d_model
        self.d_attention = d_attention

        self.Wq = nn.Parameter(torch.randn(d_model, d_attention))
        self.Wk = nn.Parameter(torch.randn(d_model, d_attention))
        self.Wv = nn.Parameter(torch.randn(d_model, d_attention))

        # 构建一个 mask 的缓存，
        # 从os 参数中读取 MAX_CONTEXT_LENGTH
        MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 2048))  # 假设的最大上下文长度
        # 将mask注册为buffer，这样在模型保存和加载时会自动处理
        # diagonal=1 表示主对角线及其上方为1，主对角线下方为0,因为是因果注意力
        self.register_buffer("mask", torch.tril(torch.ones(MAX_CONTEXT_LENGTH, MAX_CONTEXT_LENGTH), diagonal=1))  #固定参数的缓存

        self.scale = torch.sqrt(torch.FloatTensor([d_attention])) #缩放因子

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):   
        #x输入是多条文本，先确定有多少条，每条文本的长度是多少
        batch_size, seq_length, _ = x.size()

        #根据各个 W 计算q，k，v
        Q = torch.matmul(x, self.Wq)  # (batch_size, seq_length, d_attention)
        K = torch.matmul(x, self.Wk)  # (batch_size, seq_length, d_attention)
        V = torch.matmul(x, self.Wv)  # (batch_size, seq_length, d_attention)

        # 计算注意力分数，这里除以缩放因子，缩放因子是k的维度开根号
        # 为什么是 k 的维度的开根号，而不是q 的维度的开根号呢？
        # 因为两者必须是一样的，否则，如果 一个是n*x,一个 n*y 的话，点积就没法算了
        # 其实 q 和 k 的维度是一样的，都是 d_attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) #/ self.scale  # (batch_size, seq_length, seq_length)

        # 取出对应长度的 mask，先进行 mask 再进行 softmax，
        mask = self.mask[:seq_length, :seq_length].to(x.device)  # (seq_length, seq_length) 
        #这里进行了mask操作，如果q 和 k 很大的话，本质上不需要把整个 Attention Matrix 都算出来，不知道有没有相关的优化方法
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # 在 mask 之前或者之后除以k 的维度的开根号，softmax 结果是一样的，后面除效率更高
        attn_weights = torch.softmax(scores/self.scale, dim=-1)  # (batch_size, seq_length, seq_length)

        # 可以加一个 dropout
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # (batch_size, seq_length, d_attention)

        return output


#对这个类进行测试
if __name__ == "__main__":
    batch_size = 2
    seq_length = 4
    d_model = 16
    d_attention = 128

    x = torch.randn(batch_size, seq_length, d_model)

    causal_attention = CausalAttention(d_model, d_attention)
    output = causal_attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)    
    print(output)