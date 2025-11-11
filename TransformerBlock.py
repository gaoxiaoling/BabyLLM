import torch
import torch.nn as nn
# import之前准备好的 MultiHeadAttention, LayerNormalization, Feedforward
from Feedforward import Feedforward
from MultiHeadAttentionV2 import MultiHeadAttentionV2
from LayerNormalization import LayerNormalization

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionV2(d_model, num_heads)
        self.ff = Feedforward(d_model)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # transformer block的顺序，layernorm -> attention -> dropout -> residual connection，
        # 然后 layernorm -> feedforward -> dropout -> residual connection
        
        output = self.layernorm1(x)
        # Self-attention
        output = self.attention(output)
        output = self.dropout(output)
        output = output + x

        
        output2 = self.layernorm2(output)
        # Feedforward
        output2 = self.ff(output2)
        output2 = self.dropout(output2)
        output2 = output2 + output
        
        return output2
    
# 测试 TransformerBlock
# main控制
if __name__ == '__main__':
    d_model = 512
    num_heads = 8
    #d_ff = 2048
    #dropout = 0.1

    transformer_block = TransformerBlock(d_model, num_heads, dropout)
    x = torch.randn(2, 10, d_model) #batch_size=1, seq_len=10, d_model=512
    output = transformer_block(x)
    print(output.shape)