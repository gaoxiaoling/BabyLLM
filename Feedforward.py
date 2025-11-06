import math
import torch
import torch.nn as nn

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class Feedforward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.linear1 = nn.Linear(d_model, d_ff)
        # self.activation = GELU()
        # #self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_ff, d_model)

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            GELU(),
            #nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, x):
        return self.layers(x) #+ x # residual connection
    
if __name__ == '__main__':
    # 测试 Feedforward 层
    d_model = 512
    ff = Feedforward(d_model)
    x = torch.randn(1, 10, d_model)
    output = ff(x)
    print(output.shape)