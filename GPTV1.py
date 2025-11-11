# 基于TransformerBlock等 组装 GPT 模型

import torch
import torch.nn as nn
import tiktoken



from TransformerBlock import TransformerBlock





class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        #embedding layer将高维的 one-hot 编码的 token 转为 相对低维的向量表示
        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        #位置编码，将 token 位置信息 编码到 向量表示中
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])

        # dropout layer
        self.dropout = nn.Dropout(p=config["drop_rate"])

        # 连续的多个TransformerBlock 组成 Transformer 编码器，数量为n_layers，直接用nn.Sequential 包装起来
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(config["emb_dim"], config["n_heads"], config["drop_rate"]) for _ in range(config["n_layers"])
        ])

        #多层 Transformer Block 之后，再进行一次 layer normalization，然后用一个全连接层输出
        self.layer_norm = nn.LayerNorm(config["emb_dim"])
        self.output_layer = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, x):
        # 输入 x 是一个形状为 (batch_size, seq_len) 的张量，每个元素是一个 token 索引
        # 先将 token 索引转为 相对低维的向量表示，得到形状为 (batch_size, seq_len, emb_dim) 的张量
        x = self.embedding(x) # (batch_size, seq_len, emb_dim)
        # 位置编码，将 token 位置信息 编码到 向量表示中
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0) # (1, seq_len)
        x = x + self.pos_embedding(pos)

        #dropout
        x = self.dropout(x)

        # 然后将这个张量输入到 Transformer 编码器中，得到形状为 (batch_size, seq_len, emb_dim) 的张量
        x = self.transformer_blocks(x)
        # 最后对这个张量进行 layer normalization，得到形状为 (batch_size, seq_len, emb_dim) 的张量
        x = self.layer_norm(x)
        # 然后用一个全连接层将这个张量映射到 vocab size 维，得到形状为 (batch_size, seq_len, vocab size) 的张量
        x = self.output_layer(x)
        # 最后返回这个张量
        return x

# 测试




if __name__ == "__main__":
    # 测试 GPTModel

    GPT_CONFIG_124M = {
        "vocab_size" : 100277,
        "context_length" : 1024,
        "emb_dim" : 768,
        "n_heads" : 12,
        "n_layers" : 12,
        "drop_rate" : 0.1,
        "qkv_bias" : False
    }

    device = torch.device("mps")


    enc = tiktoken.get_encoding("cl100k_base")
    # with open("./LLM/data.txt", "r") as f:
    #     data = f.read()
    
    text = "我是机器人"
    encoded_data = [enc.encode(text)] 
    #batch = torch.randint(0, GPT_CONFIG_124M["vocab size"], (2, GPT_CONFIG_124M["context length"]))
    max_len = max(len(seq) for seq in encoded_data)
    print("max length is ", max_len)

    print(encoded_data)

    padded_data = [seq + [0] * (max_len - len(seq)) for seq in encoded_data]


    padded_data = torch.tensor(padded_data)

    padded_data = padded_data.to(device)

    model = GPTModel(GPT_CONFIG_124M)

    model.to(device)
    
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    # 测试模型前向传播
    with torch.no_grad():
        output = model(padded_data)

    # 取出output 里面每个序列中的最后那个有效的 token
    last_token_output = output[:, -1, :]
    
    # 将输出 decode 为 文本
    # 这里是否需要使用 softmax？ 答案是不需要，因为最后一层的输出是一个 logits 向量，直接取 argmax 即可
    # 用不用都一样，softmax 最大值也是 argmax 的值
    last_token_text = enc.decode(last_token_output.argmax(dim=-1).tolist())
    print(last_token_text)


    #根据last_token_text得到last_token_text的向量，并拼接在padded_data的最后，然后再次调用 model 生成下一个 token
    last_token_encoded = enc.encode(last_token_text)
    
    padded_data = torch.cat([padded_data, torch.tensor(last_token_encoded).to(device).unsqueeze(0)], dim=1)

    # 再次调用 model 生成下一个 token
    with torch.no_grad():
        output = model(padded_data)

    # 取出output 里面每个序列中的最后那个有效的 token
    last_token_output = output[:, -1, :]
    
    # 将输出 decode 为 文本
    last_token_text = enc.decode(last_token_output.argmax(dim=-1).tolist()) #to
    print(last_token_text)

