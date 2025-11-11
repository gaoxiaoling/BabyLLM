# 跟着教程编写一下LLM的基础代码
## pytorch基础
* tensor 的处理
* tensor 的操作，包括transpose, permute, view, unsqueeze, mask_fill, tril
* 一些基础的算法，包括 softmax，normalization

## transformer 基本组成部分
### tokenizer & embedding
### Causal Attention & MultiHeadAttention
transformer的Attention 相关的计算过程，实际计算过程，一些有意思的问题，softmax 在行上进行还是列上进行的区别

mask 上与教程的不同写法，导致后一个词被看到，模型只是学会了从 ABCD 预测 BCDX，X 是模型瞎猜的，但是由于是计算 batch 的 loss，导致 train 和 test的loss 非常低
### Layer-Normalization
### Transformer Block
### 简单的 GPT 模型

## 训练过程
### dataloader & dataset
batch size 以及 context length 对内存大小的影响
### loss的计算（cross-entropy）


## 基于红楼梦数据进行训练
> * output tokens:  112
generated text:  **荣府的门前有个小童**拿着，    所那边的小丫头拿着一件半点的，也有拿的小丫头，也有半天的．这也有一个的，    所以为这些丫头们的也有八头的，只有这些丫头的丫头都是些的，也有说是那些书的，有这一包
> * output tokens:  113
generated text:  **宝玉握着一个小苹果**品，拿着钱来拍手，拿着一件半碗池花，    看玉钏儿也哄着一声，抿着眼泪抿着，手抿着脚的一般．一时骂道：“好狠儿，你烧的狠的打着
> * output tokens:  111
generated text:  **林黛玉在房间里**拿着钓竿钓髂，    太见黛玉辉手上，躺着眼睛，眼看着眼睛睛直流泪，黛玉钓的钓．宝玉道：“你别动了，别动那鬼淫了．    我也不是这么话。”黛玉点
