---
title: Long Sequence Modeling：LongFormer, PoolingFormer
date: 2021-01-10 01:58:18
categories:
- 机器学习
tags:
- 机器学习
- 序列分析
---

<center>PAPER: <a href="https://arxiv.org/abs/2004.05150">Longformer: The Long-Document Transformer</a><br><a href="https://arxiv.org/abs/2105.04371">Poolingformer: Long Document Modeling with Pooling Attention</a></center>

## Motivations
&emsp;&emsp;Transformer-based models are unable to process long sequence due to their self-attention operation, which scales quadratically with the sequence length. Thus, most of existing transformer models set the maximum sequence length to 512, which often leads to a worse performance in long sequence tasks.

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

&emsp;&emsp;In previous work, there are three widely adopted approaches to mitigate this problem:
* Split long sequence into segments
* Approximate the dot product in self-attention such as random projection
* Sparse attention which focuses on making each token attend to less but more important context

## LongFormer
&emsp;&emsp;Longformer’s attention mechanism is a combination of a windowed local-context self-attention and an end task motivated global attention that encodes inductive bias about the task.

&emsp;&emsp;Local attention induces each token to attend to neighboring tokens corresponding to local context.
* Sliding window attention
* Dilated sliding window (similar to dilated convolution)

![attention_pattern.png](https://s2.loli.net/2022/01/12/iwvhPpsCRkAEqWV.png)

&emsp;&emsp;To get a task-specific representations, we need to find a token with a global attention attends to all tokens across the sequence, and all tokens in the sequence attend to it. For example, BERT aggregates the representation of the whole sequence into a special token `[CLS]`. In QA, global attention is provided on all question tokens.

&emsp;&emsp;LongFormer combines local and global attention to capture representation from long sequence with $\text{O}(n)$ complexity. In particular, they use small window sizes for the lower layers and increase window sizes as they move to higher layers. They do not use dilated sliding windows for lower layers to maximize their capacity to learn and utilize the immediate local context.

![lf_comp.png](https://s2.loli.net/2022/01/12/Hwn6JEfS1lTeCPX.png)

&emsp;&emsp;LongFormer adds extra position embeddings to support up to position 4096, coping from other pre-tranined models. We can see that LongFormer performs well on long sequence modeling. Performance drops slightly when using the RoBERTa model pretrained when only unfreezing the additional position embeddings.

&emsp;&emsp;In fact, it is hard to directly apply LongFormer on other types of long sequences such as time series due to the importance of global attention and position embeddings. Task-specific global attention will bring severe inductive bias which will cause domain shift problem in transfer learning.

## TODO: Transformer-XL

## PoolingFormer

![local_attention.PNG](https://s2.loli.net/2022/01/12/a9xswpY3vZ7h82K.png)

&emsp;&emsp;PoolingFormer consists of two level attention with $\text{O}(n)$ complexity. Its first level uses a smaller sliding window pattern to aggregate information from neighbors. Its second level employs a larger window to increase receptive fields with pooling attention to reduce both computational cost and memory consumption.

![PoolingFormer.png](https://s2.loli.net/2022/01/12/snHvRN9iTmfXD65.png)

&emsp;&emsp;Pooling operation is to compress key and value matrices. In PoolingFormer, they use the lightweight dynamic convolution pooling (LDCConv) as follows:

$$\text{LDConv}(v_1,v_2,\dots,v_k)=\sum_{i=1}^k\delta_i\cdot v_i$$

&emsp;&emsp;Notice they just modifies some of layers of Transformer-based models. The window sizes of the first-level and second-level is normally set to 128 and 512. In addition, they adopt a residual connection between two levels.

![pf_comp.png](https://s2.loli.net/2022/01/12/n1YHtWyAajfz5ec.png)

&emsp;&emsp;We conjecture that the reason for the poor performance of 512 windows size is that the self-attention mechanism is difficult
to deal with remote token due to redundancy noise. We think that for every distant token, there may be too little useful information to compute attention. For the tokens that are very far away, we will discard them directly. In this way, tokens can pay more attention to key information and reduce computation and memory consumed.

![pf_comp2.png](https://s2.loli.net/2022/01/12/OtjCWcBUkomJRg1.png)

&emsp;&emsp;Table 6 compares different pooling operations. In Mix, the second-level pooling attention module is built upon the input embeddings instead of the output of the first-level attention, which illustrates the effectiveness of stacking two level attentions in Poolingformer. In Weight sharing, the first level and second level share the same attention matrices. Table 7 tells us which number of the PoolingFormer layers we should apply. The best results often happen when the number of Poolingformer layer is a quarter of the total number of layers.
