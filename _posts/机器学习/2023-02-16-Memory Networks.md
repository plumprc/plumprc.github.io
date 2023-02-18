---
title: Memory Networks vs Transformer
date: 2023-02-16 10:58:08
categories:
- 机器学习
tags:
- 机器学习
---

|PAPER
|:-:
|<a href="https://arxiv.org/abs/1503.08895v5">End-To-End Memory Networks</a>
|<a href="https://arxiv.org/abs/2302.06461">A Study on ReLU and Softmax in Transformer</a>

## End-To-End Memory Networks
&emsp;&emsp;Memory Networks generally reason with inference components combined with long-term memory storage which stores the long-term dependencies in sequential data. The figure below illustrates the overall architecture of end-to-end memory networks.

![Imgur](https://i.imgur.com/7UUY27n.png)

&emsp;&emsp;Suppose we are given an input set $\{x_1,x_2,\dots,x_i\}$. The entire set of $\{x_i\}$ is converted into key vectors $M\in\mathbb{R}^{d_h\times d}$ to be stored in memory where $d_h$ is the number of memory slots. The embedded query $Q\in\mathbb{R}^{n\times d}$ is fed into memory networks to obtain the similarity $P$ between $q_i$ and each memory $m_i$ via the dot product followed by a softmax. Defined in this way $p_i$ is a probability vector over the inputs.

$$P=\text{softmax}(Q\cdot M^\top)$$

&emsp;&emsp;Each $x_i$ has a corresponding value vector $c_i$. The output of the memory networks is then a sum over the values $V\in\mathbb{R}^{d_h\times d}$ weighted by the probability vector $p_i$ as $H=\sum_ip_ic_i=PV$. For better understanding, we can rewrite it in key-value form

$$H=\text{softmax}(X\cdot K^\top)\cdot V$$

where $X\in\mathbb{R}^{n\times d}$ represents the input query vector and $K,V\in\mathbb{R}^{d_h\times d}$ represents the key-value memory.

## Memory Networks vs Transformer
&emsp;&emsp;Vanilla Transformer contains self-attention and feed-forward network (FFN) which can be viewed as key-value memories. Given the input sequence $X\in\mathbb{R}^{n\times d}$, the self-attention is calculated as

$$H=\text{softmax}(\frac{(XW_Q)\cdot(XW_K)^\top}{\sqrt{d}})\cdot XW_V$$

where $W_Q,W_K,W_V\in\mathbb{R}^{d\times d}$ are learnable queries, keys, and values parameters. The self-attention is identical to the key-value memory network. As for FFN, a standard FFN with two linear projections and a non-linear activation can be formulated as

$$H=\sigma(X\cdot W_1^\top+b_1)\cdot W_2+b_2$$

where $W_1,W_2\in\mathbb{R}^{d_h\times d}$ and $b_1,b_2$ indicate bias terms. In vision tasks, we generally use ReLU as an implementation of $\sigma$. It is worth stressing that there also exist some bias fine-tuning works like [BitFit](https://arxiv.org/abs/2106.10199). Here we omit the bias terms since it contains few parameters and has little influence on the results. When omitting the bias terms, FFN also has a similar form as memory networks.

&emsp;&emsp;Although the self-attention, FFN, and key-value memory networks have a similar formulation, previous works have not carefully discussed the impact of activation functions. [Kai, et al.](https://arxiv.org/abs/2302.06461) thoroughly investigated the difference between ReLU and softmax in self-attention and FFN. In general, softmax provides exponential normalization over all value slots and therefore highlights a small number of them while neglecting others, which may cause performance degradation when the number of slots is large. ReLU bypasses this problem but faces a variance exploding problem. The FFN and key-value memory with additional layer normalization are equivalent, and ReLU shows stronger capacity when dealing with a large number of values than softmax.

![Imgur](https://i.imgur.com/Zkgoxrd.png)

&emsp;&emsp;However, directly replacing the softmax with ReLU in self-attention may induce the model fails to converge. To control the variance exploding caused by ReLU, they introduce the scaling factor and a regularization loss to alleviate this problem. Given $n$ random variables $x_i\sim N(0,1)$ and $v_i\sim N(0,1)$, we have

$$y_i=\sum_{i=1}^n\text{ReLU}(x_i)v_i\sim N(0,\frac{n}{2})$$

&emsp;&emsp;Thus, the self-attention with ReLU activation can be denoted as

$$h_i=\sum_{j=1}^n\frac{\text{ReLU}(q_i^\top k_j)}{\gamma\sqrt{n/2}}v_j$$

where $\gamma$ is a temperature hyper-parameter. Kai also introduces a normalization regularization loss to guarantee the entropy of outputs. See the original paper for more details.
