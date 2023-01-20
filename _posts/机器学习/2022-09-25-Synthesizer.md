---
title: Synthesizer
date: 2022-09-25 14:21:39
categories:
- 机器学习
tags:
- 机器学习
- Transformer
---

<center>PAPER: <a href="https://arxiv.org/abs/2005.00743">Synthesizer: Rethinking Self-Attention for Transformer Models</a></center>

## Motivations
&emsp;&emsp;Vanilla attention relies on the dot product operations, which is formulated as follows. $Q\in\mathbb{R}^{n\times d_k}$, $K\in\mathbb{R}^{m\times d_k}$ and $V\in\mathbb{R}^{m\times d_v}$ are matrices formed with a set of query, key and value. Softmax activates and normalizes the $m$ dimension for smoothness.

$$\text{Attn}(Q,K,V)=\text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$$

&emsp;&emsp;Specifically for self-attention, we use different projection matrices $W_q,W_k,W_v$ to obtain $Q=XW_q$, $K=XW_k$, $V=XW_v$ directly from the original input $X\in\mathbb{R}^{n\times d}$.

$$\text{self-attention}(X)=\text{Attn}(XW_q,XW_k,XW_v)=\text{softmax}(\frac{XW_qW_k^\top X^\top}{\sqrt{d_k}})XW_v$$

&emsp;&emsp;In essence, self-attention is actually a pattern of similarity computation based on dot product $XX^\top$ with learnable projection matrix. That is, we can call it token-token interactions (token-wise). The fundamental role of dot product self-attention is to learn self-alignment, i.e., to determine the relative importance of a single token with respect to all other tokens in the sequence.

&emsp;&emsp;Traditionally, attention weights are learned at the instance or sample level, where weights are produced by instance-level pairwise interactions. As a result, these instance-specific interactions often fluctuate freely across different instances as they lack a consistent global context. Thus, synthesizer is proposed as an alternative of vanilla attention.

## Synthesizer
&emsp;&emsp;Synthesizer removes the notion of query-key-values in the self-attention module and directly synthesizes the attention weights (alignment matrix) instead. The figure below illustrates different ways to obtain the alignment matrix.

![Imgur](https://i.imgur.com/atQFiHS.png)

&emsp;&emsp;To clarify the mechanism of the synthesizer, we can simply rewrite the vanilla self-attention as $Y=AXW_v$ where our goal is to learn a map $X\in\mathbb{R}^{n\times d}\rightarrow Y\in\mathbb{R}^{n\times d}$. The original matrix $A$ is a normalized "gram matrix" $B$ of $X$, which can be seen as a token-wise alignment matrix.

$$A=\text{softmax}(B)=\text{softmax}(\frac{XW_qW_k^\top X^\top}{\sqrt{d_k}})$$

&emsp;&emsp;Dense synthesizer adopt a simple two layered FFN with ReLU activations to learn the alignment matrix $B$. Dense synthesizer directly learn a transformation matrix $W\in\mathbb{R}^{d\times n}$ instead of dot product.

$$B=\text{relu}(XW_1+b_1)W_2+b_2$$

$$Y=\text{softmax}(B)\cdot(XW_3+b_3)$$

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

class DenseSynthesizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) :
        super().__init__()
        self.alignment = MLP(input_dim, hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.value_matrix = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        B = self.alignment(x)
        A = self.softmax(B)
        x = torch.matmul(A, self.value_matrix(x))

        return x
```

&emsp;&emsp;The above variant learns synthetic attention by conditioning on each input of $X$ and projecting to $N$ dimensions. Hence, the dense synthesizer conditions on each token independently, as opposed to pair-wise token interactions in the vanilla attention. Random synthesizer does not condition on any input tokens. Instead, the attention weights are initialized to random values and can be trainable or kept fixed. We can denote the random synthesizer as $B=R$.

$$Y=\text{softmax}(R)\cdot (XW+b)$$

```python
class RandomSynthesizer(nn.Module):
    def __init__(self, seq_len, input_dim) :
        super().__init__()
        self.alignment = nn.Parameter(torch.rand(seq_len, seq_len))
        self.softmax = nn.Softmax(dim=-1)
        self.value_matrix = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        B = self.alignment
        A = self.softmax(B)
        x = torch.matmul(A, self.value_matrix(x))

        return x
```

&emsp;&emsp;The basic idea of the random synthesizer is to not rely on pairwise token interactions or any information from individual token but rather to learn a task-specific alignment that works well globally across many samples. MLP-Mixers are actually a specific implementation of random synthesizers.

&emsp;&emsp;The dense synthesizer and random synthesizer also require large computation due to matrix operations. We can use low rank decomposition to reduce complexity, which is called factorized synthesizer. Factorized dense synthesizer first obtains $B_1\in\mathbb{R}^{n\times a}$ and $B_2\in\mathbb{R}^{n\times b}$ where $n=a\times b$. Then respectively repeat $B_1$ and $B_2$ with $n/b$ and $n/a$ times to get $\hat{B_1}\in\mathbb{R}^{n\times n}$ and $\hat{B_2}\in\mathbb{R}^{n\times n}$. Factorized random synthesizer directly decompose $R\in\mathbb{R}^{n\times n}$ to $R_1\in\mathbb{R}^{n\times k}$ and $R_2\in\mathbb{R}^{n\times k}$ where $k<n$.

$$B_{\text{dense}}=\hat{B_1}\otimes\hat{B_2}\qquad B_\text{random}=R_1R_2^\top$$

&emsp;Finally, we note that all of proposed synthetic attention variants can be mixed in an additive fashion. Experiments on several tasks show the competitive performance of synthesizer compared to vanilla attention. More details could be seen in original paper.
