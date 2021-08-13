---
title: 《Representation Learning with Contrastive Predictive Coding》阅读笔记
date: 2021-08-11 17:57:47
categories:
- 机器学习
tags:
- 机器学习
- 表示学习
- 自监督学习
---

<center>论文地址：<a href="https://arxiv.org/abs/1807.03748">Representation Learning with Contrastive Predictive Coding</a></center>

## Background and Motivation
&emsp;&emsp;推荐阅读：[Contrastive Self-Supervised Learning
](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)

* Defects in supervised learning：
  * The underlying data has a much richer structure than what sparse labels or rewards could provide. Thus, purely supervised learning algorithms often require large numbers of samples to learn from, and converge to brittle solutions.
  * We can’t rely on direct supervision in high dimensional problems, and the marginal cost of acquiring labels is higher in problems like RL.
  * It leads to task-specific solutions, rather than knowledge that can be repurposed.
* **Key: how to model a better representations from raw observations?**
* Intuition: to learn the representations that encode the underlying shared information between different parts of the (high-dimensional) signal. At the same time it discards low-level information and noise that is more local.
* Approach: Maximize mutual information but not conditional probability $p(x\vert c)$ to get a better encoded representation

$$I(x;c)=\sum_{x,c}p(x,c)\log\frac{p(x\vert c)}{p(x)}$$

## Contrastive Predictive Coding(CPC)
&emsp;&emsp;推荐阅读：[“噪声对比估计”杂谈：曲径通幽之妙](https://kexue.fm/archives/5617/comment-page-1)；[Noise Contrastive Estimation 前世今生——从 NCE 到 InfoNCE](https://zhuanlan.zhihu.com/p/334772391)

![CPC.png](https://i.loli.net/2021/08/13/kj8S9LBsZaJ7u3C.png)

* Applicable to sequential data
* $g_{enc}$ : a non-linear encoder mapping input sequence $x_t$ to $z_t$
* $g_{ar}$ : an autoregressive model which summarizes all $z_{\leq t}$ to produce a context latent representation $c_t$
* $f_k$ is modeled to preserve the mutual information between $x_{t+k}$ and $c_t$ 

$$f_k(x_{t+k},c_t)\propto\frac{p(x_{t+k}\vert c_t)}{p(x_{t+k})}\qquad f_k(x_{t+k},c_t)=\exp(z_{t+k}^TW_kc_t)$$

&emsp;&emsp;Given a set $X=\{x_1,\dots,x_N\}$ of $N$ random samples containing one positive sample from $p(x_{t+k}\vert c_t)$ and $N-1$ negative samples from $p(x_{t+k})$, we optimize:

$$L_N=-\mathbb{E}(\log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)})$$

&emsp;&emsp;Optimizing $L_N$ is equivalent to maximizing the mutual information between $x_{t+k}$ and $c_t$

$$p(d=i\vert X,c_t)=\frac{p(d=i,X\vert c_t)}{p(X\vert c_t)}=\frac{p(x_i\vert c_t)\prod_{k\not=i}p(x_k)}{\sum_jp(x_j\vert c_t)\prod_{k\not=j}p(x_k)}=\frac{\frac{p(x_i\vert c_t)}{p(x_i)}}{\sum_j\frac{p(x_j\vert c_t)}{p(x_j)}}$$

$$I(x_{t+k},c_t)\geq\log N-L_N$$

![CPC_detail.png](https://i.loli.net/2021/08/13/msKxnL5Egf1vueb.png)

## 笔者注解
&emsp;&emsp;论文的两个关键部分，源于信号处理的预测编码（predictive coding）是自监督学习的常用方法，对比损失（contrastive loss）是优化表示学习的有效策略，是继中心损失（center loss）、三元组损失（triplet loss）后新的负采样方案。本文的主要贡献在于提出了针对可序列化数据提高表示学习能力的统一框架 CPC，同时在多个领域贡献了相当丰富的实验数据。笔者认为后续的工作应进一步着眼于如何挖掘时序数据的高效特征表示，以及在负样本的规划上应做进一步的考量。
