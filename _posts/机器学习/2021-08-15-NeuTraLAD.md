---
title: 《Neural Transformation Learning for Deep Anomaly Detection Beyond Images》阅读笔记
date: 2021-8-15 00:23:25
categories:
- 机器学习
tags:
- 机器学习
- 自监督学习
- 异常检测
---

<center>论文地址：<a href="https://arxiv.org/abs/2103.16440">Neural Transformation Learning for Deep Anomaly Detection Beyond Images</a></center>

## Background and Motivation
&emsp;&emsp;推荐阅读：[【入门】异常检测Anomaly Detection](https://zhuanlan.zhihu.com/p/116235115)

* Question 1: For anomaly detection beyond image data, it is often unclear which transformations to use
* Question 2: For data other than images, such as time series or tabular data, it is much less well known which transformations are useful, and it is hard to design these transformations manually
* Intuition: Embed the transformed data into a semantic space such that the transformed data still resemble their untransformed form (preserve relevant semantic information), while different transformations are easily distinguishable
* Approach: Instead of manually designing data transformations to construct auxiliary prediction tasks that can be used for anomaly detection, we derive a single objective function for jointly learning useful data transformations and anomaly thresholding

## NeuTraL AD
&emsp;&emsp;NeuTraL AD is a simple pipeline with two components: a set of learnable transformations, and an encoder. Both are trained jointly on a deterministic contrastive loss (DCL).

* Learnable Data Transformations
* Deterministic Contrastive Loss (DCL)

&emsp;&emsp;Considering data space $\chi$ with samples $D=\{x^{(i)}\sim\chi\}_{i=1}^N$ and $K$ transformations $T:=\{T_1,\dots,T_K\vert T_k:\chi\rightarrow\chi\}$. We assume here that the transformations are learnable with the parameters of transformation $T_k$ by $\theta_k$. DCL encourages each transformed sample $x_k=T_k(x)$ to be similar to its original sample $x$, while encouraging it to be dissimilar from other transformed versions of the same sample, $x_l=T_l(x)$ with $l\not=k$. We define $h(x_k,x_l)$ (in this paper they used cosine similarity) to measure the similarity between $x_k$ and $x_l$. Then DCL is as follows:

$$L_{DCL}=\mathbb{E}_{x\sim D}-\sum_{k=1}^K\log\frac{h(x_k,x)}{h(x_k,x)+\sum_{l\not=k}h(x_k,x_l)}$$

![NeuTraLAD.png](https://i.loli.net/2021/08/14/SjHhwVmzTk1fNAo.png)

&emsp;&emsp;Notice that negative samples are from the same original $x$ with different transformation $T_l$. One advantage of this approach over other methods is that its training loss is also anomaly score. Since the score is deterministic, it can be straightforwardly evaluated for new data points $x$ without original negative samples.

$$S(x)=-\sum_{k=1}^K\log\frac{h(x_k,x)}{h(x_k,x)+\sum_{l\not=k}h(x_k,x_l)}$$

* Semantics: The transformations should produce views that share relevant semantic information with the original data.
* Diversity: The transformations should produce diverse views of each sample

&emsp;&emsp;Based on the above hypothesis, this paper demonstrates that we will optimize our learnable transformations to two edge-cases if minimizing softmax loss or contrastive loss with negative samples from mini-batch. One will incur constant transformation $T_k(x)=c_k$ and the other will incur identity transformation $T(x)=x$.

&emsp;&emsp;Here are Some transformations used in this paper: ($M_k$ is a learnable mask)
* feed forward $T_k(x):=M_k(x)$
* residual $T_k(x):=M_k(x)+x$
* multiplicative $T_k(x):=M_k(x)\odot x$

## 笔者注解
&emsp;&emsp;本文提出了一个基于可学习变换的自监督异常检测的方法。一般来说，在自监督学习领域我们需要做数据增强（data augmentation）来增加样本的表示范围，同时提供可观的辅助任务（auxiliary task）。作者认为一个良好的变换簇不仅要尽可能保留原始数据的语义信息，同时变换之间要有区分度，因此将原始数据与变换后数据视为正样本对，将同一数据的不同变换后的数据视为负样本对，使用对比损失进行优化，正文的理论和实验部分也证明了不同的负采样方案确实会影响到模型的效果。本文同时也为时间序列、表格数据等不易设计人工变换的数据提供了新思路。
