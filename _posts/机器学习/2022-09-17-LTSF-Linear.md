---
title: LTSF-Linear
date: 2022-09-17 17:19:47
categories:
- 机器学习
tags:
- 机器学习
- 时间序列
---

<center>PAPER: <a href="https://arxiv.org/abs/2205.13504">Are Transformers Effective for Time Series Forecasting?</a></center>

## Motivations
&emsp;&emsp;Recently, Transformer-based models dominate the long-term time series forecasting (LTSF) task due to the ability of attention to capture long-term dependencies. The pipeline of existing Transformer-based LTSF solutions are summarized as below.

![Imgur](https://i.imgur.com/4zMLTwx.png)

&emsp;&emsp;However, the mechanism of attention is permutation-invariant and temporal-agnostic, which is in contrast to the intrinsic features of time series. Even if adding some positional encoding techniques can preserve some ordering information, it is still inevitable to have temporal information loss or distortion after position embedding. This paper challenges the effectiveness of existing Transformer-based models on time series and proposes a set of linear methods, named LTSF-Linear, to validate it.

## LTSF-Linear
&emsp;&emsp;LTSF-Linear is a set of linear models. Vanilla Linear is just a one-layer linear model. DLinear and NLinear are two enhanced types of linear model through disentanglement and normalization. Specifically, DLinear first decomposes the input time series into a trend part and a seasonality part. Then, two one-layer linear layers are used to predict trend and seasonality parts in future. The sum of them are the final prediction. NLinear takes the sequence after subtracting the last value of it as input. The final prediction is the output of one linear layer added with this value for alignment. The runnable code is as follows.

```python
class VanillaLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(VanillaLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        return x # [Batch, Output length, Channel]

# Disentanglement
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, kernel_size):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(self.kernel_size)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomp(x)
        trend = self.linear_trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = self.linear_seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        return trend + seasonal # [Batch, Output length, Channel]

# Normalization
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x + seq_last # [Batch, Output length, Channel]
```

![Imgur](https://i.imgur.com/C4FpOh3.png)

&emsp;&emsp;LTSF-Linear outperforms on nine benchmarks compared to existing SOTA Transformer-based models. The size of look-back window plays a vital role in forecasting accuracy of LTSF-Linear models because it determines how much we can learn from historical data. But the performance of Transformer-based models always degrades or stays stable when increasing the size of look-back window, which means they fail to capture temporal information in a long sequence.

![Imgur](https://i.imgur.com/w8xilIr.png)

&emsp;&emsp;Specifically, the prediction of Transformer-based models always has a huge level-shift (mean) or fluctuation (variance). Disentanglement and normalization can alleviate this distribution shift to some extent but this problem still remains in visualized results.

![Imgur](https://i.imgur.com/p9DJonz.png)

&emsp;&emsp;Besides, the results of supplementary experiments show the impact of embedding strategies and the size of training data. Vanilla attention is temporal-agnostic so that we have to embed extra global position information, inducing inevitable distortion in original time series data. Interestingly, the size of training data is not that important, which is different from natural language or computer vision.

![Imgur](https://i.imgur.com/nHXl7nT.png)

![Imgur](https://i.imgur.com/rmNHYa5.png)

&emsp;&emsp;In summary, we have to rethink the strategies of input embedding and the mechanism of vanilla attention in Transformer-based models on time series data. Temporal/Causal relation extraction should be also taken into consideration.
