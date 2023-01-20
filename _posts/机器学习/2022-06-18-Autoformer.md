---
title: Autoformer
date: 2022-06-18 01:09:54
categories:
- 机器学习
tags:
- 机器学习
- 时间序列
- Transformer
---

<center>PAPER: <a href="https://arxiv.org/abs/2106.13008">Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting</a></center>

## Motivations
&emsp;&emsp;Prior Transformer based models have tried various self-attention mechanisms to obtain long-range dependencies. However, pointwise attention limits the ability of the model to acquire correlation within time series. (e.g. We encode the **patches** but **not pixels** in Vision Transformer) Intuitively, the sub-series at the same phase position among periods often present similar temporal process. Thus, attention or other correlation methods among subsequences may be more reliable. Autoformer introduces auto-correlation mechanism in place of self-attention to discover **the dependencies among sub-series**.

![Imgur](https://i.imgur.com/Q55N6LR.png)

&emsp;&emsp;To avoid the impact of distribution shift caused by trend part in series, Autoformer also tries to disentangle the original time series into more stationary trend and seasonality information as follows.

$$x(t)=\text{Trend}(t)+\text{Season}(t)+\text{Noise}(t)$$

## Autoformer
&emsp;&emsp;The overview of Autoformer's architecture is as follows, which is similar as Informer. Notice that Autoformer removes the position embedding in original `DataEmbedding` and replaces `ProbSparseAttention` to `AutoCorrelation`. The decomposition module `SeriesDecomp` extracts the trend parts from the series through moving average. The encoder actually focuses on the seasonal part modeling, which will be used as the cross information to help the decoder refine prediction results. The final prediction of Autoformer consists of seasonal part and trend part.

![Imgur](https://i.imgur.com/pF4gJ1B.png)

```python
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=25, stride=1, padding=0)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Autoformer(nn.Module):
    def __init__(self, seq_len, pred_len, *args):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decomp = SeriesDecomp(kernel_size)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed)

        self.encoder = Encoder(
            AutoCorrelation(mask_flag=False), 
            LayerNorm(d_model)
        )
        self.decoder = Decoder(
            AutoCorrelation(mask_flag=True), # self-correlation
            AutoCorrelation(mask_flag=False), # cross-correlation
            LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out)
        )

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Initialization
        trend_init = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        
        # [B, seq_len, D]
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # [B, pred_len, D]
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, 
                                    cross_mask=dec_enc_mask, trend=trend_init)
        # Final prediction
        dec_out = trend_part + seasonal_part
        # [B, pred_len, D]
        return dec_out[:, -self.pred_len:, :]  
```

&emsp;&emsp;To capture periodic dependencies among similar subsequences, Autoformer utilizes autocorrelation coefficient $R(\tau)$ to describe the time-delay similarity between the original time series $x_t$ and its $\tau$ lag series $x_{t-\tau}$. 

$$R(\tau)=\lim_{L\rightarrow\infty}\frac{1}{L}\sum_{t=1}^Lx_tx_{t-\tau}=\mathcal{F}^{-1}\mathcal{F}(x_t)\overline{\mathcal{F}(x_t)}$$

&emsp;&emsp;In essence, $R(\tau)$ tells us all the possible period $\tau$ of the input series $x_t$. Autoformer introduces the time delay aggregation module to fuse top k possible periodic information. The figure below shows how to implement periodic aggregation using `torch.roll` and `softmax`. 

![Imgur](https://i.imgur.com/hiWK6rg.png)

```python
class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag

    def time_delay_agg_training(self, values, corr):
        batch, head, channel, length = values.shape
        # find top k possible period tau
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # aggregation
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        return V.contiguous()
```

&emsp;&emsp;Some prediction results are as follows. Autoformer successfully generates trend and seasonal part from historical series. However, autocorrelation and pointwise self-attention are all conducive to capture seasonality. We have to stress that they are all of the same typical paradigm: similar pattern matching, which is sensitive to trend part of series. We should further consider the distribution shift of the trend part in future. 

![Imgur](https://i.imgur.com/X5N8VHP.png)

![Imgur](https://i.imgur.com/jBP2eNu.png)