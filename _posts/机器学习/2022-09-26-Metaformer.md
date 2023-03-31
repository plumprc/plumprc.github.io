---
title: Metaformer
date: 2022-09-26 19:23:20
categories:
- 机器学习
tags:
- 机器学习
- Transformer
---

<center>PAPER: <a href="https://arxiv.org/abs/2111.11418">MetaFormer Is Actually What You Need for Vision</a></center>

## Metaformer
&emsp;&emsp;Metaformer is a special implementation of Synthesizer in computer vision, which is illustrated below. In vision tasks, we generally require a token mixer and a channel mixer to learn token-wise and channel-wise information. However, synthesizers reveal that specific mixers are not necessary. A random initialized matrix can capture token-token interaction. It is worth mentioning that [AlterNet](https://arxiv.org/abs/2202.06709) also thoroughly investigates how Vision Transformers work, finding that the attention in ViT ensembles input tokens as a trainable spatial smoothing of feature maps and flatten loss landscapes, inducing better performance and robustness due to data-specific aggregation instead of long-range dependency. (notably, MLP-Mixer underperforms compared to ViTs) See the original paper for more details.

![Imgur](https://i.imgur.com/KtGGISL.png)

```python
class ModifiedLayerNorm(nn.Module):
    """
    Modified Layer Normalization normalizes vectors along channel dimension and spatial dimensions.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        # The shape of learnable affine parameters is also [num_channels, ], keeping the same as vanilla Layer Normalization.
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean([1, 2, 3], keepdim=True) # Mean along channel and spatial dimension.
        s = (x - u).pow(2).mean([1, 2, 3], keepdim=True) # Variance along channel and spatial dimensions.
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
        + self.bias.unsqueeze(-1).unsqueeze(-1)

        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
        pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        # Subtraction of the input itself is added
        # since the block already has a residual connection.
        return self.pool(x) - x


class MLP(nn.Module):
        """
        Implementation of MLP with 1*1 convolutions.
        Input: tensor with shape [B, C, H, W]
        """
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.act = act_layer()
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)

            return x


class PoolFormer(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    options: drop path; LayerScale
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., act_layer=nn.GELU, norm_layer=ModifiedLayerNorm, drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        self.channel_mixer = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))

        return x
```

&emsp;&emsp;Metaformer leverages the core idea of synthesizers and proposes a general structure for vision tasks. In their experiments, the pooling operation is used to model token-wise information and demonstrates good performance on several vision tasks. The figure below shows the ablation for Poolformer on ImageNet-1K classification benchmark. More details could be seen in original paper. I recommend you to read Synthesizer to understand intrinsic mechanism of metaformer. 

![Imgur](https://i.imgur.com/SyO49yh.png)
