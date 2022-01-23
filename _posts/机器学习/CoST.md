https://openreview.net/forum?id=PilZY3omXV2

## Motivations
disentangled seasonal trend representation

time and frequency contrastive learning

* Observed: raw data with noise, may causing over-fitiing
* Season
* Trend

The situation is exacerbated when the learned representations are entangled

distribution shift - to find other invariant module

existing method: formulate time series as a sum of trend, seasonal and error variables

trend representations are learned in the time domain while seasonal representations are learned via frequency domain.

noise!

data augmentation to generate error: scale, shift, jitter

## CoST
learn representation for each timestamp

Trend Feature Disentangler: a mixture of `CasualConv` with the look-back windows of different size

?positive samples

how to capture intra-frequency level interactions?

learnable Fourier layer: apply a linear layer on FFT

amplitude and phase

positive samples are the different views of the same data

negative samples are the same view of augmented data in a mini-batch

