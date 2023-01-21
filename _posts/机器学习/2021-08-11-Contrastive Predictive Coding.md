---
title: Contrastive Predictive Coding
date: 2021-08-11 17:57:47
categories:
- 机器学习
tags:
- 机器学习
- 自监督学习
---

<center>PAPER: <a href="https://arxiv.org/abs/1807.03748">Representation Learning with Contrastive Predictive Coding</a></center>

## Background and Motivation
&emsp;&emsp;Recommended: [Contrastive Self-Supervised Learning
](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)

* Defects in supervised learning：
  * The underlying data has a much richer structure than what sparse labels or rewards could provide. Thus, purely supervised learning algorithms often require large numbers of samples to learn from, and converge to brittle solutions.
  * We can’t rely on direct supervision in high dimensional problems, and the marginal cost of acquiring labels is higher in problems like RL.
  * It leads to task-specific solutions, rather than knowledge that can be repurposed.
* **Key: how to model a better representations from raw observations?**
* Intuition: to learn the representations that encode the underlying shared information between different parts of the (high-dimensional) signal. At the same time it discards low-level information and noise that is more local.
* Approach: Maximize mutual information but not conditional probability $p(x\vert c)$ to get a better encoded representation

$$I(x;c)=\sum_{x,c}p(x,c)\log\frac{p(x\vert c)}{p(x)}$$

## Contrastive Predictive Coding (CPC)

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

```python
class CPC(nn.Module):
    def __init__(self, seq_len, K, in_dim, d_model):
        """
        seq_len: sequence length
        K: future K steps to predict
        """
        super(CPC, self).__init__()

        self.seq_len = seq_len
        self.K = K
        self.z_dim = in_dim
        self.c_dim = d_model

        self.encoder = nn.Sequential( 
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len)
        )
        self.gru = nn.GRU(in_dim, self.c_dim, num_layers=1, bidirectional=False, batch_first=True)
        
        # Predictions
        self.Wk = nn.ModuleList([nn.Linear(self.c_dim, self.z_dim) for _ in range(self.K)])
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size, device, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, self.c_dim).to(device)
        else: return torch.zeros(1, batch_size, self.c_dim)

    def forward(self, x, hidden):
        batch_size = x.size()[0]
        # z: [batch_size, seq_len, z_dim]
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        # Pick timestep to be the last in the context, time_C, later ones are targets 
        highest = self.seq_len - self.K # 96 - 3 = 93
        time_C = torch.randint(highest, size=(1,)).long() # between 0 and 93

        # z_t_k: [K, batch_size, z_dim]
        z_t_k = z[:, time_C + 1:time_C + self.K + 1, :].clone().cpu().float()
        z_t_k = z_t_k.transpose(1, 0)

        z_0_T = z[:,:time_C + 1,:]
        output, hidden = self.gru(z_0_T, hidden)
        
        # Historical context information
        c_t = output[:, time_C, :].view(batch_size, self.c_dim)
        
        # For the future K timesteps, predict their z_t+k, 
        pred_c_k = torch.empty((self.K, batch_size, self.z_dim)).float() # e.g. size 12*8*512
        
        for k, proj in enumerate(self.Wk):
            pred_c_k[k] = proj(c_t)
        
        nce = 0
        for k in np.arange(0, self.K):
            # [batch_size, z_dim] x [z_dim, batch_size]
            zWc = torch.mm(z_t_k[k], torch.transpose(pred_c_k[k],0,1))         
            logsof_zWc = self.lsoftmax(zWc)
            nce += torch.sum(torch.diag(logsof_zWc))
            
        nce /= -1. * batch_size * self.K
        
        argmax = torch.argmax(self.softmax(zWc), dim=0)
        correct = torch.sum(torch.eq(argmax, torch.arange(0, batch_size))) 
        accuracy = 1. * correct.item() / batch_size

        return accuracy, nce, hidden

    def predict(self, x, hidden):        
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        output, hidden = self.gru(z, hidden)

        return output, hidden
```
