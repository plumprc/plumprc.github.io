---
title: Variational Information Bottleneck
date: 2023-04-04 15:42:11
categories:
- 机器学习
tags:
- 机器学习
- 数学
---

<center>PAPER: <a href="https://arxiv.org/abs/1612.00410">Deep Variational Information Bottleneck</a></center>

## Information Bottleneck
&emsp;&emsp;In latent variable models, the information in the observation $X$ diminishes as it passes to the target $Y$. Given the Markov chain $Y\leftrightarrow X\leftrightarrow Z$ corresponding to any neural network, we have

$$p(x,y,z)=p(z\vert x,y)\cdot p(y\vert x)\cdot p(x)=p(z\vert x)\cdot p(y\vert x)\cdot p(x),$$

where the latent representation $Z$ depends only on the observation $X$ and is conditionally independent of the label $Y$. Our objective is to obtain an effective encoding $Z$ for various downstream tasks. This can be achieved by maximizing the mutual information between $Z$ and $Y$ as

$$I(Z,Y;\theta)=\int\text{d}z\text{d}yp(z,y\vert\theta)\log\frac{p(z,y\vert\theta)}{p(z\vert\theta)p(y\vert\theta)}.$$

&emsp;&emsp;According to the [Data Processing Inequality](https://en.wikipedia.org/wiki/Data_processing_inequality), we have

$$I(X,Y)\geq I(Y,Z),$$

where the maximum value of $I(Y,Z)$ is always equal to the constant $I(X,Y)$ when $Z=X$. However, this trivial solution is not a useful representation. A natural and useful constraint to apply is on the mutual information between the original data and the encoding, $I(X,Z)\leq I_c$, where $I_c$ is the information constraint. This suggests the objective

$$\max_\theta I(Z,Y;\theta)\quad\text{s.t. }I(X,Z;\theta)\leq I_c.$$

&emsp;&emsp;With the introduction of a Lagrange multiplier $\beta\geq0$, we can maximize the objective function

$$L(\theta)=I(Z,Y;\theta)-\beta I(Z,X;\theta).$$

&emsp;&emsp;This approach is known as Information Bottleneck, which aims to learn an encoding $Z$ that is maximally expressive about $Y$ while being maximally compressive about $X$. Essentially it forces $Z$ to act like a minimal sufficient statistic of $X$ for predicting $Y$.

## Variational Information Bottleneck
&emsp;&emsp;Computing mutual information in IB is challenging, but we can simplify it by converting it into an optimization problem through finding its bounds via variational inference. The first term of IB is

$$I(Y,Z)=\int\text{d}y\text{d}zp(y,z)\log\frac{p(y,z)}{p(y)p(z)}=\int\text{d}y\text{d}zp(y,z)\log\frac{p(y\vert z)}{p(y)},$$

where the $p(y\vert z)$ is intractable due to $p(z)=\int p(z\vert\theta)\text{d}\theta$. Let $q(y\vert z)$ be a variational approximation to $p(y\vert z)$. Given that $\text{KL}(p(y\vert z)\Vert q(y\vert z))\geq0$, we can get a lower bound on $I(Y,Z)$ as

$$\begin{aligned}
    I(Y,Z) &= \int\text{d}y\text{d}zp(y,z)\log p(y\vert z)-\int\text{d}y\text{d}zp(y,z)\log p(y) \\
    &\geq\underbrace{\int\text{d}y\text{d}zp(y,z)\log q(y\vert z)}_\text{lower bound}-\underbrace{\int\text{d}yp(y)\log p(y)}_{H(Y)}.
\end{aligned}$$

&emsp;&emsp;For the second term in IB, we have

$$I(X,Z)=\int\text{d}x\text{d}zp(x,z)\log\frac{p(x,z)}{p(x)p(z)}=\int\text{d}x\text{d}zp(x,z)\frac{p(z\vert x)}{p(z)}$$

where the $p(z)$ is still intractable. Let $q(z)$ be a variational approximation to $p(z)$. We can obtain an upper bound on $I(X,Z)$ as

$$\begin{aligned}
    I(X,Z) &= \int\text{d}x\text{d}zp(x,z)\log p(z\vert x)-\int\text{d}zp(z)\log p(z) \\
    &\leq\int\text{d}x\text{d}zp(x,z)\log p(z\vert x)-\int\text{d}zp(z)\log q(z) \\
    &= \underbrace{\int\text{d}x\text{d}zp(x,z)\log\frac{p(z\vert x)}{q(z)}}_\text{upper bound}.
\end{aligned}$$

&emsp;&emsp;Combining both of these bounds we have that

$$\begin{aligned}
    I(Z,Y)-\beta I(Z,X) &\geq\int\text{d}y\text{d}zp(y,z)\log q(y\vert z)-\int\text{d}x\text{d}zp(x,z)\log\frac{p(z\vert x)}{q(z)} \\
    &= \int\text{d}x\text{d}y\text{d}zp(z\vert x)p(x,y)\log q(y\vert z)-\beta\int\text{d}x\text{d}y\text{d}zp(z\vert x)p(x,y)\log\frac{p(z\vert x)}{q(z)} \\
    &= \mathbb{E}_{(x,y)\sim p(x,y),z\sim p(z\vert x)}\big[\log q(y\vert z)-\beta\text{KL}(p(z\vert x)\Vert q(z))\big] \\
    &= -J_{IB}
\end{aligned}$$

&emsp;&emsp;We estimate $p(x,y)$ using empirical data distribution through sampling and leverage the reparameterization trick with $\mu$ and $\sigma$ from the encoder $p(z\vert x)$ and $\epsilon\sim N(0,1)$. The final objective is to minimize

$$\begin{aligned}
    J_{IB} &= \frac{1}{n}\sum_{i=1}^n\bigg[\beta\text{KL}(p(z\vert x_i)\Vert q(z))-\int\text{d}zp(z\vert x_i)\log q(y_i\vert z)\bigg] \\
    &= \frac{1}{n}\sum_{i=1}^n\bigg[\beta\underbrace{\text{KL}(p(z\vert x_i)\Vert q(z))}_\text{same as VAE}-\underbrace{\mathbb{E}_{\epsilon\sim p(\epsilon)}\log q(y_i\vert z=\mu+\sigma\odot\epsilon)}_\text{likelihood}\bigg]
\end{aligned}$$

where the encoder $p(z\vert x)$ is a a multivariate Gaussian, the decoder $q(y\vert z)$ corresponds to the likelihood of $z$, and the narrowed parameter space $q{z}\sim N(0,1)$ is the same as in VAE. We present a PyTorch implementation of DeepVIB applied to MNIST as follows.

```python
class DeepVIB(nn.Module):
    def __init__(self, beta=1e-3, image_size=784, out_features=10, h_dim=400, z_dim=20):
        super(DeepVIB, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_features)
        )
        self.beta = beta
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp

        return z
    
    def forward_loss(self, y_pred, y, mu, logvar, beta):
        BCE = F.cross_entropy(y_pred, y, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())

        return BCE + beta * KLD

    def forward(self, x, y):
        # x: [batch, 784]
        h = self.encoder(x)
        # encoder: [batch, 40]
        mu, logvar = torch.chunk(h, 2, dim=1)
        # paras, z: [batch, 20]
        z = self.reparameterize(mu, logvar)
        y_pred = self.decoder(z)

        return y_pred, self.forward_loss(y_pred, y, mu, logvar, self.beta)
```

&emsp;&emsp;VIB and [$\beta$-VAE](https://openreview.net/forum?id=Sy2fzU9gl) have the same mathematical formulation, with the former being a generalized version of VAE for unsupervised learning with a tunable hyperparameter $\beta$. The standard VAE is obtained when $\beta=1$.
