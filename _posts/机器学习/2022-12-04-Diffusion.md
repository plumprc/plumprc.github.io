---
title: Diffusion Models
date: 2022-12-04 19:25:54
categories:
- 机器学习
tags:
- 机器学习
- 表示学习
---

|PAPER
|:-:
|<a href="https://dl.acm.org/doi/10.1145/1390156.1390294">Extracting and composing robust features with denoising autoencoders</a>
|<a href="https://arxiv.org/abs/1312.6114">Auto-Encoding Variational Bayes</a>
|<a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Models</a>

## Preliminary
### Autoencoders
&emsp;&emsp;Autoencoders generally contain an encoder and decoder. The encoder $f_\theta$ projects the original input $x$ into the latent representation $z$, and the decoder $g_\phi$ reconstructs the input data from $z$. That is, Autoencoders are used to learn an identical map between the input and output, which are suitable for most compression/decompression tasks.

$$x\xrightarrow{f_\theta}z\xrightarrow{g_\phi}\tilde{x}$$

$$L_{\theta,\phi}=\Vert x-g_\phi(f_\theta(x))\Vert_2$$

```python
class Autoencoders(nn.Module):
    def __init__(self, input_dim=96, hidden_states=128, z_dim=512):
        super(Autoencoders, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, z_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, input_dim)
        )
        
    def forward(self, x):
        # x: [batch, length, dimension]
        x = self.encoder(x)
        recon_x = self.decoder(x)

        return recon_x
```

&emsp;&emsp;Since the autoencoders learn the identity function, the overfitting problem is ubiquitous when the network scale expands. Denoising Autoencoders (DAE) are proposed to avoid overfitting via "denoising". DAE corrupts the input $x$ (i.e. introduce the noise) and recovers the original input data to alleviate overfitting mainly caused by the noise. Specifically, in the original DAE paper, the noise is introduced by randomly setting a fixed proportion of values in input data to zero, which is similar to dropout. In practice, we can choose any type of noise such as Gaussian noise. Recently the prevailing Masked Autoencoders (MAE) are actually a special DAE which applies masking noise to the input data.

### Variational Autoencoders
&emsp;&emsp;Different from all the autoencoders, the primary goal of Variational Autoencoders (VAE) is to learn how to **generate** natural data (i.e. estimate the true distribution $p(x)$ of $x$). For compression and decompression tasks, the encoder $f_\theta(z\vert x)$, decoder $g_\phi(x\vert z)$ and $z$ are accessible to each data $x$. But for generation tasks, we do not know the specific $z$ of each target $\tilde{x}$ to be generated. Thus, we should first determine the latent variables $z$. 

$$x\xrightarrow{p_\theta(z\vert x)}z\xrightarrow{p_\theta(x\vert z)}\tilde{x}$$

&emsp;&emsp;VAE assumes that there is no simple interpretation of the dimensions of $z$ and instead asserts that samples of $z$ can be drawn from a simple Gaussian distribution $p(z)\sim N(0,1)$. Then we can sample a large number of $z$ values to estimate $p(x)\approx\frac{1}{n}\sum_ip_\theta(x\vert z_i)$. However, $p_\theta(x\vert z)$ will be nearly zero for most $z$. The key idea behind VAE is to attempt to sample values of $z$ that are likely to have produced $x$, which means we need to learn a new approximation function $q_\phi(z\vert x)$ where we can get a distribution over $z$ values which are likely to produce $x$. Then we should make $q_\phi(z\vert x)$ and the true posterior distribution $p(z\vert x)$ more similar (i.e. minimize the KL divergence between them) so that we can estimate $p(x)\approx\mathbb{E}_{z\sim q}p_\theta(x\vert z)$.

$$KL(q(z\vert x)\Vert p(z\vert x))=\mathbb{E}_{z\sim q}[\log q(z\vert x)-\log p(z\vert x)]$$

$$\log p(z\vert x)=\log p(x\vert z)+\log p(z)-\log p(x)$$

$$\log p(x)-D(q(z\vert x),p(z\vert x))=\mathbb{E}_{z\sim q}\log p(x\vert z)-KL(q(z\vert x)\Vert p(z))$$

&emsp;&emsp;To apply SGD on the right hand side of above equation, we should specify all the terms. We know $p(z)\sim N(0,1)$ and $q$ is often initialized as Gaussian with learnable mean and variance. The expectation $\mathbb{E}_{z\sim q}\log p(x\vert z)$ can be estimated by [reparameterization trick]().

![VAE.png](https://s2.loli.net/2021/12/16/IrnsQz2dLAb47w8.png)

$$z=\mu+\sigma\odot\varepsilon$$

$$-L_{\theta,\phi}=\mathbb{E}_{\varepsilon\sim N(0,1)}\log p_\theta(x\vert z)-KL(q_\phi(z\vert x)\Vert p_\theta(z))\leq\log p(x)$$

$$\begin{aligned}
    KL(N(\mu,\sigma^2)\Vert N(0,1)) &= \int\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}\cdot\log\frac{e^{-(x-\mu)^2/2\sigma^2}/\sqrt{2\pi\sigma^2}}{e^{-x^2/2}/\sqrt{2\pi}}\text{d}x \\
    &= \frac{1}{2}\int\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}\cdot[-\log\sigma^2+x^2-(x-\mu)^2/\sigma^2]\text{d}x \\
    &= \frac{1}{2}(-\log\sigma^2+\mu^2+\sigma^2-1)
\end{aligned}$$

```python
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def forward(self, x):
        # x: [batch, 784]
        h = self.encoder(x)
        # encoder: [batch, 40]
        mu, logvar = torch.chunk(h, 2, dim=1)
        # paras, z: [batch, 20]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    # BCE with sigmoid can be replaced by nn.MSELoss()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD
```

&emsp;&emsp;In essence, VAE postulates that each data corresponds to a Gaussian distribution $p_\theta(z\vert x)$. Our goal is to learn a generator $q_\phi(z\vert x)$ and narrow the space of $z$ via minimizing the KL divergence between $p_\theta(z\vert x)$ and $N(0,1)$ for better sampling and generation. $-L_{\theta,\phi}$ is actually the Evidence Lower Bound (ELBO) of $\log p(x)$, which is derived from variational inference. That's why we call it VAE.

## TODO: DDPM