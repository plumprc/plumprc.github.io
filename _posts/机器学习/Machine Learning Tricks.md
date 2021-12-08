---
title: Machine Learning Tricks
date: 2021-12-08 16:28:25
categories:
- 机器学习
tags:
- 机器学习
- Estimation
---

<center>Original blog：<a href="https://blog.shakirm.com/ml-series/trick-of-the-day/">Machine Learning Trick of the Day</a></center>

## Replica Trick
&emsp;&emsp;The Replica trick is used for analytical computation of log-normalising constants. Consider a probabilistic model whose normalising constant is $Z(x)$, for data $x$. The Replica trick says that:

$$\mathbb{E}(\ln Z)=\lim_{n\rightarrow0}\ln\mathbb{E}(Z^n)$$

&emsp;&emsp;The left-hand side is often called the quenched free energy (free energy averaged over multiple data sets). The replica trick transforms the expectation of the log into the log of an expectation, while the partition function $Z$ is not easy to compute. Notice the right hand $\mathbb{E}(Z^n)$ is just the $n$th-order moment of $Z$. 

## Gaussian Integral Trick
&emsp;&emsp;The Gaussian integral trick allows us to turn a function which is an exponential in $x^2$ into an exponential which is linear in $x$. Notice $y$ is the auxiliary variable we introduce and $a$ is the scaling factor. 

$$\int\exp(-ay^2+xy)\text{d}y=\int\exp(-a(y-\frac{x}{2a})^2+\frac{x^2}{4a})\text{d}y=\sqrt{\frac{\pi}{a}}\exp(\frac{x^2}{4a})$$

&emsp;&emsp;The left-hand side integral is a Gaussian with mean $\mu=\displaystyle\frac{x}{2a}$ and variance $\sigma^2=\displaystyle\frac{1}{2a}$. Its normalisation constant is just $\sqrt{2\pi\sigma}$.

&emsp;&emsp;For example, we can apply this trick to binary Markov Random Field. Binary MRFs have a joint probability, for binary random variable $x$, we have:

$$p(x)=\frac{1}{Z}\exp(\theta^Tx+x^TWx)$$

&emsp;&emsp;Where $Z$ is the normalising constant which is hard to compute. Applying Gaussian integral trick we can turn this energy function into a Gaussian whose normalisation constant is easy to get.

&emsp;&emsp;The Gaussian integral trick is just one from a large class of [variable augmentation](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture19.pdf) strategies that are widely used in statistics and machine learning. They work by introducing auxiliary variables into our problems that induce an alternative representation, and that then give us additional statistical and computational benefits.

## Hutchinson's Trick
&emsp;&emsp;The Hutchinson's trick allows us to compute an unbiased estimator of the trace of one matrix. Notice $z$ is a random variable sampled from some typical distribution.

$$\text{tr}(A)=\mathbb{E}(z^TAz)$$

&emsp;&emsp;If $z$ is sampled at random from a Gaussian or Rademacher distribution which has statistical and computational benefits, it will reduce the computational complexity of estimation. This trick reminds us that sometimes we can avoid complex computation through randomised (Monte Carlo) sampling.

## Reparameterization Trick
&emsp;&emsp;The Reparameterization trick allows us to replace complex random variable by simpler random variable. For instance, we have done three replacement on the tricks mentioned above.

### One-liners
&emsp;&emsp;One-liners are tools which can generate random variates in one line of code. We can regard a one-liner as a transformation $g(x;\theta)$ sampling $x$ from the initial base distribution $p(x)$, ultimately generating a sample from a desired distribution $p(z)$.

$$p(x)\xrightarrow{g(x;\theta)}p(z)$$

### Reparameterization for Stochastic Optimisation
&emsp;&emsp;In optimization, we often need to compute the gradient of an expectation of a smooth function $f$:

$$\nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z))=\nabla_\theta\int p(z;\theta)f(z)\text{d}z$$

&emsp;&emsp;The problem is that the integral on the right-hand side is unknown and the parameter $\theta$ is with respect to $p$. Reparameterization transforms $p(z)$ into $p(x)$ to decouple gradient and distribution.

$$\nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z))=\mathbb{E}_{p(x)}(\nabla_\theta f(g(x;\theta)))$$

&emsp;&emsp;Using reparameterization has given us a new approach for optimisation: we can obtain an unbiased estimator of the gradient by computing the expectation by Monte Carlo integration.

$$\mathbb{E}_{p(x)}(\nabla_\theta f(g(x;\theta)))=\frac{1}{S}\sum_{s=1}^S\nabla_\theta f(g(x^{(s)},\theta))\quad\varepsilon^{(s)}\sim p(x)$$

## Log Derivative Trick
&emsp;&emsp;In machine learning, we always need to cope with probabilities or log-probabilities. This trick can help us solve stochastic optimisation problems, and will give us a new way to derive stochastic gradient estimators.

### Score Functions
&emsp;&emsp;Apply log derivative trick on the logarithm of a function $p(x;\theta)$ with respect to parameters $\theta$, we can get:

$$\nabla_\theta\log p(x;\theta)=\frac{\nabla_\theta p(x;\theta)}{p(x;\theta)}$$

&emsp;&emsp;The significance of this trick is realised when the function $p(x;\theta)$ is a likelihood function. The left-hand side is the score function and the right-hand side is the score ratio. Notice the expected value of the score is zero.

$$\mathbb{E}_{p(x;\theta)}(\nabla_\theta\log p(x;\theta))=\mathbb{E}_{p(x;\theta)}(\frac{\nabla_\theta p(x;\theta)}{p(x;\theta)})=\int\nabla_\theta\log p(x;\theta)\text{d}x=\nabla_\theta1=0$$

&emsp;&emsp;This property allows us to subtract any term with zero expectation from the score. Notice the variance of the score is the [Fisher information](https://en.wikipedia.org/wiki/Fisher_information).

### Score Function Estimators
&emsp;&emsp;Now we should consider how to compute the gradient of an expectation of a function $f$.

$$\nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z))=\nabla_\theta\int p(z;\theta)f(z)\text{d}z$$

&emsp;&emsp;This gradient is difficult to compute because the integral is unknown and the parameters $\theta$ is with respect to the distribution $p(z;\theta)$. Applying log derivative trick we will get an excellent stochastic estimator.

$$\begin{aligned}
    \nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z)) &= \int\nabla_\theta p(z;\theta)f(z)\text{d}z \\
    &= \int\frac{p(z;\theta)}{p(z;\theta)}\nabla_\theta p(z;\theta)f(z)\text{d}z \\
    &= \int p(z;\theta)\nabla_\theta\log p(z;\theta)f(z)\text{d}z \\
    &= \mathbb{E}_{p(z;\theta)}(f(z)\nabla_\theta\log p(z;\theta)) \\
    &\approx \frac{1}{S}\sum_{s=1}^Sf(z^{(s)})\nabla_\theta\log p(z^{(s)};\theta)\quad z^{(s)}\sim p(z)
\end{aligned}$$

&emsp;&emsp;$z^{(s)}$ is sampled from $p(z)$ using Monte Carlo methods. This is an unbias estimator of the gradient. Notice taht $f(z)$ may not be differentiable.

### Control variates
&emsp;&emsp;To make this Monte Carlo estimator effective, we must ensure that its variance is as low as possible. We need to introduce the [control variates](https://en.wikipedia.org/wiki/Control_variates) to solve this problem. For example, we can use the modified estimator as follows:

$$\nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z))=\mathbb{E}_{p(z;\theta)}((f(z)-\lambda)\nabla_\theta\log p(z;\theta))$$

&emsp;&emsp;Generally speaking, control variates is always with zero expectation which is used to reduce the variance of one known estimator. Assume we have $\mathbb{E}(m)=\mu$ where $m$ is an unbias estimator for $\mu$. Suppose we calculate $\mathbb{E}(t)=r$, then:

$$m^*=m+c(t-r)$$

$$\text{Var}(m^*)=\text{Var}(m)+c^2\text{Var}(t)+2c\text{Cov}(m,t)$$

&emsp;&emsp;We can choose $c^*=-\text{Cov}(m,t)/\text{Var}(t)$ so that we can get:

$$\text{Var}(m^*)=\text{Var}(m)-\frac{\text{Cov}^2(m,t)}{\text{Var}(t)}$$

&emsp;&emsp;Thus, we successfully reduce the variance of the estimator $m$ while maintaining its expectation.

## Density Ratio Trick
&emsp;&emsp;Apart from difference, we can use density ratio to compare two probability densities as follows.

$$r(x)=\frac{p(x)}{q(x)}$$

&emsp;&emsp;The density ratio gives the correction factor needed to make two distributions equal. We can see it in Bayes' Theorem, KL divergence, mutual information and many sampling methods.

### Density Ratio Estimation
&emsp;&emsp;It is important for us to efficiently compute the density ratio. In practice, directly estimating $p(x)$ or $q(x)$ is hard due to complex unknown distribution. Based on sampling method, we can construct a binary classifier $S(x)$ to distinguish between samples from the two distributions so that we can convert the problem of estimating density ratio to binary classification.

$$r(x)=\frac{p(x)}{q(x)}=\frac{p(x\vert y=+1)}{p(x\vert y=-1)}=\frac{p(y=+1\vert x)\cdot p(x)\cdot p(y=-1)}{p(y=-1\vert x)\cdot p(x)\cdot p(y=+1)}=\alpha\frac{S(x)}{1-S(x)}$$

&emsp;&emsp;The idea of using classifiers to compute density ratios is widespread. We can see it in many unsupervised learning methods such as contrastive learning and generative adversarial networks.

## Summary