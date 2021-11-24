## Replica trick
&emsp;&emsp;The Replica trick is used for analytical computation of log-normalising constants. Consider a probabilistic model whose normalising constant is $Z(x)$, for data $x$. The Replica trick says that:

$$\mathbb{E}(\ln Z)=\lim_{n\rightarrow0}\ln\mathbb{E}(Z^n)$$

&emsp;&emsp;The left-hand side is often called the quenched free energy (free energy averaged over multiple data sets). The replica trick transforms the expectation of the log into the log of an expectation, while the partition function $Z$ is not easy to compute. Notice the right hand $\mathbb{E}(Z^n)$ is just the $n$th-order moment of $Z$. 

## Gaussian integral trick
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

## Reparameterisation Trick
&emsp;&emsp;The Reparameterisation trick allows us to replace complex random variable by simpler random variable. For instance, we have done three replacement on the tricks mentioned above.

### One-liners
&emsp;&emsp;One-liners are tools which can generate random variates in one line of code. We can regard a one-liner as a transformation $g(x;\theta)$ sampling $x$ from the initial base distribution $p(x)$, ultimately generating a sample from a desired distribution $p(z)$.

$$p(x)\xrightarrow{g(x;\theta)}p(z)$$

### Reparameterisation for Stochastic Optimisation
&emsp;&emsp;In optimization, we often need to compute the gradient of an expectation of a smooth function $f$:

$$\nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z))=\nabla_\theta\int p(z;\theta)f(z)\text{d}z$$

&emsp;&emsp;The problem is that the integral on the right-hand side is unknown and the parameter $\theta$ is with respect to $p$. Reparameterisation transforms $p(z)$ into $p(x)$ to decouple gradient and distribution.

$$\nabla_\theta\mathbb{E}_{p(z;\theta)}(f(z))=\mathbb{E}_{p(x)}(\nabla_\theta f(g(x;\theta)))$$

&emsp;&emsp;Using reparameterisation has given us a new approach for optimisation: we can obtain an unbiased estimator of the gradient by computing the expectation by Monte Carlo integration.

$$\mathbb{E}_{p(x)}(\nabla_\theta f(g(x;\theta)))=\frac{1}{S}\sum_{s=1}^S\nabla_\theta f(g(x^{(s)},\theta))\quad\varepsilon^{(s)}\sim p(x)$$

## Log Derivative Trick
