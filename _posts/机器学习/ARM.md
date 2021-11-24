## REINFORCE vs Reparameterization Trick
&emsp;&emsp;REINFORCE and reparameterization trick are two of the many methods which allow us to calculate gradients of expectation of a function.

&emsp;&emsp;Given a random variable $x\sim p_\theta(x)$ where $p_\theta$ is a parametric distribution and a function $f$, for which we need to compute the gradient as follows in some optimization problems.

$$\nabla_\theta\mathbb{E}_{x\sim p_\theta(x)}[f(x)]$$

&emsp;&emsp;For an optimization problem, the above refers to the derivative of the expected value of the loss function. The problem is that the expectation is always unknow and the the derivative is with respect to $p_\theta$.
