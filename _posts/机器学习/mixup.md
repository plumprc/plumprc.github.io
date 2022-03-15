---
title: mixup
date: 2022-03-16 01:06:46
categories:
- 机器学习
tags:
- 机器学习
- 数据增广
---

<center>PAPER: <a href="https://arxiv.org/abs/1710.09412">mixup: Beyond Empirical Risk Minimization</a></center>

## Motivations
&emsp;&emsp;Traditional machine learning methods always focus on **empirical risk minimization** (ERM). Given the preditions $f(x)$ and actual targets $y$, we could minimize the average of the loss function $L$ over the joint distribution $P(x,y)$:

$$R(f)=\int L(f(x),y)\text{d}P(x,y)$$

&emsp;&emsp;$R(f)$ is known as the **expected risk**. The distribution $P$ is unknown in most practical situations. Thus, we usually have access to a set of training data to approximate $P$, which is called **empirical distribution**. Minimizing it is known as the ERM principle.

&emsp;&emsp;However, the convergence of ERM is guaranteed as long as the size of the learning machine does not increase with the number of training data (few-shot). Two shortages of ERM are as follows:
* ERM allows large neural networks to memorize (instead of generalize from) the training data
* Neural networks trained with ERM change their predictions drastically when evaluated on examples just outside the training distribution (sensitive to corrupt or adversarial examples)

&emsp;&emsp;ERM is unable to explain or provide generalization on testing distributions that differ only slightly from the training data. Thus, we need data augmentation to describe a vicinity or neighborhood around each example in the training data.

## Mixup
&emsp;&emsp;Mixup is a simple and data-agnostic data augmentation routine. It can be understood as the linear interpolation of samples:

$$\hat{x}=\lambda x_i+(1-\lambda)x_j$$

$$\hat{y}=\lambda y_i+(1-\lambda)y_j$$

&emsp;&emsp;Where $x_i$ and $x_j$ are raw input vectors; $y_i$ and $y_j$ are one-hot label encodings. Figure 1b shows that mixup leads to decision boundaries that transition linearly from class to class, providing a smoother estimate of uncertainty.

![mixup_1.png](https://s2.loli.net/2022/03/15/a6CSFUd8oicZlf9.png)

---

&emsp;&emsp;Figure 2 shows the strong robustness and generalization brought by mixup. Figure 3 shows that the models trained using mixup significantly outperform their analogues trained with ERM on the CIFAR-10 and CIFAR-100 datasets.

![mixup_2.png](https://s2.loli.net/2022/03/15/JUYeoAfCEWyXbdM.png)

![mixup_3.png](https://s2.loli.net/2022/03/15/s8avNuxgRnATBbq.png)

&emsp;&emsp;The following experiemnts show that mixup improves generalization of the model to corrupted labels (true labels are replaced by random noise). Dropout and mixup can effectively reduce overfitting.

![mixup_4.png](https://s2.loli.net/2022/03/15/7DTb9xYigGecsKR.png)

&emsp;&emsp;To improve robustness of the model to adversarial examples, conventional approaches always penalize the norm of the Jacobian of the model to control its Lipschitz constant, or do data augmentation by producing and training on adversarial examples. All of these methods add significant computational overhead to ERM.

$$\max_g\min_d\mathbb{E}L(d(x),1)+L(d(g(z),0))$$

&emsp;&emsp;Where $g(z)$ are fake samples from generator. Mixup could improve the smoothness of the discriminator, which guarantees a stable source of gradient information to the generator.

$$\max_g\min_d\mathbb{E}L(d(\lambda x+(1-\lambda)g(z)), \lambda)$$

&emsp;&emsp;Table 5 shows the ablation study on mixup. We can see the effect of smoothness brought by mixup. They find that $\alpha\in[0.1,0.4]$ leads to improved performance over ERM. Notice that convex combinations of three or more examples with weights does not provide further gain.

![mixup_5.png](https://s2.loli.net/2022/03/15/hstB6R7nioPluq4.png)

&emsp;&emsp;To sum up, mixup is a data augmentation method that consists of only two parts: random convex combination of raw inputs, and correspondingly, convex combination of one-hot label encodings. Like other methods like label smoothing and AutoAugment, mixup improves the smoothness of the data distribution, and then enhances robustness and generalization of the model.

## Extension of mixup
&emsp;&emsp;[CutMix](https://arxiv.org/abs/1905.04899v2): patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. 

![mixup_6.png](https://s2.loli.net/2022/03/16/qAJ4GtmZKkWNw5f.png)

&emsp;&emsp;CutMix is motivated by regional dropout where informative pixels dropped are not be utilized. Notice that the binary rectangular masks are uniformly sampled.

---

&emsp;&emsp;[Manifold mixup](https://arxiv.org/abs/1806.05236): interpolate hidden representations. High-level representations are often low-dimensional and useful to linear classifiers. Thus, linear interpolations of hidden representations could explore meaningful regions of the feature space effectively.

![mixup_7.png](https://s2.loli.net/2022/03/16/A7jrLxFt8TCaMBb.png)

![mixup_8.png](https://s2.loli.net/2022/03/16/JEHcjxP9OtqDV4G.png)

&emsp;&emsp;Manifold Mixup improves the hidden representations and decision boundaries of neural networks at multiple layers.

---

&emsp;&emsp;[PatchUp](https://arxiv.org/abs/2006.07794): a hidden state block-level regularization technique, which is applied on selected contiguous blocks of feature maps from a random pair of samples.

![mixup_9.png](https://s2.loli.net/2022/03/16/A74epGM2blXVRf8.png)

&emsp;&emsp;PatchUp could be seen as CutMix on Manifold of hidden states.

---

&emsp;&emsp;[PuzzleMix](https://arxiv.org/abs/2009.06962): exploit **saliency** and local statistics for optimal mixup.

![mixup_10.png](https://s2.loli.net/2022/03/16/eT7HCWNr3zxqkt9.png)

![mixup_11.png](https://s2.loli.net/2022/03/16/RiBTlXca2UjnyxJ.png)

&emsp;&emsp;Randomly selecting mask is not a proper strategy because it may mask some regions with important information (objects). PuzzleMix utilizes the saliency to avoid this problem.

---

&emsp;&emsp;[FMix](https://arxiv.org/abs/2002.12047): include masks of arbitrary shape rather than just square. It samples a low frequency grey-scale mask from Fourier space which can then be converted to binary with a threshold.

![mixup_12.png](https://s2.loli.net/2022/03/16/5VoNZ6fDb8mhqMF.png)
