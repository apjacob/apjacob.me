---
layout: post
title:  "A Tutorial On Boundary-Seeking Generative Adversarial Networks"
date:   2018-01-12
desc: "A Tutorial On Boundary-Seeking Generative Adversarial Networks"
keywords: "GAN,BGAN,Discrete-GAN"
categories: [GAN]
tags: [GAN]
icon: icon-python
authors: Devon Hjelm and Athul Paul Jacob
---
$$
\newcommand{\pd}[2]{\partial{#1}/\partial{#2}}
\newcommand{\pdf}[2]{\frac{\partial{#1}}{\partial{#2}}}
\newcommand{\sam}[2]{\{#1\}^{(#2)}}
\newcommand{\dom}[1]{\mathcal{#1}}
\newcommand{\DD}{F}
\newcommand{\GG}{G}
\newcommand{\GN}{D}
\newcommand{\SN}{T}
\newcommand{\DD}{F}
\newcommand{\GG}{G}
\newcommand{\GN}{D}
\newcommand{\SN}{T}
\newcommand{\SNp}{\SN_{\dparams}}
\newcommand{\SNo}{\SN^{\star}}
\newcommand{\nonlin}{\nu}
\newcommand{\wnorm}{\beta}
\newcommand{\wnormz}{\alpha(z)}
\newcommand{\DV}{\mathcal{D}}
\newcommand{\iw}{w}
\newcommand{\iwo}{\iw^{\star}}
\newcommand{\iwt}{\tilde{\iw}}
\newcommand{\RR}[0]{\mathbb{R}}
\newcommand{\QQ}{\mathbb{Q}}
\newcommand{\PP}{\mathbb{P}}
\newcommand{\EE}{\mathbb{E}}
\newcommand{\ND}[2]{\mathcal{N}(#1, #2)}
\newcommand{\gparams}{\theta}
\newcommand{\dparams}{\phi}
\newcommand{\fd}{f}
\newcommand{\fdc}{\{\fd\}^{\star}}
\newcommand{\genprob}{\QQ}
\newcommand{\realprob}{\PP}
\newcommand{\gendens}{q}
\newcommand{\realdens}{p}
\newcommand{\genprobp}{\genprob\_{\gparams}}
\newcommand{\gendensp}{\gendens\_{\gparams}}
\newcommand{\gencond}[2]{g(#1 \mid #2)}
\newcommand{\gencondp}[2]{g\_{\gparams}(#1 \mid #2)}
\newcommand{\realcondt}[2]{\tilde{p}(#1 \mid #2)}
\newcommand{\prior}{h}
\newcommand{\realdensest}{\tilde{\realdens}}
\newcommand{\realdenslow}{\hat{\realdens}}
\newcommand{\dist}[3]{\mathcal{V}(#1, #2, #3)}
\DeclareMathOperator\*{\argmin}{\arg \min}
\DeclareMathOperator\*{\argmax}{\arg \max}
$$



In this post, we discuss our paper, [Boundary-Seeking Generative Adversarial Networks](https://openreview.net/pdf?id=rkTS8lZAb) which was recently accepted as a conference paper to ICLR 2018. 

## **Generative Adversarial Networks**
Generative adversarial networks(a.k.a GANs)[Goodfellow et. al., 2014] is a unique generative learning framework that uses two separate models, a generator and discriminator, with opposing or *adversarial* objectives. Training a GAN only requires back-propagating a learning signal that originates from a learned objective function, which corresponds to the loss of the discriminator trained in an adversarial manner.
This framework is powerful because it trains a generator without relying on an explicit formulation of the probability density, using only samples from the generator to train. 
We will give a brief overview of this framework.

<figure>
  <img src="{{ site.img_path_blog_1 }}/BGAN/GAN-IMG1.png">
  <figcaption>Fig 1.</figcaption>
</figure>


Suppose we have empirical samples from a distribution \\(\mathbb{P}\\), \\({x^{(i)} \in \mathcal{X}}\_{i=1}^M\\), where \\(\mathcal{X}\\) is the domain of images, word or character-based representations of natural language etc.

Our objective is then to find an induced distribution, \\(\mathbb{Q}\_\theta\\) that describes these samples well.<br/>
We start with a simple prior, \\(h(z)\\) which is typically a gaussian or a uniform distribution and two *players*:  The **generator** (\\(\mathbb{G}\_{\theta}: \mathcal{Z} \rightarrow \mathcal{X}\\)) and the **discriminator** (\\(\mathbb{D}\_{\phi}: \mathcal{X} \rightarrow \mathcal{Z}\\))

The goal of the generator is to find the parameters \\(\theta\\) to find the induced distribution \\(\mathbb{Q}\_\theta\\). While, the goal of the discriminator is to classify the real and fake(generated) samples correctly.

In essense, the generator is trained to fool the discriminator into thinking that the generated samples comes from the true distribution \\(\mathbb{P}\\). While, the discriminator, is trained to distiguish between the samples from \\(\mathbb{P}\\) and the samples from \\(\mathbb{Q}\_\theta\\).

This can then be formalized as a minimax game:
$$ \mathcal{V}(\mathbb{P}, \mathbb{Q}\_{\theta}, \mathbb{D}\_{\phi}) = \mathbb{E}\_{\mathbb{P}}[\log \mathbb{D}\_{\phi}(x)] + \mathbb{E}\_{h(z)}[\log(1 - \mathbb{D}\_{\phi}(\mathbb{G}(z))]$$

$$(\hat{\theta}, \hat{\phi}) = \argmin\_{\theta} \argmax\_{\phi} \mathcal{V}(\mathbb{P}, \mathbb{Q}\_{\theta}, \mathbb{D}\_{\phi}) $$

GANs have been shown to generate often-diverse and realistic samples even when trained on high-dimensional large-scale continuous data. GANs however have a serious limitation on the type of variables they can model, because they require the composition of the generator and discriminator to be fully differentiable.

With discrete variables, this is not true. For instance, consider using a step function at the end of a generator in order to generate a discrete value. In this case, back-propagation alone cannot provide the training signal, because the derivative of a step function is 0 almost everywhere. This is problematic, as many important real-world datasets are discrete, such as character- or word-based representations of language.



## **Probability Difference Measures**
<figure>
  <img src="{{ site.img_path_blog_1 }}/BGAN/Difference.png">
  	<figcaption>Fig 2. </figcaption>
</figure>

Let's take a step back and try to understand one of the key choices that differentiates GANs.
Suppose we are given two probability distributions, \\(\realprob\\) and \\(\genprobp\\). We need a *difference* measure between these distributions. The goal is to essentially, find \\(\theta\\) such that this difference measure is minimized.
There are a variety of difference measure families such as:

1. **f-divergences:** KL-divergence, reverse KL-divergence, Hellinger distance, Total variation distance, \\(\mathcal{X}^{2}\\)-divergence etc.
2. **IPM:** Kantorovich (Wasserstein dual), MMD, Fisher distance etc. 

In the next section, we will focus on the \\(f\\)-divergence family of distances.


#### \\(f\\)-divergences
<figure>
  <img src="{{ site.img_path }}/BGAN/FD-1.png">
  	<figcaption>Fig 3. </figcaption>
</figure>

Suppose, we introduce a **convex** (lower semi-continuous) **function**:
$$\fd: \RR_{+} \rightarrow \RR$$ 
$$\fd(1) = 0$$
The divergence generated by \\(f\\):
$$ \DV\_{\fd}(\realprob || \genprob\_{\gparams}) = \EE\_{\genprob\_{\gparams}}\left[\fd\left(\frac{\realdens(x)}{\gendens(x)}\right)\right] $$
Note that, 
$$ \DV\_{\fd}(\realprob || \genprob\_{\gparams}) = 0 \Longleftrightarrow \realprob = \genprob\_{\gparams}$$
Examples: KL, Jensen-Shannon, Squared Hellinger, Pearson \\(\mathcal{X}^2\\) etc.

This is the foundation of many generative learning algorithms. 

#### The Convex Dual representation

Consider the **convex conjugate** of \\(f\\), \\(\fdc\\) and a family of functions, \\(\SN\\). 

<figure>
  <img src="{{ site.img_path }}/BGAN/FD-2.png">
  	<figcaption>Fig 4.</figcaption>
</figure>

Then the dual form of the \\(f\\)-divergence is:
$$\DV\_{\fd}(\realprob || \genprob\_{\gparams}) = \EE\_{\genprob\_{\gparams}}\left[\fd\left(\frac{\realdens(x)}{\gendens(x)}\right)\right] $$
$$= \sup\_{\SN \in \dom{\SN}} \EE\_{\realprob}[\SN(x)] - \EE\_{\genprob}\left[\fdc\left(\SN(x)\right)\right]$$
$$= \EE\_{\realprob}[\SNo(x)] - \EE\_{\genprob}\left[\fdc\left(\SNo(x)\right)\right]$$

However, finding this supremum is hard. So we can instead use a family of neural networks(classifiers), \\(\SN\_{\phi}(x)\\) which can be learnt!.

Hence, we have:
$$\DV\_{\fd}(\realprob || \genprob\_{\gparams})  \geq \EE\_{\realprob}[\SNp(x)] - \EE\_{\genprob}\left[\fdc\left(\SNp(x)\right)\right]$$

## **BGAN For Discrete Data**
Now, we can proceed to show how BGAN can handle discrete data \\TODO

#### Estimating The Likelihood Ratio

Given a perfect discriminator (i.e \\(\SNo\\)), we show in Theorem 1 of our paper that, 
$$ \realdens(x) = (\pd{\fdc}{\SN})(\SN^{\star}(x)) \gendens\_{\gparams}(x) $$

<div class="figurecenter">
	<figure>
	  <img src="{{ site.img_path }}/BGAN/EL-1.png">
	  	<figcaption>Fig 5.</figcaption>
	</figure>
</div>

#### \\(f\\)-Importance Weight Estimator
Now we don't have a perfect discriminator (i.e \\(\SNo\\)), but we do have a sub-optimal discriminator(i.e \\(\SNp(x)\\)).
Let \\(\iw(x) = (\pd{\fdc}{\SN})(\SNp(x))\\) and \\(\beta = \EE\_{\genprob\_{\dparams}}[\iw(x)]\\) be a partition function. Then we can have a \\(\fd\\)-divergence importance weight estimator, \\(\realdensest(x)\\) as:
$$ \realdensest(x) = \frac{w(x)}{\beta} \gendens\_{\gparams}(x) $$

<div class="figurecenter">
	<figure>
	  <img src="{{ site.img_path }}/BGAN/EL-2.png">
	  	<figcaption>Fig 6.</figcaption>
	</figure>
</div>

#### Policy Gradient Based on Importance Sampling
With this importance weight estimator, we have an option for training the generator in an adversarial way with the gradient of the \\(KL\\): 
$$ \nabla\_{\gparams} \DV\_{KL}(\realdensest(x) || \gendens\_{\gparams}) = -\EE\_{\genprob\_{\gparams}}\left[\frac{\iw(x)}{\beta} \nabla\_{\gparams} \log{\gendens\_{\gparams}(x)}\right] $$

This gradient resembles other importance sampling methods for training generative models in the discrete setting. However, the variance of this estimator will be high, as it requires estimating the partition function, \\(\beta\\) (Through say, Monte-Carlo sampling). We address reducing this issue next.

#### Lower-variance Policy Gradient
We can use the expected KL over the conditionals instead!
And so, we have, $$\gendensp(x) = \int\_{\dom{Z}}{\gencondp{x}{z} \prior(z) dz}$$
$$\realcondt{x}{z} = \frac{w(x)}{\alpha(z)} \gencondp{x}{z}$$
We can then derive the normalized weights:
$$ \iwt(\sam{x}{m}) = \frac{\iw(\sam{x}{m})}{\sum\_{m'}{\iw(\sam{x}{m'})}} $$
And the new partition function:
$$ \alpha(z) = \EE\_{\gencondp{x}{z}}[\iw(x)] = \int\_{\dom{X}}{\gencondp{x}{z} \iw(x) dx} $$
So now, the gradient for training the generator becomes:
$$ \nabla\_{\gparams} \EE\_{\prior(z)}[\DV\_{KL}\left(\realcondt{x}{z} \middle || \gencondp{x}{z} \right)] = -\EE\_{\prior(z)}\left[\sum\_m \iwt(\sam{x}{m}) \nabla\_{\gparams} \log{\gencondp{\sam{x}{m}}{z}}\right] $$

<div class="figurecenter">
	<figure>
	  <img src="{{ site.img_path }}/BGAN/VP.png">
	  	<figcaption>Fig 7. </figcaption>
	</figure>
</div>

From figure 7, we can see that this new policy gradient estimator (bold) has lower variance than the original policy gradient estimator (dashed) in estimating \\(2 \times JSD - log 4\\).

This can be implemented in Pytorch as:

{% gist c3e2c95eaf884f464eba513cd2ac2c4a %}


#### REINFORCE BGAN
REINFORCE is a common technique for dealing with discrete data in GANs. The lower-variance policy gradient estimator described above is a policy gradient in the special case that the reward is the normalized importance weights. This reward approaches the likelihood ratio in the non-parametric limit of an optimal discriminator.
And so, we also make a connection to REINFORCE as it is commonly used, with baselines, by deriving the gradient of the reversed KL-divergence:
$$     \nabla\_{\gparams} \EE\_{\prior(z)}\DV\_{KL}\left(\gencondp{x}{z} \middle\|| \realcondt{x}{z} \right) = -\EE\_{\prior(z)}\left[\sum\_m (\log{\iw(\sam{x}{m})} - \log{\beta} + 1) \nabla\_{\gparams} \log{\gencondp{\sam{x}{m}}{z}}\right]
    \nonumber $$

## **BGAN For Continuous Data**

For continuous variables, minimizing the variational lower-bound suffices as an optimization technique as we have the full benefit of back-propagation to train the generator parameters, \\(\gparams\\). However, while the convergence of the discriminator is straightforward, to our knowledge there is no general proof of convergence for the generator except in the non-parametric limit or near-optimal case. What's worse is the value function can be arbitrarily large and negative. Let us assume that \\(\max \SN = M < \infty\\) is unique. As \\(\fdc\\) is convex, the minimum of the lower-bound over \\(\gparams\\) is:
$$ \inf\_{\gparams} \dist{\realprob}{\genprob\_{\gparams}}{\GN\_{\dparams}} = \inf\_{\gparams} \EE\_{\realprob}[\SN\_{\dparams}(x)] - \EE\_{\genprob\_{\gparams}}[\fdc(\SN\_{\dparams}(x))] $$  $$= \EE\_{\realprob}[\SN\_{\dparams}(x)] - \sup\_{\gparams} \EE\_{\genprob\_{\gparams}}[\fdc(\SN\_{\dparams}(x))] = \EE\_{\realprob}[\SN\_{\dparams}(x)] - \fdc(M) $$


The generator objective is optimal when the generated distribution, \\(\QQ_{\gparams}\\), is nonzero only for the set \\(\\{x \mid \SN(x) = M \\}\\).
Even outside this worst-case scenario, the additional consequence of this minimization is that this variational lower-bound can become looser w.r.t. the \\(\fd\\)-divergence, with no guarantee that the generator would actually improve. Generally, this is avoided by training the discriminator in conjunction with the generator, possibly for many steps for every generator update. However, this clearly remains one source of potential instability in GANs.
And so, we can instead aim for the decision boundary and this can improve stability. We observe that for a given estimator \\(\realdensest(x)\\), \\(\gendensp(x)\\) matches when \\(\iw(x) = (\pd{\fdc}{\SN})(\SN(x)) = 1\\). And so, we can define the continuous BGAN objective as:
$$ \hat{\gparams} = \argmin\_{\gparams} (\log{\iw(\GG\_{\gparams}(z))})^2 $$
This objective can be seen as changing a concave optimization problem (which is poor convergence properties) to a convex one.

## **Code**
Our paper presents 10 sets of experiments. The code is available both in [Pytorch](https://github.com/rdevon/cortex2.0) as well as in [Theano](https://github.com/rdevon/BGAN).

## **Conclusion**



