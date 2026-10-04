# Curse of dimensionality

&#x20;   There are three problems in high dimension

* First, More length is needed to capture same rate of data.
* Second, All sample points are close to an edge of the sample.
* Third, We need much more sample in high dimension to capture same percentage of data in low dimension.

<mark style="background-color:yellow;">**✏️ First Problem**</mark>

&#x20;   Let's think about a unit hypercube in p dimension, $$\{x|\; x_i<1, x \in \mathbb{R}^p\}$$. We assume the density of x is uniformly distributed.

&#x20;The average length of one side in p dimension is that: $$e_p(r)=r^{1/p}$$

`There is a subcube in hypercube occupying r% of all data.`

* In$$\mathbb{R}^1$$, one side length of a subcube is $$1/4$$ for $$1/4$$ data.
* In$$\mathbb{R}^2$$, one side length of a subcube is $$1/2$$for $$1/4$$data.
* In$$\mathbb{R}^3$$, one side length of a subcube is $$1/\sqrt{2}$$ for $$1/4$$data.

&#x20;   When dimension becomes higher, an amount of information becomes increased. More information means longer length of one side for same rate of data.

$$
e_{10}(0.01)=0.63
$$

&#x20;   In $$\mathbb{R}^{10}$$, we need the length 0.63 to capture 1% of data.



✏️ **Second Problem**

&#x20;    `All sample becomes close into an edge.`&#x20;

$$
\omega_p=\frac{\pi^{p/2}}{(p/2)!} \\ 
F(x)=x^p, \; 0\leq x \leq 1. \\ f(x)=px^{p-1}, \; 0\leq x \leq 1. \\ g(y)=n(1-y^p)^{n-1}py^{p-1} \\ G(y)=1-(1-y^p)^n
$$

$$
d(p,N)= (1-\frac{1}{2}^{1/N})^{1/p}
$$

&#x20;   Find the y value satisfying $$G(y)=1/2$$. The distance the closest point from origin.



✏️ **Third Problem**

&#x20;   The distance becomes very short when the number of sample is large and the dimensionality is high. For our fixed sample, the sampling density is proportional to $$N^{1/p}$$

* To maintain the same density, we need one hundred samples in one dimension. However, we need $$100^{10}$$samples in 100 dimension to maintain the same density.&#x20;

&#x20;   In this context, sampling density means the number of sample in unit length.
