---
description: Centered on normal distribution
---

# Two parameter model

### Background knowledge about Distribution

$$
\eta \sim \Gamma^{-1}(\alpha,\beta)  \\
p(\eta|\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}(\frac{1}{\eta})^{\alpha+1}e^{-\frac{\beta}{\eta}} \\
Mean: \; \frac{\beta}{\alpha-1}, Var: \; \frac{\beta^2}{(\alpha-1)^2(\alpha-2)}
$$

$$
\theta \sim \chi^2(\nu)\equiv \Gamma(\nu/2,1/2)\\ \theta \sim \chi^{-2}(\nu,\tau^2) \equiv \Gamma^{-1}(\nu/2,\nu\tau^2/2)
$$

## One parameter

### Normal model with known mean

$$
y|\sigma^2 \sim N(\mu, \sigma^2) \\
\sigma^2 \sim \chi^{-2}(\nu_0, \sigma^2_0) \\
\sigma^2|y \sim \chi^{-2}(\nu_n,\sigma^2_n)
$$

$$
\nu_n=\nu_0+n \\
\sigma^2_n = \frac{\nu_0\sigma^2_0+ns(y)}{\nu_0+n} \\
f(y|\sigma^2)=h(y)c(\sigma^2)exp(\sigma^2 s(y))
$$

## Two parameter

### Normal data with a conjugate prior

$$
p(y|\mu,\sigma^2) \propto \sigma^{-n}exp(-\frac{1}{2\sigma^2}\Sigma(y_i-\mu)^2) \\

p(\mu, \sigma^2)=p(\mu|\sigma^2)p(\sigma^2) \propto \sigma^{-1}(\sigma^2)^{-(\frac{\nu_0}{2}+1)}exp[-\frac{1}{2\sigma^2}(\nu_0\sigma^2_0+\kappa_0(\mu_0-\mu)^2)] \\ Ninv\chi^2(\mu_0,\sigma^2_0/\kappa_0;\nu_0,\sigma^2_0) 
 \\
\mu|\sigma^2 \sim N(\mu_0, \sigma^2/\kappa_0) \\
\sigma^2 \sim \chi^{-2}(\nu_0,\sigma^2_0)
$$

$$
p(\mu,\sigma^2|y) \sim Ninv\chi^2(\mu_n,\frac{\sigma^2_n}{\kappa_n};\nu_n,\sigma^2_n) \\
\mu_n=\frac{\kappa_0}{\kappa_0+n}\mu_0+\frac{n}{\kappa_0+n}\bar{y} \\
\kappa_n=\kappa_0+n \\
\nu_n=\nu_0+n
$$



### Example

![](<../.gitbook/assets/image (11).png>)

(a) Give your posterior distribution for $$\theta$$

$$
y|\theta \sim N(\theta, 20^2) \\
\theta \sim N(180, 40^2) \\
\theta|y \sim N(\mu, \tau_n^2) \\
\tau_n^2=\frac{1}{1/\tau_0^2 + n/\sigma^2}=\frac{1}{1/1600+n/400}=\frac{1600}{1+4n} \\
\mu_n=\tau_n^2(\frac{1}{\tau^2_0}\mu_0+\frac{n}{\sigma^2}\bar{y})=\frac{180+600n}{1+4n}
$$

(b) Give a posterior predictive distribution for $$\tilde{y}$$

$$
E(\tilde{y}|y)=E(E(\tilde{y}|\mu)|y)=E(\mu|y)=\mu_n \\
V(\tilde{y}|y)=E(V(\tilde{y}|\mu)|y)+V(E(\tilde{y}|\mu)|y)=\sigma^2 + \tau^2_n
$$

$$
\tilde{y}|y \sim N(\mu_n, 400+\tau^2_n)
$$

(c) 95% posterior interval for $$\theta$$ and posterior predictive interval for $$\tilde{y}$$(n=10)

95% posterior interval for $$\theta|y$$

$$
(\mu_n-1.96\frac{\tau_n}{\sqrt{10}},\mu+1.96\frac{\tau_n}{\sqrt{10}}) \\
\because \theta|y \sim N(\mu_n,\tau^2_n)
$$

In the same way, we can also get posterior predictive interval with the distribution in (b)



(d) Do the same for n=100&#x20;



