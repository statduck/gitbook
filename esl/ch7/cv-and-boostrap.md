# CV and Boostrap

### Cross validation



$$
CV(\hat{f})=\frac{1}{N}\sum^N_{i=1}L(y_i, \hat{f}^{-\kappa(i)}x_i)) \\
CV(\hat{f},\alpha)=\dfrac{1}{N}\sum^N_{i=1}L(y_i,\hat{f}^{-\kappa(i)}x_i, \alpha))
$$



&#x20;When f is linear, it follows that

$$
\frac{1}{N}\sum^N_{i=1}[y_i-\hat{f}^{-i}(x_i)]^2=\frac{1}{N}\sum^N_{i=1}[\frac{y_i-\hat{f}(x_i)}{1-S_{ii}}]^2
$$

&#x20;In left side we pick out one value of our data, but in right side there is no picking out. It means without actual removing process we can calculate the value of left side.

$$
\sum^N_{i=1}(y_i-\hat{f}(x_i))^2\leq \sum^N_{i=1}(y_i-{f}(x_i))^2 \\
\hat{f}^{(k)}=argmin_{f}{\sum^n_{i \neq k} (y_i-f(x_i))^2} \\
\sum^N_{i \neq k}(y_i-\hat{f}^{(k)}(x_i))^2\leq \sum^N_{i \neq k}(y_i-{f}(x_i))^2 \\
$$



### Boostrap Methods

Training point $$Z=(z_1,\dots,z_N)$$. We randomly pick data set in replacement. We want to estimate some aspects of the distribution of S(Z).&#x20;

$$
\hat{Var}[S(Z)]=\frac{1}{B-1}\sum^B_{b=1}(S(Z^{*b})-\bar{S}^*)^2
$$



Our estimate is below:

$$
\hat{Err}_{boot}=\frac{1}{B}\frac{1}{N}\sum^B_{b=1}\sum^N_{i=1}L(y_i,\hat{f}^{*b}(x_i))
$$

The bootstrap datasets are acting as the training samples, but the original training set is acting as the test sample, and these two samples have observations in common. This overlap can make **overfit predictions**.



$$
\hat{Err}^{(1)}=\frac{1}{N}\sum^N_{i=1}\frac{1}{|C^{-i}|}\sum_{b\in C^{-i}}L(y_i,\hat{f}^{*b}(x_i)) \\
\hat{Err}^{.632}=.368\bar{err}+.632\hat{Err}^{(1)}
$$

