# Novelty Detection

### PreFace

&#x20;Novelty Detection is the detection for whether a new data point is an outlier, and outlier detection is the detection for whether a train data is an outlier. In other words, we find the most concentrated area in outlier detection.

![](<../../.gitbook/assets/image (142).png>)

[Reference](https://scikit-learn.org/stable/modules/outlier\_detection.html)

&#x20;For outlier detection, we first fit density. We define the data point as outlier if it has in low density. $$density \; function \leq t$$&#x20;

&#x20;We also can find the boundary between inlier and outlier. To do this, we have to find the smallest ball such that it includes all data points. This problem is converted into the problem finding a center c and radius r as below.

$$
min_{r,c} \; r \\ s.t.  \; (x_i-c)^T(x_i-c)\leq r, \; r \geq 0
$$



### Minimum enclosing ball

$$
L(r,c,\alpha)=r+\sum^m_{i=1}\alpha_i((x_i-c)^T(x_i-c)-r), \; \alpha_i \geq 0
$$

$$
\dfrac{\partial L}{\partial r}=1-\Sigma \alpha_i=0  \\ \dfrac{\partial L}{\partial c}=\Sigma \alpha_i(-2x_i+2c)=0
$$

&#x20;This leads to $$\Sigma \alpha_i =1 , \; c=\Sigma \alpha_ix_i$$

$$
\begin{align} L & =r+\Sigma\alpha_i((x_i-c)^T(x_i-c)-r) \\ &= r+\Sigma\alpha_ix_i^Tx_i-\Sigma \alpha_ix_i^Tc-\Sigma \alpha_ic^Tx_i+\Sigma \alpha_ic^Tc-\Sigma \alpha_ir \\ & = \Sigma \alpha_i x_i^Tx_i -\Sigma_i\Sigma_j \alpha_i \alpha_j x_i^Tx_j \quad \quad \quad \quad s.t.\Sigma \alpha_i=1, \alpha_i \geq 0 \end{align}
$$

​Now, the problem becomes a quadratic problem with simple constraints (dual problem)

$$
max_ag(a)=b^T\alpha-\alpha^TA\alpha \quad\quad\quad\quad s.t. \Sigma \alpha_i=1, \; \alpha_i \geq 0  \\ A_{ij}=x_i^Tx_j, \; b_i=x_i^Tx_i, \; \alpha
$$

When $$\alpha_i$$ becomes 0, the points will be inside the circle. When $$\alpha_i$$ is bigger than 0, the points will be exactly on the boundary. Thus, the solution in $$\alpha$$ is very sparse.





