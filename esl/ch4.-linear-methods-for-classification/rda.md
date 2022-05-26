# RDA

### ✏️ Definition

RDA(Regularized Discriminant Analysis) is the combinational model between LDA and QDA.

$$
\hat{\Sigma}_k(\alpha)=\alpha\hat{\Sigma}_k+(1-\alpha)\hat{\Sigma}, \quad \alpha \in[0,1]
$$

Vector internal division. This is most common in combinational model. $$\hat{\Sigma}$$is a pooled covariance matrix from LDA.  If we replace $$\hat{\Sigma}$$as $$\hat{\Sigma}(\gamma)$$,  this also can be changed into  $$\hat{\Sigma}(\alpha,\gamma)$$. For generalization, parameter is just added.

$$
\hat{\Sigma}(\gamma)=\gamma\hat{\Sigma}+(1-\gamma)\hat{\sigma}^2I, \quad \gamma \in[0,1]
$$

$$\hat{\sigma}^2I$$ also can be changed into $$diag(\hat{\Sigma}), \hat{\Sigma}/p,...$$&#x20;



![](<../../.gitbook/assets/image (22).png>)
