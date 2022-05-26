# Linear method: Regression

## Variable Selection

This part is of selecting necessary independent variables. With several unnecessary variables, model has low predictive power and explanatory power. There are three ways to pick up the valuable variables:

* Best Subset Selection
* Forward & Backward Stepwise Selection
* Forward Stagewise Regression

#### Best Subset Selection

$$
k \in \{0,1,2,...,p\}
$$

It is the method that every possible regression fitting by Subset size k. ( The optimal value in subset size 1 doesn't have to be optimal in size 2.)

![](<../../.gitbook/assets/image (14).png>)



#### Forward & Backward Stepwise Selection

Forward: Starting from zero model(Only intercept term exists), we put variables into our model one by one. This process could be faster in QR decomposition.

Backward: Starting from Full model, we remove variables from our model one by one.

![](<../../.gitbook/assets/image (15).png>)



QR decomposition can lower our computational cost.

$$
X=QR, \quad X \in \mathbb{R}^{n\times p}  \; Q\in \mathbb{R}^{n \times p} \\
Q^T[X \quad y]=
\begin{bmatrix}
R \quad z \\ 0 \quad \rho \\ 0 \quad 0
\end{bmatrix}
\\
\hat{\beta}=(X^TX)^{-1}X^Ty=(R^TQ^TQR)^{-1}R^TQ^Ty=R^{-1}Q^Ty=R^{-1}z\\



RSS(\hat{\beta})=||y-X\hat{\beta}||^2=z+\rho-y
$$

$$
X_s=XS , \quad \beta_S=S^T\beta \\
y=X_S\beta_S+\epsilon, \quad z=RS\beta_S+\xi \\
\hat{\beta}=R_s^{-1}z_s, \quad 
 RSS(\beta_S)=RSS(\hat{\beta})+\rho^2_S
$$

#### Forward-Stagewise Regression

It is the Forward-stepwise regression with more constraints.

1\) We pick the variable most correlated to the variable in our fitted model.

2\) Let this variable be a target and the residual be an explanatory variable. Calculate the coefficient.

3\) Adding this coefficient to the coefficient of our existing model.



## Shrinkage

&#x20; Variable Selection is way to choose values in discrete way. The decision is only zero or one(removing or putting in). Because of this property, model variance would be increased. Shrinkage method is more free to variance because it choose variables in continuous way.

**Lasso**

$$L_1$$penalty is imposed.

$$
\hat{\beta}^{lasso}=argmin_\beta\sum^N_{i=1}(y_i-\beta_0-\sum^p_{j=1}x_{ij}\beta_j)^2, subject\;to \sum^p_{j=1} |\beta_j| \leq t \\
$$

&#x20;

**Ridge**&#x20;

$$
\hat{\beta}^{ridge}=argmin_\beta\sum^N_{i=1}(y_i-\beta_0-\sum^p_{j=1}x_{ij}\beta_j)^2, subject\;to \sum^p_{j=1} \beta_j^2 \leq t \\
RSS(\lambda)=(y-X\beta)^T(y-X\beta)+\lambda \beta^T\beta \\
\hat{\beta}^{ridge}=(X^TX+\lambda I)^{-1}X^Ty
$$

$$
X=UDV^T \\
X\hat{\beta}^{ls}=X(X^TX)^{-1}X^Ty = UU^Ty \\
$$

$$
\begin{split}
X\hat{\beta}^{ridge}= {} & X(X^TX+\lambda I)^{-1}X^Ty \\
= & UD(D^2+\lambda I)^{-1}DU^Ty \\
= & \sum^p_{j=1} u_j \frac{d_j^2}{d_j^2+\lambda} u_j^Ty

\end{split}
$$



## Algorithm

By adding some processes, we can make a more elaborate model. Many algorithms make emphasize on the relationship between X variables and residual.&#x20;

$$\hat{y}$$is on $$col(X)$$. With the view of linear combination, it is expressed as $$\hat{y}=\hat{\beta_1}X_1+\hat{\beta_2}X_2+...+\hat{\beta_p}X_p$$. The residuals is $$\varepsilon = y-\hat{y}$$.

Our aim is to minimize $$||\varepsilon||^2=\varepsilon \cdot \varepsilon$$.

What is the meaning of minimizing in terms of variable selection?

$$\begin{split} min(\varepsilon \cdot \varepsilon) & =  {}  min((y-\hat{y}) \cdot (y-\hat{y})) \\ & =  min((y-(\hat{\beta_1}X_1+\hat{\beta_2}X_2))\cdot(y-(\hat{\beta_1}X_1+\hat{\beta_2}X_2))) \  \\ & =min((\hat{\beta_1}X_1 - y)\cdot \hat{\beta_2}X_2)  \end{split}$$

$$X_1$$is an existing variable and $$X_2$$is a new variable.In this case, $$y$$and $$\hat{\beta_1}X_1$$are fixed and $$\hat{\beta_2}X_2$$is an unfixed vector. Let's assume the norm of$$\hat{\beta_2}X_2$$is fixed, and the direction is only changed. The important thing is the relationship between existing residual and $$X_2$$. Digging into the relationship between them is the key to decide whether the new variable would be put in or not. Many methods have been made from it.



LAR, PCR and PLS are representative examples. This algorithm is the method to make new features and the feature selection proceeds in such a way to minimize the error.

### LAR(Least Angle Regression)

$$
y=\bar{y}+r
$$

&#x20;   In this situation, we find $$\beta_j$$ of which $$x_j$$ has a high correlation with $$r$$&#x20;

$$
y=\hat{\beta}_0+\hat{\beta}_1X_1+r
$$

$$\hat{\beta}_1\in[0,\hat{\beta}_1^{LS}], \; s.t. \dfrac{x_j\cdot r}{||x_j||}<\dfrac{x_k\cdot r}{||x_k||}$$

&#x20;   Our fitted beta move from $$0$$ to $$\hat{\beta}_1^{LS}$$ until the correlation between another input variable and error becomes bigger. **This approach makes beta come closer to the beta of least square, and the correlation between all input variables and error get reduced**. This method can mine as many information as possible from our data. In the situation that $$\hat{\beta}_j$$ can't get close to $$\hat{\beta}_j^{LS}$$, $$\hat{\beta}_j$$ is fixed as $$0$$ like the Lasso regression.

