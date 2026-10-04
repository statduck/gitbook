# Basis Expansions & Regularization

We can't assure our function is linear.

To deal with non-linear problem, we can use transformed X instead of original X.

## Basis Expansions and Regularization

$$
f(X)=\sum^M_{m=1}\beta_mh_m(X)
$$

&#x20;The basis function, f(X), is linear on h even though $$h(X)$$is non linear&#x20;

| Form                                    |                    |
| --------------------------------------- | ------------------ |
| $$h_m(X)=X_m$$                          | Basic linear model |
| $$h_m(X)=X_j^2 \; or \; h_m(X)=X_jX_k$$ | Polynomial model   |
| $$h_m(X)=log(X_j), \sqrt{X_j}$$         | Log model          |
| $$h_m(X)=I(L_m\leq X_k \leq U_m)$$      | Range model        |

When we add a third term into model, we have to add second, first, and constant term into this model. So polynomial model has the flaw of high dimension(too many independent variables.)

If you want to locally analysis the model, we have to put a range variable into this model.

To reduce the number of basis function, there are following three methods:

| Methods        | Example                                        |
| -------------- | ---------------------------------------------- |
| Restriction    | limited to additional model                    |
| Selection      | Select only significant variables on the model |
| Regularization | Constrained coefficients                       |

## Natural Cubic Spline

$$
N_1(X)=1,\; N_2(X)=X, \; N_{k+2}(X)=d_k(X)-d_{K-1}(X) \\
d_k(X)=\frac{(X-\xi_k)^3_+-(X-\xi_K)^3_+}{\xi_K-\xi_k}
$$

$$
\hat{\theta}=(N^TN+\lambda\Omega_N)^{-1}N^Ty \\
\hat{f}(x)=\sum^N_{j=1}N_j(x)\hat{\theta}_j
$$

```python
class spline:
    def __init__(self, x, y):
        x = np.array([[1]*x.shape[0], x, np.power(x,2), np.power(x,3)])
        b1 = min(x[1]) + (max(x[1])-min(x[1]))/3
        b2 = min(x[1]) + 2*(max(x[1])-min(x[1]))/3

        x1 = np.append(x, [np.power(x[1],3), np.power(x[1],3)], axis=0)
        x1 = np.transpose(x1)[x[1]<b1]
        x2 = np.append(x, [np.power(x[1],3)], axis=0)
        x2 = np.transpose(x2)[(b1<=x[1])&(x[1]<b2)]
        self.x = np.transpose(x)
        self.y = y
        self.x1 = x1
        self.x2 = x2
        
    def training(self):
        x = self.x # col vec expression
        y = self.y
        x1 = self.x1
        x2 = self.x2
        
        xt = np.transpose(x)
        beta = np.linalg.inv(xt@x)@xt@y
       #  잔차에다가 또 피팅해주는 방식이에요. y값만 바뀌는 거겠죠

        y_fit = y-(x@beta)
        x1t = np.transpose(x1)
        beta1= np.linalg.inv(x1t@x1 + np.diag([0.01]*x1.shape[1]))@x1t@y_fit
        x2t = np.transpose(x2)
        beta2 = (1/(x2t@x2))*(x2t@y_fit)

        return(np.array([beta1,beta2]))
    def prediction(self, X_test):
        X_test = np.insert(X_test,0,1,axis=1)
        y_pred = (X_test@self.beta > 0).astype('uint8')
        return(y_pred)

```
