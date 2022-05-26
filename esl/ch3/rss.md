# Residual Sum of Squares

## RSS - Error

&#x20;   Residual Sum of Squares is important. The more strict notation is error, not residual because Error is the random variable but residual is a constant after fitted.

![](<../../.gitbook/assets/image (16).png>)

$$
f(X)=\beta_0+X_1\beta_1+X_2\beta_2 \\
RSS(\beta)=(\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta) \\
\frac{\partial RSS}{\partial \beta}=-2\mathbf{X}^T(\mathbf{y}-\mathbf{X}\beta)\\
\frac{\partial^2 RSS}{\partial \beta \partial \beta^T}=-2\mathbf{X}^T\mathbf{X}
$$

$$
\mathbf{X}^T(\mathbf{y}-\mathbf{X}\beta)=0\\ \hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \\
\hat{y}=\mathbf{X}\hat{\beta}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}=\mathbf{H}\mathbf{y}
$$

👀 Geometrical view

![](<../../.gitbook/assets/image (17).png>)

$$Y$$ is the projection onto the column space of $$X$$. This is because $$\mathbf{H}$$ is the projection matrix that has symmetric / idempotent properties. $$\mathbf{H}$$ is called as hat matrix (giving $$y$$ a hat)



|   Q   | A (Under the condition that $$\hat{\beta}=\hat{\beta}^{LS}$$) |
| :---: | :-----------------------------------------------------------: |
|  What |                         $$\mathbf{y}$$                        |
| Where |                  Col Space of $$\mathbf{X}$$                  |
|  How  |                           Projection                          |

&#x20; $$\varepsilon \perp x_i$$, because $$\epsilon=y-\hat{y}$$. If we estimate $$\beta$$in other methods with exclusion of $$LSM$$ method, the form $$\hat{y}=\beta_0+X_1\beta_1+X_2\beta_2$$ still remains. $$\hat{y}$$ is interpreted still as the vector on $$\mathbf{col(X)}$$. However, In this case $$\hat{y}$$ is not a projected vector so that the residual and variables are not orthogonal.

