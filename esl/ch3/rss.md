# Residual Sum of Squares

## RSS - Error

&#x20;   Residual Sum of Squares is important. The more strict notation is error, not residual because Error is the random variable but residual is a constant after fitted.

![](<../../.gitbook/assets/image (151).png>)

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

![](<../../.gitbook/assets/image (166).png>)

$$Y$$ is the projection onto the column space of $$X$$. This is because $$\mathbf{H}$$ is the projection matrix that has symmetric / idempotent properties. $$\mathbf{H}$$ is called as hat matrix (giving $$y$$ a hat)



<table data-header-hidden><thead><tr><th width="150" align="center">Q</th><th align="center">A (Under the condition that )</th></tr></thead><tbody><tr><td align="center">Q</td><td align="center">A (Under the condition that <span class="math">\hat{\beta}=\hat{\beta}^{LS}</span>)</td></tr><tr><td align="center">What</td><td align="center"><span class="math">\mathbf{y}</span></td></tr><tr><td align="center">Where</td><td align="center">Col Space of <span class="math">\mathbf{X}</span></td></tr><tr><td align="center">How</td><td align="center">Projection</td></tr></tbody></table>

&#x20; $$\varepsilon \perp x_i$$, because $$\epsilon=y-\hat{y}$$. If we estimate $$\beta$$in other methods with exclusion of $$LSM$$ method, the form $$\hat{y}=\beta_0+X_1\beta_1+X_2\beta_2$$ still remains. $$\hat{y}$$ is interpreted still as the vector on $$\mathbf{col(X)}$$. However, In this case $$\hat{y}$$ is not a projected vector so that the residual and variables are not orthogonal.

