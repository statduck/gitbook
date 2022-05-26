# Conditional Gradient

Conditional Gradient\[Frank-Wolfe Method]

**The problem :**&#x20;

$$
minf(x) \\ subject \; to \\ x  \in P
$$

​Where $$f$$is convex on the bounded polygonal region $$P$$

How can we determine the search direction $$\mathbf{d}_k$$in this problem? It is connected to a linear problem.

$$
Z_k(y):=f(x_k)+\nabla f(x_k)^T(y-x_k) \\  min \ z_k(y)=\nabla f(x_k)^T(y-x_k) \\ subject \; to \\ y \in P
$$

We first look into $$(y-x_k)$$.&#x20;

Suppose $$y_k$$is a solution of this problem.

* $$y_k$$ is an extreme point of the polygon P and hence, since P is convex, the line joining $$y_k$$and $$x_k$$is contained in $$P$$and so the vector $$y_k-x_k$$is a feasible direction.
* Because of the convexity of $$f$$, $$\nabla f(\bar{x})^T(y-\bar{x}) \geq 0  \;\; \forall y\in P$$. It implies that this direction is also a descent direction.
* $$z_k(y_k)\leq z(x_k) = \nabla f(x_k)^T(x_k-x_k)=0$$ , so $$z_k(y_k)=0 \; or \; z_k(y_k)<0$$









