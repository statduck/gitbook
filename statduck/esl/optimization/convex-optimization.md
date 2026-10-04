# Convex Optimization

**Definition**

$$
minimize \ f_0(x) \\ subject \; to \; f_i(x)\leq 0, \; h_i(x)=0
$$

* The objective function $$f_0(x)$$ must be convex
* The inequality constraint functions $$f_i(x)$$must be convex
* The equality constraint functions $$h_i(x)$$must be affine



**Optimal Point**

Global Optimum: $$f_0(x^*) \leq f_0(x)$$

Local Optimum: $$x^*$$is an optimal point iff there exists $$r>0$$ $$x \in \{x| ||x-x^*||\leq r\}, \;  f_0(x^*) \leq f_0(x)$$

Any local optimum becomes global optimum in convex optimization problem.



**First order optimality condition**

$$
\nabla f_0(x)^T(y-x) \geq 0\; \forall y\in X
$$



**KKT Optimality conditions**

* $$\nabla f_0(x^*)+\Sigma^m_{i=1} \lambda _i^*\nabla f_i(x^*)+\Sigma^p_{i=1}u_i^*\nabla h_i(x^*)=0$$​
* $$\lambda _i^*f_i(x^*)=0$$
* $$f _i(x^*) \leq 0$$
* $$h_i(x^*)=0$$
* $$\lambda _i^* \geq 0$$







