# Kernel method

Kernel method is to estimate the function using kernel. It estimates the function value by investigating near x values. The near point is defined by the distance, so it gives a weight by the distance.

## ✏️ knn as a kernel method

KNN can be viewed as a kernel method because it calculates distance in choosing nearest k points. &#x20;

$$
\hat{f}(x)=Ave(y_i|x_i \subset N_k(x))
$$

&#x20;$$N_k(x)$$is the set containing nearest k points of euclidean distance. In this case, this function is not continuous because the support isn't continuous. KNN finds nearest k points, but this is based on the number of points(k) so knn differs from a metric based way.

|             | Metric                   | knn                      |
| ----------- | ------------------------ | ------------------------ |
| Bias        | constant                 | Inverse of local density |
| Var         | Inverse of local density | constant                 |
| Tied values |                          | Additional weights       |

* Metric: Defining the distance -> containing the points
* KNN: Defining the points&#x20;

## ✏️ Another method

Nadaraya-Watson kernel weighted average with the Epanechnikov quadratic kernel. &#x20;

$$
\hat{f}(x_o) = \dfrac{\Sigma^N_{i=1} K_\lambda(x_0,x_i)y_i}{\Sigma^N_{i=1} K_\lambda(x_0,x_i)}, \quad K_\lambda(x_o,x)=D(\dfrac{|x-x_o|}{\lambda})
$$

$$
D(t) = \begin{cases} \frac{3}{4}(1-t^2) & \quad if |t|\leq1, \\ 0 & otherwise. \end{cases}
$$

&#x20;We consider more values when lambda becomes bigger. In this case, distance is defined as a quadratic form and lambda has a role in scaling factor. This scaling factor determines the distance between solution in quadratic equation.

![](../../.gitbook/assets/ch6\_1.png)

&#x20;We just can use distance function for kernel, why do we have to make a composite function by calling D function. When lambda is fixed,  $$K(x,x_0)=D(f(x,x_0))$$.&#x20;

* D(t): Density function
* f: Distance function

&#x20;D(t) is a density function, and the sum of D in support t has to be 1. So 3/4 is multiplied by D(t).&#x20;

&#x20;The interesting thing is that we doesn't want to get the density for one point, but the density for the distance between two points. The support of probability set function is a set, and the distance becomes set in this case.&#x20;

&#x20;When x = x0, the x maximizing density would be mode. When two points get closer, the density becomes bigger. It means that the closer point of x from fixed x0 has more density.&#x20;



> More generally, kernel can be defined as below:

$$
K_\lambda(x_o,x)=D(\dfrac{|x-x_o|}{h_\lambda(x_0)}), \quad in \; knn \; h(k(x_0))=|x_0-x{[k]}|
$$

$$
tricube:D(t) = \begin{cases} (1-|t|^3)^3 & \quad if |t|\leq1, \\ 0 & otherwise. \end{cases}
$$

$$
Gaussian:D(t) =\phi(t)
$$

