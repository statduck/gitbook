# Ensemble

## ECOC(Error Correcting Output Codes)

**Multiclass classification**

* **OVR(One vs Rest):** Red VS \[Blue, Green] - Logistic regression
* **OVO (One vs One):** Red VS Blue - SVM
* **ECOC:** Random code assignment&#x20;

&#x20;   In a multiclass classification, we need to encode a categorical variable into several dummy variables. There are three ways to encoding, and ECOC is introduced as the way for adding randomness.

&#x20;   Random code assignment worked as well as the optimally constructed error-correcting codes. The idea is that the redundant "error correcting" bits allow for some inaccuracies, and can improve performance.

![](<../../.gitbook/assets/image (118).png>)

1. Learn a separate classifier for each of the $$L=15$$
2. At a test point $$x$$, $$\hat{p}_l(x)$$is the predicted probability of a one for the $$l_{th}$$ response.
3. $$\delta_k(x)=\sum^L_{l=1} |C_{kl}-\hat{p}_l(x)|$$

## <mark style="background-color:yellow;">Boosting and Regularization Paths</mark>

**Forward-stagewise(FS) regression**

&#x20;   Forward-stagewise regression is a constrained version of forward-stepwise regression. At each step, this algorithm identifies the most correlated variable with the current residual. You can check the specific algorithm in page 5 in the research below.&#x20;

{% embed url="https://arxiv.org/pdf/0705.0269.pdf" %}

&#x20;   This research shows the algorithm for forward-stagewise regression.&#x20;



**Generalized Additive Models**

&#x20;   Usually effects are often not linear, $$f$$ might be a non-linear function.

$$
E(Y|X_1,...,X_p)=\alpha+f_1(X_1)+f_2(X_2)+\cdots f_p(X_p)
$$

**Tree**

$$
f(x)=\sum^M_{m=1} c_m I(x\in R_m)
$$

**Penalized regression**

<mark style="color:orange;">Penalty approach</mark>&#x20;

$$
f(x) = \sum^K_{k=1} \alpha_k T_k(x), \; K=card(T),\;T=\{T_k\} \\ \min_\alpha \Big\{  \sum^N_{i=1}  \Big( y_i-\sum^K_{k=1} \alpha_k T_k(x_i) \Big)^2+\lambda \cdot J(\alpha)\Big \} \\ J(\alpha)=\sum^K_{k=1}|\alpha_k|^2 \;(Ridge), \;J(\alpha)=\sum^K_{k=1}|\alpha_k| \;(Lasso)
$$

&#x20;   $$T=\{T_k\}$$ is the dictionary of all possible $$J$$terminal node regression trees. $$T$$ can be realized on the training data as basis functions in $$\mathbb{R}^p$$. Also, all coefficients for each tree are estimated by least squares.&#x20;

&#x20;  However, when the number of basis functions become so large, solving the optimization problem with the lasso penalty is not possible. In this case, following forward stagewise approach can be used by approximation of the lasso penalty approach.



<mark style="color:blue;">Forward Stagewise approach(FS)</mark>

1. Initialization $$\hat{\alpha}_k=0, \;k=1,...,K.$$
2. $$(a) \; (\beta^*,k^*)=argmin_{\beta, k}\sum^N_{i=1}\Big(y_i-\sum^K_{l=1} \hat{\alpha}_lT_l(x_i)-\beta T_k(x_i)\Big)^2 \\ (b) \; \hat{\alpha}_{k^*}\leftarrow \hat{\alpha}_{k^*} + \varepsilon \cdot sign(\beta^*)$$
3. $$f_M(x)=\sum^K_{k=1} \hat{\alpha}_kT_k(x)$$



&#x20;   This algorithm leverages the forward-stagewise algorithm. As like the regression, it adopts the tree most correlated with the current residual. The iteration number $$M$$ is inversely related to $$\lambda$$, when $$\lambda=\infty$$ the forward stagewise algorithm is on the initialization stage that every coefficients are zero.($$M=0$$)&#x20;

&#x20;   $$|\hat{\alpha}_k(\lambda)|<|\hat{\alpha}_k(0)|$$ and also $$|\hat{\alpha}_k(M)|<|\hat{\alpha}_k(0)|$$. When $$\lambda=0$$, the regularization term disappears, so the solution becomes a least square solution.&#x20;



![  ](<../../.gitbook/assets/image (92).png>)

&#x20;   When all of the basis functions $$T_k$$ are mutually uncorrelated, FS shows exactly the same solution with lasso for bound parameter $$t=\sum_k |\alpha_k|$$ ( Even if these functions are not uncorrelated, when $$\hat{\alpha}_k(\lambda)$$ is a monotone function of $$\lambda$$, FS also becomes same with lasso regression.) The regularization term $$\lambda$$ is inversely proportional to the Lagrange constant $$t$$. $$M =250, \;\varepsilon=0.01$$ in the right panel.&#x20;





**"Bet on Sparsity" Principle**

&#x20;   Usually, the shrinkage with $$L_1$$penalty is better suited to sparse situations, where there are few basis functions with nonzero coefficients.&#x20;

* Dense Scenario: 10,000 data points and a linear combination of million trees with coefficients from a Gaussian distribution. → Ridge works better than Lasso

![](<../../.gitbook/assets/image (91).png>)

* Sparse Scenario: 10,000 data points and a linear combination of 1,000 tress with nonzero coefficients. → Lasso works well.



&#x20;   The degree of sparseness is determined by the true target function and the chosen dictionary $$T$$. Noise-to-signal ratio(NSR) can also be a key in determining the sparseness. Because larger training sets allow the estimation for coefficients with smaller standard errors. When the size of dictionary becomes increased, it leads to a sparser representation for our function and higher variance in searching.&#x20;



![](<../../.gitbook/assets/image (116).png>)

&#x20;   In this example, NSR is defined as $$Var(Y|\eta(X))/Var(\eta(X))$$. The nominator is the variance of $$Y$$(unexplained part), and the denominator is the variance of our model. The bigger NSR is, the bigger the rate of unexplainable error is. It has been known as that lasso works well for sparse setting.

**>>** [**Reference**](https://www.stat.cmu.edu/\~ryantibs/statml/lectures/sparsity.pdf) **<<**

****

****

**Regularization Paths, Over-fitting and Margins**

&#x20;   ****   &#x20;

![](<../../.gitbook/assets/image (164).png>)

&#x20;   ****    Lasso suffers somewhat from the multi-collinearity problem(you can check the reason in Ex. 3.28). Because in the exercise 3.28, when lasso has the exact same copy $$X_j^*=X_j$$, the coefficients for $$X_j^*,X_j$$ become $$a/2, a/2$$ which is the a half of the original coefficient $$a$$.

![ESL Solution](<../../.gitbook/assets/image (159).png>)

****

****

**Monotone version of the lasso**

**>>** [**Reference**](https://arxiv.org/pdf/0705.0269.pdf) **<<**

&#x20;   We create an expanded data matrix $$\tilde{X}=[X:-X]$$

$$
\min_{\beta_0,\beta_j^+,\beta_j^-}\sum^N_{i=1} \Big(y_i-\beta_0-\Big[\sum^p_{j=1}x_{ij}\beta_j^+-\sum^p_{j=1}x_{ij}\beta_j^- \Big]\Big)^2 \\ s.t. \; \beta_j^+,\beta_j^- \geq 0 \; \forall j \;and \; \sum^p_{j=1}(\beta_j^++\beta_j^-)\leq s
$$

&#x20;   In this setting,&#x20;

![Hastie & Taylor & Tibshirani  \&Walther, Forward stagewise regression and the monotone lasso, Electronic Journal of Statistics Vol1. (2007) 1-29](<../../.gitbook/assets/image (152).png>)

&#x20;       The monotone lasso coefficient path $$\beta(l)$$ for a dataset $$\tilde{X}=\{X,-X\}$$ is the solution to the different equation $$\dfrac{\partial \beta}{\partial l}=\rho_{ml}(\beta(l))$$. The $$\rho_{ml}(\beta)$$ is standardized to have unit $$L_1$$ norm, which is the $$L_1$$ arc length in the monotone lasso situation.





&#x20;   The margin of a fitted model $$f(x)=\sum_k \alpha_kT_k(x)$$ is defined as&#x20;

$$
m(f)=\min_i \dfrac{y_if(x_i)}{\sum^K_{k=1} |\alpha_k|}
$$



**Learning Ensembles**

&#x20;The ensemble can be divided into two steps.&#x20;

* A finite dictionary $$T_L=\{T_1(X),...,T_M(x)\}$$is induced from the training data.
* $$\alpha(\lambda)=argmin_\alpha \sum^N_{i=1} L[y_i, \alpha_0+\sum^M_{m=1} \alpha_mT_m(x_i)]+\lambda \sum^M_{m=1} |\alpha_m|$$

&#x20;   These step are seen as a way of post-processing boosting or random forests, because in a first step $$T_L$$ is already fitted by the gradient boosting or random forest.



$$
f(x)=\int \beta(\gamma)b(x;\gamma)d\gamma \approx f_M(x)=\alpha_0+\sum^M_{m=1}\alpha_mb(x;\gamma_m)
$$

&#x20;   The measure of (lack of) relevance using loss function is evaluated on the training data by this formula. $$Q(\gamma)=\min_{c_0,c_1} \sum^N_{i=1} L(y_i, c_0+c_1b(x_i;\gamma))$$. $$\gamma^*=argmin_{\gamma \in \Gamma} Q(\gamma)$$, which is the global minimizer so $$Q(\gamma) \geq Q(\gamma^*)$$



$$
\sigma=E_S[Q(\gamma)-Q(\gamma^*)]
$$

* Too narrow $$\sigma$$ suggests too many of the $$b(x;\gamma_m)$$ look alike, and similar to $$b(x;\gamma^*)$$
* Too wide $$\sigma$$ suggests a large spread in the $$b(x;\gamma_m)$$, but possibly consisting of many irrelevant cases.



<details>

<summary>ISLE Ensemble Generation</summary>

&#x20; 1\. $$f_0(x)=argmin_c \sum^N_{i=1} L(y_i,c)$$

&#x20; 2\. For $$m=1$$ to $$M$$ do

$$(a) \; \gamma_m=argmin _\gamma \sum_{i \in S_m(\eta)} L(y_i,f_{m-1}(x_i)+b(x_i;r))\\ (b) \; f_m(x) = f_{m-1}(x)+\nu b(x;\gamma_m)$$

&#x20; 3\. $$T_{ISLE}=\{b(x;\gamma_1),b(x;\gamma_2),...,b(x;\gamma_M)\}$$

</details>



* &#x20;$$S_m(\eta)$$ refers to a subsample of $$N \times \eta (\eta \in (0,1])$$ of the training observations, typically without replacement.&#x20;
* $$\nu \in [0,1]$$ is the memory into the randomization process; the larger $$\nu$$, the more the procedure avoids $$b(x;\gamma)$$ similar to these found before.



&#x20;   Randomization schemes we already deal with are special cases of **ISLE Ensemble Generation**

* Bagging: Sampling with replacement has $$\eta=1, \nu=0$$. Sampling without replacement with $$\eta=1/2$$ is equal to sampling without replacement with $$\eta=1$$, but former one is much more efficient.
* Random Forest: More randomness is introduced by the selection of the splitting variable. Smaller  $$\eta <1/2$$  is, smaller $$m$$ is in random forests.
* Gradient Boosting: With shrinkage uses $$\eta=1$$, but typically doesn't produce sufficient width $$\sigma$$
* Stochastic Gradient Boosting

&#x20;   **Importance Sampled Learning Ensemble(ISLE)** is recommended with $$\nu=0.1, \eta \leq1/2$$







&#x20;sparse versus dense can be calculated with the noise-to-signal ratio(NSR)&#x20;







Revisit the Gradient Tree Boosting Algorithm

In Boosting tree, one tree is formally expressed as $$T(x; \Theta)=\sum^J_{j=1} \gamma_j I(x\in R_j)$$ with parameter $$\Theta=\{R_j,\gamma_j\}^J_1$$. This parameter can be found by solving $$\hat{\Theta}=argmin_\Theta \sum^J_{j=1} \sum_{x_i \in R_j} L(y_i,\gamma_j)$$





The parameter&#x20;

<details>

<summary>Gradient Tree Boosting Algorithm</summary>

1. Initialization $$f_0(x)=argmin _\gamma \sum^N_{i=1} L(y_i,\gamma)$$&#x20;
2. For $$m=1 \;to \; M$$

(a) $$For \;i=1,2,...,N \; \quad \quad r_{im}=-\Big[ \dfrac{\partial L(y_i,f(x_i))}{\partial f(x_i)} \Big]{f=f_{m-1}} \$$

(b) Fit a regression tree to the targets $$r_{im}$$

(b) $$For \;j=1,2,...,J_m \quad \gamma_{jm}=argmin_\gamma \sum_{x_i \in R_{jm}} L(y_i,f_{m-1}(x_i)+\gamma)$$



&#x20;

&#x20; 3\. Output $$\hat{f}(x)=f_M(x)$$



</details>







