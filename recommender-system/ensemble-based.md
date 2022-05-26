# Ensemble-Based

<mark style="background-color:yellow;">**Background**</mark>

&#x20;   Recommender system is the generalization of a classification algorithm. The difference is explained by this figure.

![](<../.gitbook/assets/image (100).png>)

&#x20;   The only difference is that missing entries can occur in any column. Nevertheless, the bias-variance structure still remain. $$Error = Bias^2 + Var + Noise$$

&#x20;   For reference,  to apply a classification algorithm into recommender system, two steps are needed. First, one item has to be regarded as a target variable, and other items as independent variables. Second, the target variable should move from first column to last column, the $$n$$ algorithms work.



<mark style="background-color:yellow;">**How to work**</mark>

&#x20;   When one can access different data sources, one can make a more robust inference by combining several models together.

* **Ensemble design**: $$\hat{R}_k$$ is the prediction matrix of $$m$$ users for $$n$$ items by the $$k_{th}$$algorithm, where $$k \in \{1,...,q\}$$. The final result can be a weighted average of $$\hat{R}_1,...,\hat{R}_q$$. In some sequential ensemble algorithm, $$\hat{R}_k$$ may depend on the previous output $$\hat{R}_{k-1}$$. In other cases, outputs may not be directly combined. One output can also be used as an input to the next output.
* **Monolithic design**: An integrated recommendation algorithm is created by using various data types.
* **Mixed system**: TV program is a composite entity containing multiple items, so the combination of the items creates the recommendation.



## Weighted Hybrids

<mark style="background-color:yellow;">**Key concept**</mark>: You can mix the concepts of choosing optimal weight and linear regression!! 🔥

$$
\hat{R}=\sum^q_{i=1}\alpha_i \hat{R}_i
$$

&#x20;   $$\hat{R}_1,...,\hat{R}_q$$ is the $$m \times n$$ completely specified ratings matrices, in which the unobserved entries are predicted by $$q$$ different algorithms. To determine the value of $$\alpha_i$$, we need to evaluate the metric like MSE of MAE.

$$
MSE(\bar{\alpha})=\dfrac{\sum_{(u,j)\in H}(\hat{r}_{uj}-r_{uj})^2}{|H|} \\ MAE(\bar{\alpha})=\dfrac{\sum_{(u,j)\in H}|\hat{r}_{uj}-r_{uj}|}{|H|}
$$

&#x20;   In this expression, $$H$$ means the hold out user-item matrix, which could be thought as a validation or test set. For the case of MSE, we can regard $$\alpha_1,...,\alpha_q$$ as independent variables and $$r_{uj}$$ becomes a target variable.  Using linear regression approach, we can solve this problem and get optimal $$\hat{\alpha}_1,...,\hat{\alpha}_q$$(coefficients)&#x20;

&#x20;   `In short, you solve linear regression problem with coefficient` $$\alpha$$ !!

&#x20;    Usually, the cross-validation method is also used.



&#x20;   However, the method of sum of squared is sensitive to presence of noise and outlier, because the residual value is overemphasized due to the squareness. For the correction of this weakness, you can just use sum of absolute value, called as MAE metric. In other words, instead of linear regression, one can adjust it into **robust regression method**.&#x20;

$$
\dfrac{\partial MAE(\bar{\alpha})}{\partial \alpha_i}=\dfrac{\sum_{(u,j)\in H}\tfrac{\partial |(\hat{r}_{uj}-r_{uj})}{\partial \alpha_i}}{|H|} =\dfrac{\sum_{(u,j) \in H} sign(\hat{r}_{uj}-r_{uj})\hat{r}^i_{uj}}{|H|}
$$

$$
\overline{\nabla MAE}=\Big(\dfrac{\partial MAE(\bar{\alpha})}{\partial \alpha_1}\cdots \dfrac{\partial MAE(\bar{\alpha})}{\partial \alpha_q}\Big)
$$

<details>

<summary>Process</summary>

1. Initialization $$\bar{\alpha}^{(0)}=(1/q  \; ... \; 1/q), \;\;t=0$$
2. Update $$\bar{\alpha}^{(t+1)}=\bar{\alpha}^{(t)}-\gamma\cdot \overline{\nabla MAE}, \;\; t=t+1$$
3. Check If MAE has improved then go updating
4. Report $$\bar{\alpha}^{(t)}$$

</details>



&#x20;   To apply the weighted hybrids to more generalized prediction, we also can add on a regularization term or put constraints on $$\alpha_i$$ such as non-negativity or summation to $$1$$.



### Model Combination

<mark style="background-color:yellow;">**Homogeneous data type and model classes**</mark>

&#x20;   Different models are applied on the same data, and results are aggregated into a single predicted value. This approach is robust because it avoids each bias which each model has. Same model with different parameter can be also used.

<mark style="background-color:yellow;">**Heterogeneous data type and model classes**</mark>

&#x20;   One component of the model might be a collaborative recommender using a rating matrix, whereas another component of the model might be a content-based recommender.



### Bagging

&#x20;   For modification of bagging into collaborative filtering, we face two challenges as follows.

* No clear demarcation between training and test rows.&#x20;
* No clear distinction between dependent and independent columns.

&#x20;   To execute bagging, $$q$$ training sets are created with _bootstrapped sampling_. **Bootstrapping** is a **way to sample with replacement** in order to create a new training data. For example, if we have a data \[9, 7, 5, 3, 1], we can conclude that the sample mean is 5. However, we don't know the true value of a mean. By sampling from this data with replacement, we can make a several data set like this: \[9, 7, 7, 1, 1], \[5, 7, 3, 3,1], \[9, 7, 5, 3, 1]. By calculating sample mean from each data list, we can predict the variance of a sample mean and also get a confidence interval. It helps for estimation when we don't know the distribution from which the data comes.

&#x20;   The expected fraction of rows which is not represented in a given bootstrapped sample is given by $$1/e \approx 1/3$$. In other words, each bootstrapped sample can contain original rows with about 2/3 probability. $$\lim_{n \rightarrow \infty}(1-\dfrac{1}{n})^n=1/e$$ .  Bagging is a way to average prediction from $$q$$ models for a given test data. It reduces variance of our model.



**Initialization** : missing entries are initialized with row averages, column averages, or something used simple cf algorithm.&#x20;

**Update**: Missing entries of each columns are set as the target variable and remaining columns as the feature variables. When the target variable is observed, the row is treated as a train set. The only thing to do is apply algorithm $$A$$ for this matrix.

Update is iteratively repeated to convergence.



* Row-wise bootstrapping: The rows of $$R$$ are sampled with replacement to create new ratings matrices $$R_1 ,...,R_q$$. Then an existing cf filtering(e.g. latent factor model) can be applied for prediction. Final predicted rating is the average rating of that item over the duplicate occurrences of one user. Each user will be at least in one ensemble component with $$(1-(1/e)^q$$ with large $$q$$.
* Row-wise subsampling: It is similar to row-wise bootstrapping, except that the rows are sampled without replacement. The fraction $$f$$ of rows sampled is chosen randomly from $$(0.1, 0.5)$$. $$q$$ should be grater than $$10$$ to ensure that all rows are represented. Because it is difficult to predict all entries in this setting, so some entries are averaged over only a small number of samples, leading not to good variance reduction
* Entry-wise bagging
* Entry-wise subsampling



<mark style="background-color:yellow;">**Randomness Injection**</mark>

* Injection into a neighborhood model: Instead of using top $$k$$ nearest neighbors, one can use top $$\alpha \cdot k$$ neighbors with $$\alpha >\!\!> 1$$.
* Injection into a matrix factorization model:&#x20;



## Switching Hybrids

&#x20;   In the context of a model selection, switching hybrids are mostly used in recommender system. The basic idea is that **some model works better in earlier stage**s but **other models better in later stages**.&#x20;



&#x20;<mark style="background-color:yellow;">**Bucket-of-Models**</mark>

&#x20;A fraction(e.g. 25% to 33%) of the specified entries are held out, and various models are applied to the resulting matrix. The held out set is used for validation set to evaluate the metric like MSE or MAE.



## Cascade Hybrids

&#x20;   Cascade Hybrids are the way to refine the recommendations made by the previous recommender.

### Successive way

1. First recommender provided a rough ranking and might eliminate many potential items.
2. Second recommender uses this rough ranking to refine it and break ties.
3. Resulting recommender is presented to the user.&#x20;



### Boosting

&#x20;   The weight associated with $$(u,j)_{th}$$ entry of the rating matrix is dented by $$W_t(u,j)$$.&#x20;

* $$\hat{r}_{uj}$$ is different from $$r_{uj}$$ by at least a predefined amount $$\delta$$ ->  $$(u,j) \in S$$ incorrect&#x20;
* $$\epsilon_t$$ is the fraction of specified ratings in $$S$$ where predicted value is incorrect.
* For correctly predicted value, the weight becomes reduced. $$W_t(u,j)$$=$$\epsilon_t \times W_t(u,j)$$ But, for incorrectly predicted value, the weight stays unchanged.
* The weights are always normalized to sum to $$1$$ in each iteration.



### Weighted Base Models



<mark style="background-color:yellow;">**Neighborhood-based algorithm**</mark>

<mark style="background-color:yellow;">****</mark>

![](<../.gitbook/assets/image (162).png>)

![](<../.gitbook/assets/image (187).png>)



<mark style="background-color:yellow;">**Latent factor models**</mark>

&#x20;   The weighted sum of squares of the errors have to be minimized.&#x20;

![](<../.gitbook/assets/image (114).png>)

## Feature Augmentation

* Generate some recommendation like "related authors" and "related titles" for items by using collaborative recommender system. Then, content-based recommender can be used in leveraging these features.
* Use content-based recommender to fulfill the missing entries so that it is no longer sparse. Then, a collaborative recommender is used on this dense rating matrix. The final prediction is the weighted sum of each prediction from content-based way and collaborative way.



## Feature combination hybrids

&#x20;   This is the way to integrate various data sources into a unified representation. In this case, the objective function becomes like this:

$$
J=CollaborativeObjective(\bar{\theta})+\beta ContentObjective(\bar{\theta})+Regularization
$$

&#x20;   For example, let $$R$$ be an $$m\times n$$ implicit feedback rating matrix, and $$C$$ be a $$d \times n$$ content matrix, in which each item is described by non-negative frequencies of $$d$$ words. $$W$$ is an $$n \times n$$ item-item coefficient matrix s.t. $$\hat{R}=RW$$.

$$
Min J=||R-RW||^2+\beta \cdot ||R-CW||^2+\lambda ||W||^2 +\lambda_1 \cdot ||W||_1 \\ subject \;to: \;W\geq0 , \;Diag(W)=0
$$



