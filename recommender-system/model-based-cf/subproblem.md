---
description: Bias correction and implicit feedback
---

# Subproblem

## Biases of user and item

$$
\hat{r}_{ij}=o_i+p_j+\sum^k_{s=1}u_{is}\cdot v_{js}
$$

* $$o_i$$ is the general bias of users to rate items. It is a positive value for a generous person, and a negative value for a curmudgeon.&#x20;
* $$p_j$$is the bias in the ratings of item $$j$$. Highly liked items will have a larger value, while disliked items have a negative value.
* $$(i,j)_{th}$$ rating is explained by $$o_i+p_j$$ and the remainder is explained by the product of the latent variables.

$$
e_{ij}=r_{ij}-\hat{r}_{ij}=r_{ij}-o_i-p_j-\sum^k_{s=1}u_{is}\cdot v_{js}
$$

![](<../../.gitbook/assets/image (181).png>)

&#x20;   The regularization factor $$\lambda$$ can be differ from user biases, item biases, and factor variables. Instead of having separate bias variable $$o_i$$ and $$p_j$$, we just can increase the size of the factor matrices to incorporate these bias variables as follows:

$$
u_{i,k+1}=o_i \;\; \forall i\in\{1,...,m\} \\ u_{i,k+2}=1\;\;\forall i \in \{1,...,m\} \\ v_{j,k+1}=1 \;\; \forall j \in \{1,...,n\} \\ v_{j,k+2}=p_j \;\;\forall j \in \{1,...,n\}
$$

&#x20;   Now $$U$$ is $$m\times (k+2)$$ matrix and $$V$$ is $$n \times (k+2)$$ matrix. The optimization problem is changed as follows:

![](<../../.gitbook/assets/image (130).png>)

![](<../../.gitbook/assets/image (89).png>)

<details>

<summary>Update algorithm</summary>

$$u_{iq} \Leftarrow u_{iq}+\alpha(e_{ij}\cdot v_{jq} -\lambda \cdot u_{iq}) \; \; \forall q\in \{1,...,k+2\}$$

$$v_{jq} \Leftarrow v_{jq}+\alpha(e_{ij}\cdot u_{iq} -\lambda \cdot v_{jq}) \; \; \forall q\in \{1,...,k+2\}$$

Reset the entries in $$(k+2)_{th}$$column of $$U$$ and $$(k+1)_{th}$$column of $$V$$to $$1s$$

</details>

&#x20;   Adding such bias terms reduces overfitting in many cases.&#x20;

> "Of the numerous new algorithmic contributions, I would like to highlight one - those humble baseline predictors (or biases), which capture main effects in the data. While the literature mostly concentrates on the more sophisticated algorithmic aspects, we have learned that an accurate treatment of main effects is probably at least as significant as comping up with modeling breakthroughs. **- Netflix Prize contest**"

&#x20;   One can just use biases in modeling, by doing so one can subtract this value from a rating matrix before applying collaborative filtering. It is similar with the row-wise mean centering for bias-correction in a neighborhood model, but it is a more sophisticated way because it adjusts for both user and item biases.



## Implicit Feedback

&#x20;   Explicit feedback is a clear feedback such as rating and like, whereas implicit feedback is more unclear. For example, whether one clicks on the item is an implicit feedback. However, even in cases in which users explicitly rate items, the identity of the items they rate can be viewed as an implicit feedback.&#x20;

> "Intuitively, a simple process could explain the results \[showing the predictive value of implicit feedback]: users chose to rate songs they listen to, and listen to music they expect to like, while avoiding genres they dislike. Therefore, most of the songs that would get a bad ratings are not voluntarily rated by the users. Since people rarely listen to random songs, or rarely watch random movies, we should expect to observe in many areas a difference between the distribution of ratings for random items and the corresponding distribution for the items selected by the users." - R. Devooght, N. Kourtellis, and A. Mantrach. Dynamic matrix factorization with priors on unknown values. ACM KDD Conference, 2015.

&#x20;       Asymmetric factor models and SVD++ have been proposed to incorporate implicit feedback. It uses two item factor matrices $$V$$and$$Y$$which reflect explicit and implicit feedback, respectively.&#x20;



<mark style="background-color:yellow;">**Solution: Asymmetric Factor Models**</mark>

&#x20;   This basic idea of this model is that two users will have similar user factors if they have rated similar items, irrespective of the values of the ratings. This model can incorporate other independent implicit feedback into matrix $$F$$.

$$
R\approx [FY]V^T
$$

* The $$m\times n$$implicit feedback matrix$$F$$ is a row-scaled matrix of rating matrix.

![](<../../.gitbook/assets/image (175).png>)

* The $$n\times k$$implicit item-factor matrix $$Y$$: if the element is large, it means that the act of rating item $$i$$ contains significant information about the affinity of that action for the $$j_{th}$$ latent component, no matter what the actual value of the rating might be.
* The $$n\times k$$explicit item-factor matrix $$V$$.
* In the item-based parameterization, $$[YV^T]$$ can be viewed as an $$n\times n$$ item-to-item prediction matrix. It tells us that how the action selecting item $$i$$ affects the predicted rating of item$$j$$.
* This model can work well for out-of-sample users, although it doesn't work for out-of-sample items.



<mark style="background-color:yellow;">**Solution: SVD++**</mark>

$$
R\approx (U+FY)V^T
$$

* $$FY$$is used to adjust the explicit user-factor matrix $$U$$
* The implicit feedback component of the predicted rating is given by $$(FY)V^T$$

![](<../../.gitbook/assets/image (96).png>)

![](<../../.gitbook/assets/image (141).png>)

&#x20;   $$(i,s)_{th}$$ entry of $$[FY]$$ is given by $$\sum_{h\in I_i}\frac{y_{hs}}{\sqrt{|I_i|}}$$. This model can be viewed as a combination of the unconstrained matrix factorization model and the asymmetric factorization model. In terms of its having an implicit feedback term together with its regularizer, it's different from the model in the previous section.

<details>

<summary>Updates</summary>

$$u_{iq} \Leftarrow u_{iq}+\alpha(e_{ij}\cdot v_{jq}-\lambda\cdot u_{iq}) \;\; \forall q \in \{1,...,k+2\}$$

$$v_{jq} \Leftarrow v_{jq} +\alpha(e_{ij}\cdot[u_{iq}+\sum_{h \in I_i} \dfrac{y_{hq}}{\sqrt{|I_i|}}]-\lambda\cdot v_{jq}) \;\; \forall q\in \{1,...,k+2\}$$

$$y_{hq} \Leftarrow y_{hq}+\alpha(\dfrac{e_{ij}\cdot v_{jq}}{\sqrt{|I_i|}}-\lambda\cdot y_{hq}) \;\; \forall q\in\{1,...,k+2\}, \; \forall h\in I_i$$



</details>



<mark style="background-color:yellow;">**Solution: Non-negative Matrix Factorization**</mark>

![](<../../.gitbook/assets/image (166).png>)

&#x20;   NMF provides great interpretability to implicit feedback situation. Especially, it is useful for the mechanism to specify a liking for an item, but no mechanism to specify a dislike. In customer transaction data, not buying an item does not necessarily imply a dislike because there is a probability of customers buying this item.



<details>

<summary>Updates</summary>

$$u_{ij} \Leftarrow \dfrac{(RV)_{ij}u_{ij}}{(UV^TV){ij}+\epsilon}\;\;\forall i\in\{1,...,m\}, \;\forall j\in \{1,...,k\}$$

$$v_{ij} \Leftarrow \dfrac{(R^TU)_{ij}v_{ij}}{(VU^TU)_{ij}+\epsilon}\;\; \forall i \in \{1,...,n\}, \; \forall j \in \{1,...,k\}$$



$$\epsilon$$ is a small value such as $$10^{-9}$$ to increase numerical stability. The entries are initialized to random values in $$(0,1)$$.

[Proof](https://angeloyeo.github.io/2020/10/15/NMF.html)



The  modified version is as follows:

$$u_{ij} \Leftarrow max\{[\dfrac{(RV)_{ij}-\lambda_1 u_{ij}}{(UV^TV)_{ij}+\epsilon}]u_{ij},0\}, \;\; \forall i\in\{1,...,m\}, \; \forall j \in\{1,...,k\}$$

$$v_{ij} \Leftarrow max\{[\dfrac{(R^TU)_{ij}-\lambda_2 v_{ij}}{(VU^TU)_{ij}+\epsilon}]v_{ij},0\}, \;\; \forall i \in \{1,...,n\}, \; \forall j\in \{1,...,k\}$$

</details>

![](<../../.gitbook/assets/image (88).png>)

* There are clear two classes of dairy products and drinks.
* All customers seem like juice, but there is a high correlation between user and buying aspects.&#x20;
* Customer 1 to 4 like dairy products, whereas customer 4 to 6 like drinks.

![Each of a part can be viewed as a user-item co-cluster](<../../.gitbook/assets/image (172).png>)

![](<../../.gitbook/assets/image (122).png>)

$$
UV^T=\sum^k_{i=1}\bar{U}_i\bar{V}_i^T
$$

&#x20;  &#x20;

&#x20;   Unlike explicit feedback data sets, it is not possible to ignore the missing entries in the optimization model because of the lack of negative feedback in such data. NMF just treats these missing entries as $$0s$$. However, too many zeros can cause computational challenges in large matrices. It can be handled with an ensemble method or by weighting less zero entries.



