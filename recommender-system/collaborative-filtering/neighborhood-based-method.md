---
description: Neighborhood-Based Collaborative Filtering
---

# Neighborhood-Based method

## <mark style="background-color:red;">Introduction</mark>

![Reference url](<../../.gitbook/assets/image (126).png>)

### Motivation

&#x20;   This algorithm assumes that similar users show similar patterns in rating. It can be categorized into to methods.

* User-based collaborative filtering(cf): The predicted ratings of user A are computed from the peer group ratings.
* Item-based collaborative filtering(cf): The predicted ratings of user A are computed from the similar items with target item.

&#x20;   This algorithm uses matrix $$R$$(user-item rating matrix) that has $$[r_{uj}]$$elements. Most of elements are missing in recommendation problem, so we need to handle with sparse matrix $$R$$.



### Problem

&#x20;   The problems we need to solve are two as follows:

* Predicting the rating value of a user-item combination
* Determining the top-k items or top-k users

&#x20;   These two problems are closely connected, because in order to determine the top-k items for one user, one can predict the ratings of each item for that user. For neighborhood-based method to solve the prediction problem, it makes use of similar user/item information.



## <mark style="background-color:red;">Rating Matrices</mark>

&#x20;   The thing is that the most important part in recommendation is assuming rating matrix $$R$$

### Rating methods

| Rating                 | Example                    |
| ---------------------- | -------------------------- |
| Continuous ratings     | Real value from -10 to 10  |
| Interval-based ratings | Integer value from 1 to 5  |
| Ordinal ratings        | Ordered categorical values |
| Binary ratings         | Positive or Negative       |
| Unary ratings          | Like button                |

&#x20;   The element in rating matrix varies by the type of rating method. Above this table, unary rating is an implicit feedback. It needs to be considered PU learning problem(Positive and Unlabeled learning).

### Rating distribution

&#x20;   The distribution of this rating often has a long tail property because some items are popular but others are not. It leads to a highly skewed distribution of the underlying ratings.&#x20;

![](<../../.gitbook/assets/image (101).png>)

* In many cases, high-frequency items give a little profit than the lower frequency item.
* It is hard to provide robust rating prediction in the long tail part because of the rarity of this region. Many recommendation system has a tendency to suggest popular items rather than infrequent items.
* The neighborhoods are often defined on the high frequency item, which doesn't reflect the low-frequency item. It leads to misleading evaluations of recommender system.

&#x20;   To resolve this long tail problem, we can get key concept from Inverse Document Frequency (idf). $$m_j$$is the number of ratings of item $$j$$, and $$m$$is the total number of users. It makes weights for each item.

$$
w_j=log(\dfrac{m}{m_j}), \;\;\; \forall j\in\{1,...,n\}
$$

With this correction, the Pearson expression is also changed like this(re-read this part after you read the other parts!)

$$
Pearson(u,v)=\dfrac{\Sigma_{k\in I_u \cap I_v}w_k\cdot(r_{uk}-\mu_u)\cdot(r_{vk}-\mu_v)}{\sqrt{\Sigma_{k\in I_u \cap I_v}w_k\cdot(r_{uk}-\mu_u)^2} \cdot \sqrt{\Sigma_{k\in I_u \cap I_v}w_k\cdot(r_{vk}-\mu_v)^2}}
$$



## <mark style="background-color:red;">Prediction</mark>

### User-based model

The system predicts rating of unspecified item for user A.

![](<../../.gitbook/assets/image (151).png>)

#### <mark style="background-color:yellow;">**Basic setting**</mark>

<mark style="background-color:yellow;">****</mark>$$I_u$$ is a set of item indices for which ratings have been specified by user $$u$$

$$I_1=\{1,2,3,4,5,6\}, \; I_3=\{2,3,4,5\}, \;I_1\cap I_3=\{2,3,4,5\}$$

&#x20;$$\mu_u=\dfrac{\Sigma_{k\in I_u}r_{uk}}{|I_u|}\;\;\; \forall u\in \{1,...,m\}$$

#### <mark style="background-color:yellow;">**Correlation**</mark>

$$
Sim(u,v)=Pearson(u,v)=\dfrac{\Sigma_{k\in I_u \cap I_v}(r_{uk}-\mu_u)\cdot(r_{vk}-\mu_v)}{\sqrt{\Sigma_{k\in I_u \cap I_v}(r_{uk}-\mu_u)^2} \cdot \sqrt{\Sigma_{k\in I_u \cap I_v}(r_{vk}-\mu_v)^2}}
$$

&#x20;   In many implementations, $$\mu_u$$ and $$\mu_v$$ are calculated in pairwise fashion during the Pearson computation, which needs items rated both by user $$u$$ and $$v$$. Aside this way, you just can calculate $$\mu_u$$ with ignoring user $$v$$ like in basic setting. This way is good to evaluate $$\mu_u$$ over a single common item. The calculation in this posting adopts the latter way.

&#x20;   To determine the similar group for user $$u$$, we can select $$k$$ users with the highest Pearson coefficient with the user $$u$$. However, closest users can vary by the item, so it's better to select $$k$$users repeatedly by each predicted item.&#x20;

#### <mark style="background-color:yellow;">**Prediction function**</mark>

$$
\hat{r}_{uj}=\mu_u+\dfrac{\Sigma_{v\in P_u(j)}Sim(u,v)\cdot s_{vj}}{\Sigma_{v \in P_u(j)}|Sim(u,v)|}, \;\; s_{vj}=r_{vj}-\mu_v
$$

&#x20;   After defining the similar group, we predict rating for user $$u$$ using a weighted average of ratings from similar group. For the calculation, we need to consider that some users might like to rate highly but others can make rating usually lowly. To correct this bias, raw ratings need to be mean-centered in row-wise fashion. $$P_u(j)$$ is the set of $$k$$ closest users to target user $$u$$, who have specified ratings for item $$j$$( When some users in this set have low or negative correlations with target user we can eliminate these users from this set as a heuristic enhancement. )

#### <mark style="background-color:yellow;">**Example**</mark>

$$
Cosine(1,3)=\dfrac{6*3+7*3+4*1+5*1}{\sqrt{6^2+7^2+4^2+5^2}\cdot \sqrt{3^2+3^2+1^2+1^2}}
$$

$$
Pearson(1,3)=\dfrac{(6-5.5)*(3-2)+(7-5.5)*(3-2)+(4-5.5)*(1-2)+(5-5.5)*(1-2)}{\sqrt{1.5^2+1.5^2+(-1.5)^2+(-0.5)^2}\cdot \sqrt{1^2+1^2+(-1)^2+(-1)^2}}
$$

$$Cosine(1,3)=0.956, \;\; Pearson(1,3)=0.894$$

$$
\hat{r}_{31}=\dfrac{7*0.894+6*0.939}{0.894+0.939}\approx 6.49 , \;\;  \hat{r}_{36}=\dfrac{4*0.894+4*0.939}{0.894+0.939}=4
$$

&#x20;   This result implies that item 1 should be prioritized over item 6 as a recommendation to user 3, and user 3 is likely to be interested in both movies 1 and 6 to a greater degree than any of the movies this user has already rated.

$$
\hat{r}_{31}=2+\dfrac{1.5*0.894+1.2*0.939}{0.894+0.939} \approx 3.35 ,\;\; \hat{r}_{36}=2+\dfrac{-1.5*0.894-0.8*0.939}{0.894+0.939}\approx0.86
$$

&#x20;   However, after the mean-centered computation, we can check the preference for item 6 is lowest among items user 3 has rated, which is opposed to the result above. It suggests that there are some biases in rating of users, so it must be corrected by mean centered way.

#### <mark style="background-color:yellow;">**Variation**</mark>

_Similarity Function Variants_

$$
RawCosine(u,v)=\dfrac{\Sigma_{k \in I_u \cap I_v} r_{uk} \cdot r_{vk}}{\sqrt{\Sigma_{k\in I_u \cap I_v} r^2_{uk}}\cdot \sqrt{\Sigma_{k\in I_u \cap I_v} r^2_{vk}}}
$$

$$
RawCosine(u,v)=\dfrac{\Sigma_{k\in I_u \cap I_v} r_{uk} \cdot r_{vk}}{\sqrt{\Sigma_{k\in I_u} r^2_{uk}} \cdot \sqrt{\Sigma_{k \in I_v} r^2_{vk}}}
$$

&#x20;   Pearson correlation coefficient is preferable to the raw cosine method because Pearson method adjusts the bias effect using mean-centering.&#x20;



$$
DiscountedSim(u,v)=Sim(u,v)\cdot \dfrac{min\{|I_u \cap I_v|,\beta\}}{\beta}
$$

&#x20;   Two users can have only a small number of ratings in common. In this case, the similarity function has to be reduced to de-emphasize the importance of these two users. For this purpose, we can add the discount factor $$\frac{min\{|I_u \cap I_v|, \beta\}}{\beta}$$ , called _significance weighting_.



_Prediction Function_

$$
\sigma_u=\sqrt{\dfrac{\Sigma_{j\in I_u}(r_{uj}-\mu_u)^2}{|I_u|-1}} ,\;\; \forall u\in \{1,...,m\}
$$

$$
\hat{r}_{uj}=\mu_u+\sigma_u \dfrac{\Sigma_{v\in P_u(j)} Sim(u,v) \cdot z_{vj}}{\Sigma_{v\in P_u(j)}|Sim(u,v)|}, \;\; z_{uj}=\dfrac{r_{uj}-\mu_u}{\sigma_u}=\dfrac{s_{uj}}{\sigma_u}
$$

&#x20;   The $$Z$$score can be used to rank the items in order of desirability for user $$u$$, even if there are some cases that predicted values are outside the range of permissible ratings.&#x20;

&#x20;   Furthermore, there is a way to amplify the connectivity with $$Sim(u,v)=Pearson(u,v)^\alpha$$. Also by treating ratings as categorical values we can approach the recommender system as a classification problem.

### Item-Based Models

![](<../../.gitbook/assets/image (127).png>)

$$
AdjustedCosine(i,j)=\dfrac{\Sigma_{u\in U_i \cap U_j}s_{ui}\cdot s_{uj}}{\sqrt{\Sigma_{u \in U_i \cap U_j} s^2_{ui}}\cdot \sqrt{\Sigma_{u\in U_i \cap U_j}s^2_{uj}}}
$$

&#x20;   The ratings are mean-centered.&#x20;

$$
\hat{r}_{ut}=\dfrac{\Sigma_{j\in Q_t(u)}AdjustedCosine(j,t)\cdot r_{uj}}{\Sigma_{j \in Q_t(u)}|AdjustedCosine(j,t)|}
$$

&#x20;   $$Q_t(u)$$ is a set of top $$k$$ matching items to item $$t$$. The concept is same with the user-based methods.&#x20;

1. Determine similar items for item $$t$$
2. Leverage the user's own ratings on similar items
3. Predict $$\hat{r}$$

<mark style="background-color:yellow;">**Example**</mark>

$$
AdjustedCosine(1,3)=\dfrac{1.5*1.5+(-1.5)*(-0.5)+(-1)*(-1)}{\sqrt{1.5^2+(-1.5)^2+(-1)^2}\cdot\sqrt{1.5^2+(-0.5)^2+(-1)^2}}=0.912
$$

$$
\hat{r}_{31}=\dfrac{3*0.735+3*0.912}{0.735+0.912}=3 ,\;\; \hat{r}_{36}=\dfrac{1*0.829+1*0.730}{0.829+0.730}=1
$$

&#x20;   This result suggests item 1 is more likely to be preferred by user 3 than item 6.&#x20;

### Actual Implementation

&#x20;   The actual process is to compute all possible rating predictions for the relevant user-item pairs and then rank them. It needs a lot of computational time, so recommender system have an offline phase to compute these intermediate computations.

* Offline phase: the user-user(or item-item) similarity values and peer groups are computed.
* Online phase: Similarity values and peer groups are leveraged to make predictions

| Time complexity                 |      User-based     |     Item-based     |
| ------------------------------- | :-----------------: | :----------------: |
| Calculating Similarity          | $$O(m^2 \cdot n')$$ | $$O(n^2\cdot m')$$ |
| Online computation              |       $$O(k)$$      |      $$O(k)$$      |
| Ranking them for a target user  |   $$O(k\cdot n)$$   |  $$O(k \cdot n)$$  |
| Ranking users for a target item |   $$O(k \cdot m)$$  |   $$O(k\cdot m)$$  |

( $$m'$$ is the maximum number of specified ratings and $$n'$$ is the maximum running time for computing the similarity between a pair of users)

<mark style="color:red;">The thing is that computationally hard works have to be allocated to the offline phase.</mark>

### Advantage of item-based methods

* Provides more relevant recommendations because of the fact that a user's own ratings are used to perform the recommendation.
* Provides a concrete reason for the recommendation._(Because you watched "Secrets of the Wings," \[the recommendations are] \<List> - Netflix)_
* Being stable with changes to the ratings. Because the number of users is much larger than the number of items and new users are likely to be added more frequently in commercial systems than new items.

&#x20;   One can also use a unified method using a combination function of users and items.



&#x20;

## <mark style="background-color:red;">Dimensionality Reduction</mark>

* Compressing the item dimension or the user dimension
* Alleviating the sparsity problem
* Determining the latent representations of both the row space and the column space
* Being computed using either SVD or PCA

Example(SVD method)

1. Filling the incomplete $$m\times n$$ matrix $$R$$, by the mean of the corresponding row or column.
2. Computing $$n \times n$$ matrix $$S=R_f^TR_f$$, s.t. $$R_f$$ is the filled matrix in the previous step.
3. Decomposing $$S$$ into $$P\Delta P^T$$, $$P$$ is a $$n\times n$$ matrix whose columns have orthonormal eigenvector of S. Let $$P_d$$ be the $$n\times d$$ matrix containing only the columns of $$P$$ corresponding to the largest $$d$$ eigenvectors.
4. $$R_fP_d$$ is $$m\times d$$ matrix, which represents m users in a $$d$$dimensional space. We can use this matrix when we drive the peer group of each user.

&#x20;In PCA method, the covariance matrix of $$R_f$$ is used instead of the similarity matrix $$R_f^TR_f$$. The difference is that to compute the covariance matrix, the mean centered process is needed. It shows benefits in reducing bias.



### Bias

![](<../../.gitbook/assets/image (144).png>)

&#x20; This table clearly tells the fact that the ratings between Gladiator and Nero are same. It means the correlation between these two movies is high. However, in filling unspecified values(like in the SVD method above), the covariance structure changes as follows. In this case we fill out the missing value as 4(the mean of \[1, 7, 1, 7])

![Estimated Covariance Matrix](<../../.gitbook/assets/image (201).png>)

&#x20;This estimated covariance matrix seems wrong, because the covariance between Godfather and Gladiator(4.36) is bigger than one between Gladiator and Nero(2.18). It doesn't match the result of the rating table.



As remedies for bias, <mark style="color:red;">following methods</mark> are suggested.

<mark style="color:red;">Assuming covariance matrix based on a generative model.</mark>(Generative model is the term used in a semi-supervised model, and it focuses on the distribution of each class.)

![Estimated Covariance Matrix(From maximum likelihood estimation)](<../../.gitbook/assets/image (188).png>)



$$R$$ can be directly projected on the reduced matrix $$P_d$$, rather than projecting the filled matrix $$R_f$$ on $$P_d$$. $$a_{ui}$$ is the averaged contribution of user $$u$$ on the $$i$$th latent vector.

$$
a_{ui}=\dfrac{\Sigma_{j\in I_u} r_{uj} e_{ji}}{|I_u|}
$$

More direct approach for bias correction is to use matrix factorization method. When the matrix is sparse, estimation for covariance matrix becomes unreliable because it lost robustness.&#x20;



$$
R = Q\Sigma P^T
$$





