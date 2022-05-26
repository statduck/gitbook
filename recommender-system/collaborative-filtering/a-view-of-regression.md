# A View of Regression

## <mark style="background-color:red;">A Regression Modeling View of NB methods.</mark>

$$
\hat{r}_{ut}=\dfrac{\Sigma_{j\in Q_t(u)}AdjustedCosine(j,t)\cdot r_{uj}}{\Sigma_{j \in Q_t(u)}|AdjustedCosine(j,t)|}
$$

### User-Based Nearest Neighbor Regression

&#x20; The predicted rating is a weighted linear combination of other ratings of the same item. If $$P_u(j)$$contains all ratings of item j, this combination becomes similar to a linear regression. The difference is that the linear regression find coefficients by solving optimization problems, whereas the recommender system chooses coefficients in a heuristic way with the user-user similarities.&#x20;

$$
\hat{r}_{uj}=\mu_u+\dfrac{\Sigma_{v\in P_u(j)}Sim(u,v)\cdot s_{vj}}{\Sigma_{v \in P_u(j)}|Sim(u,v)|}, \;\; s_{vj}=r_{vj}-\mu_v
$$

&#x20;The above expression is changed into the below expression.

$$
\hat{r}_{uj}=\mu_u+\sum_{v \in P_u(j)} w^{user}_{vu} \cdot (r_{vj}-\mu_v)
$$

$$
minJ_u=\sum_{j\in I_u}(r_{uj}-\hat{r}_{uj})^2=\sum_{j\in I_u}(r_{uj}-[\mu_u+\sum_{v \in P_u(j)} w^{user}_{vu}\cdot (r_{vj}-\mu_v)])^2
$$

$$
min\sum^m_{u=1}J_u=\sum^m_{u=1}\sum_{j\in I_u}(r_{uj}-[\mu_u+\sum_{v \in P_u(j)} w^{user}_{vu}\cdot (r_{vj}-\mu_v)])^2
$$

&#x20;To reduce model complexity, the regularization term like $$\lambda \Sigma_{j \in I_u} \Sigma_{v \in P_u(j)} (w^{user}_{vu})^2$$ could be added as regression do. $$P_u(j)$$ can be vastly different for the same user $$u$$ and varying item indices(denoted by $$j$$), because of the extraordinary level of sparsity inherent in rating matrices. Let me consider a scenario where one similar user rated movie $$Nero$$ whereas four similar user rated $$Gladiator$$ for target user $$u$$. The regression coefficient $$w^{user}_{vu}$$is heavily influenced by the rating for $$Gladiator$$ because it has more sample. It leads to overfitting problem, so scaling method needs to be applied in $$P_u(j)$$.

$$
\hat{r}_{uj}\cdot \dfrac{|P_u(j)|}{k}=\mu_u+\sum_{v \in P_u(j)} w^{user}_{vu} \cdot (r_{vj}-\mu_v)
$$

This expression predicts a fraction $$\frac{|P_u(j)|}{k}$$ of the rating of target user $$u$$ for item $$j$$.

$$
\hat{r}_{uj}=b^{user}_u+\dfrac{\Sigma_{v \in P_u(j)} w^{user}_{vu} \cdot (r_vj - b^{user}_v)}{\sqrt{|P_u(j)|}}
$$

$$\mu_v$$ is replaced by a bias variable $$b_u$$

$$
\hat{r}_{uj}=b^{user}_u+b^{item}_j+\dfrac{\Sigma_{v \in P_u(j)} w^{user}_{vu} \cdot (r_{vj} - b^{user}_v - b^{item} _ j )}{\sqrt{|P_u(j)|}}
$$

### Item-Based Nearest Neighbor Regression



$$
\hat{r}_{ut}=\sum_{j \in Q_t(u)}w^{item}_{jt} \cdot r_{uj}
$$

$$
minJ_t=\sum_{u \in U_t} (r_{ut}-\hat{r}_{ut})^2=\sum_{u \in U_t} (r_{ut}-\sum_{j \in Q_t(u)} w^{item}_{jt} \cdot r_{uj})^2
$$

$$
min\sum_{t=1}^n\sum_{u \in U_t}(r_{ut} -\sum_{j \in Q_t(u)} w^{item}_{jt}\cdot r_{uj})^2
$$

$$
\hat{r}_{ut}=b^{user}_u+b^{item}_t+\dfrac{\Sigma_{j \in Q_t(u)}w^{item}_{jt} \cdot (r_{uj}-b^{user}_u-b^{item}_j)}{\sqrt{|Q_t(u)|}}
$$

### Combined Method

$$
\hat{r}_{uj}=b^{user}_{u}+b^{item}_j+\dfrac{\Sigma_{v \in P_u(j)} w^{user}_{vu} \cdot (r_{vj}-B_{vj})}{\sqrt{|P_u(j)|}}+\dfrac{\Sigma_{j\in Q_t(u)} w^{item}_{jt} \cdot (r_{uj}-B_{uj})}{\sqrt{|Q_t(u)|}}
$$

$$
\
$$

