# Association Rule

**Goal**: Find the most frequently appearing $$X=(X_1,...,X_p)$$. This problem can be viewed as to the problem finding the frequent subsets $$(v_1,...v_L), \;v_j \subset X$$, such that the probability density $$P(v_l)$$ evaluated at each of those values is relative large.



&#x20;   In most cases $$X_j\in \{0,1\}$$, where it is referred to as "market basket" analysis. For observation $$i$$, each variable $$X_j$$ is assigned one of two values; $$x_{ij}=1$$ if the $$j_{th}$$ item is purchased. In this setting of the goal, $$X=v_l$$ will nearly always be too small for reliable estimation. Thus we need to modify our goal as following way.



**Modified Goal**: Instead of seeking values $$x$$ where $$P(x)$$is large, We seeks regions of the $$X$$space with high probability content relative to their size or support. Then, the modified goal is to find subsets of variable $$s_1,...,s_p$$ such that the probability of each of the variables is relative large.

$$
P\Big[\bigcap^p_{j=1} (X_j \in s_j)\Big]
$$





The intersection part is called a conjunctive rule. The subsets $$s_j$$ are interval for quantitative $$X_j$$.

$$
P\Big[\bigcap _{j \in J} (X_j=v_{0j})\Big]
$$

$$K\subset \{1,...,P\},\; P=\sum^p_{j=1}|S_j|$$.   $$|S_j|$$ is the number of distinct values attainable by $$X_j$$. $$K$$ is called an item set.

$$
P\Big[\bigcap_{k\in K} (Z_k=1)\Big] =P\Big[\prod_{k \in K} Z_k=1\Big ] \\ \widehat{Pr}\Big[\prod _{k \in K} (Z_k=1) \Big]=\dfrac{1}{N} \sum^N_{i=1} \prod _{k \in K} z_{ik}
$$





### Market Basket Analysis

&#x20;  &#x20;









