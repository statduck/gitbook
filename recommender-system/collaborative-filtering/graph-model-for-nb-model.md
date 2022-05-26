# Graph Model for NB model

## User-Item graphs



![\\](<../../.gitbook/assets/image (134).png>)

### Neighborhood definition

<mark style="background-color:yellow;">**Random Walks**</mark>

&#x20;   Personalized PageRank or the SimRank method to determine the k most similar users.



<mark style="background-color:yellow;">**Katz Measure**</mark>

$$
Katz(i,j)=\sum^\infty_{t=1}\beta^t \cdot \eta^{(t)}_{ij}
$$

$$\eta^{(t)}_{ij}$$ is the number of walks of length $$t$$ between nodes $$i$$ and $$j$$. The value of $$\beta$$ is a discount factor.&#x20;

$$
K=\sum^\infty_{i=1}(\beta A)^i=(I-\beta A)^{-1}-I
$$

## User-User Graphs

![](<../../.gitbook/assets/image (139).png>)



## Item-Item Graphs

