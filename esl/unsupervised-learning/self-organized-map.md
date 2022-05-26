# Self-Organized Map

&#x20;   SOM is the way that our prototypes are project onto one or two dimensional space. $$K$$ prototypes $$m_j \in \mathbb{R}^p, \;j=1,...,K$$ are parametrized with respect to an integer coordinate pair $$l_j \in Q_1\times Q_2, \; Q_1=\{1,2,...,q_1\},\;Q_2=\{1,2,...,q_2\},\;K=q_1\cdot q_2$$.  In this setting, prototypes can be viewed as "buttons" on **the principal component plane** in a regular pattern. At first, the prototypes $$m_j$$ are initialized, and these are updated. For all neighbors $$m_k$$ of the closest prototype $$m_j$$ to $$x_i$$, we have $$m_k$$ move toward $$x_i$$ via this update.

$$
m_k \leftarrow m_k+\alpha(x_i-m_k)
$$

&#x20;   The neighbors of $$m_j$$ are defined to be all $$m_k$$ such that the distance between $$l_j$$ and $$l_k$$ is small. The small is determined by a threshold $$r$$. This distance is defined in the space $$Q_1 \times Q_2$$. The sophisticated version is as follows.

$$
m_k \leftarrow m_k +\alpha h(||l_j-l_k||)(x_i-m_k)
$$

&#x20;   If we set the threshold $$r$$ small enough so that each neighborhood contains only one point, then the spatial connection between prototypes is lost. In that case SOM algorithms becomes an online version of K-means clustering. Thus, we can call SOM a constraint version of K-means, and the constraint is that the prototypes are projected on lower dimension.





![](<../../.gitbook/assets/image (199).png>)

![](<../../.gitbook/assets/image (193).png>)



