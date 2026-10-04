# LightGCN

Summarizing of [the paper](https://arxiv.org/pdf/2002.02126.pdf)

Head - LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.



**Algorithms:**

$$
\mathbf{e}^{k+1}_u=\sum_{i\in N_u}\dfrac{1}{\sqrt{|N_u|}\sqrt{|N_i|}}\mathbf{e}^{(k)}_i \\ \mathbf{e}^{k+1}_i=\sum_{u\in N_i}\dfrac{1}{\sqrt{|N_i|}\sqrt{|N_u|}}\mathbf{e}^{(k)}_u
$$

The final representation is the form of combined layer embeddings.$$\mathbf{e}_u=\sum^K_{k=0}\alpha_k \mathbf{e}_u^{(k)}; \;\; \mathbf{e}_i=\sum^K_{k=0}\alpha_k\mathbf{e}^{(k)}_i$$&#x20;



The model prediction is defined as the inner product of user and item final representations: $$\hat{y}_{ui}=\mathbf{e}^T_u\mathbf{e}_i$$ . It implies the similarity between the user and item.



Matrix Form:

$$
\mathbf{A}=\begin{bmatrix} \mathbf{0} \;\;\; \;\mathbf{R} \\ \mathbf{R}^T \;\; \mathbf{0} \end{bmatrix} , \;\; \mathbf{E}^{(k+1)}=(\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2})\mathbf{E}^{(k)}
$$

* $$\mathbf{R}$$ is a $$M \times N$$ user-item interaction matrix. Each entries 1 if $$u$$ is connected to $$i$$
* $$\mathbf{D}$$is a $$(M+N)\times(M+N)$$ diagonal matrix, in which each entry $$D_{ii}$$ denotes the number of nonzero entries in the $$i_{th}$$row vector of $$\mathbf{A}$$
* $$\mathbf{E}$$ is a $$(M+N)\times T$$matrix where $$T$$ is the embedding size.



We easily make this as a code using torch\_geomtric.utils ([reference](https://pytorch-geometric.readthedocs.io/en/latest/\_modules/torch\_geometric/utils/get\_laplacian.html))











