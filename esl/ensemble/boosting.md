# Boosting

## Idea

* It is the method combining several classifiers in a sequential way.
* Correctly classified data doesn't need to be considered in our model, so it puts a weight on wrong classified data.&#x20;

## Notation

* Classifiers: $$H=\{h_1,\cdots,h_m\}, \;s.t. \;h_j:X\rightarrow\{1,-1\}$$
* Data: $$X=\{x_1,\cdots,x_n\}, \;Y=\{y_1,\cdots,y_n\}, \;y_i\in\{1,-1\}$$
* Matrix: $$A_{ij}=y_ih_j(x_i) \;$$$$A_{ij}=1(correct),A_{ij}=-1(wrong)$$
* Weight for classifiers: $$w\in\mathbb{R}^m$$
* Weight f or data: $$\lambda \in \mathbb{R}^n$$

## Problem

$$
p(\lambda)=min\{w^TA\lambda:w\in\Delta_n\}, \;s.t. \;\Delta_n=\{w|\Sigma^n_{i=1}=1,w_i \geq0\}
$$

$$
(A\lambda)=\begin{bmatrix}A_1 \cdots A_m\end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_m\end{bmatrix}, w^TA\lambda=w^TA_1\lambda_1+\cdots+w^TA_m\lambda_m
$$

$$
\max_{\lambda \in \Delta_m} \{p(\lambda)=\min_{w\in \Delta_n}w^TA\lambda\} \\ \min_{w\in \Delta_n}\{f(w)=\max_{\lambda \in \Delta_m}w^TA\lambda\}
$$

&#x20;The above optimization problem is an original problem, and the below one is the duality problem.&#x20;

In this situation, if $$A_j\lambda_j$$is lower than other values, it means the classifier $$A_j$$is not good at predicting. So $$p(\lambda)$$ means the lowest value which the worst classifier has. When we make more weight on the worst classifier by making $$\lambda$$ bigger, $$p(\lambda)$$would be bigger.





To solve this problem, we can use subgradient method or mirror descent method.





## Subgradient method



The optimization problem is defined as:

$$
\max_{\lambda \geq 0 }p(\lambda)
$$



















