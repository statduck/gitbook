# Model Assessment & Selection

## Model Complexity

$$
Err_\tau =E[L(Y,\hat{f}(X)|\mathcal{T}] \\
Err=E[L(Y,\hat{f}(X))]=E[Err_\mathcal{T}]
$$

$$Err_\tau$$is **a test error**, and $$Err$$is **an expected test error**.

![](<../../.gitbook/assets/image (1).png>)

&#x20;In this situation, we all talk about random data so we can't get the exact value of this error. Conditioned on $$\mathcal{T}$$, random elements of $$\mathcal{T}$$become realization.

$$
Err_\mathcal{T}=E_{X^0,Y^0}[L(Y^0,\hat{f}(X^0))|\mathcal{T}] \\
Err=E_\mathcal{T}E_{X^0,Y^0}[L(Y^0,\hat{f}(X^0))|\mathcal{T}]
$$

$$\mathcal{T}=\{{(x_1,y_1),\dots,(x_N,y_N)\}}$$ It is a realization version of random quantity. $$X^0, Y^0$$refer to sample randomly chosen from test set. We want to predict $$Err_\tau$$, but it is hard to predict. Instead of $$Err_\tau$$, we'll predict **the expected error** $$Err$$.



&#x20;We predict this error for two reasons:

**Model selection**: estimating the performance of different models in order to choose the best one.

**Model assessment**: having chosen a final model, estimating its prediction error(generalization error) on new data.

&#x20;In an ideal situation, we split our data into three parts: Train(0.5), Validation(0.25), and Test(0.25). In train set we fit our model to data, and select model in validation set(Most well performed model in validation set). After that, we predict  $$Err_\mathcal{T}$$ of our final model and assess this model.

****

##



