# Kernel Density Estimation

## Kernel Density Estimation

$$
\hat{f}_X (x)=\dfrac{1}{N}\Sigma^n_{i=1}\phi_\lambda(x-x_i)=(\hat{F}\star\phi_\lambda)(x)
$$

&#x20;각 데이터 포인트들을 평균으로 가지는 정규분포들을 더한 후에 데이터의 갯수만큼 나눈 것이 Kernel Density Estimation 결과이다. 이는 $$\hat{F}$$와 $$\phi_\lambda$$의 convolution으로 볼 수 있고 이렇게 보는 경우에 $$\hat{F}$$은 각 관측치에 대해 1/N의 mass를 할당한다고 해석할 수 있다. 아래의 discrete convolution의 정의를 이용한 것이다.&#x20;

$$
(f*g)(m)=\sum _{n}{f(n)g(m-n)}
$$



### f(x) 추정

LDA와 Logistic의 비교에 대해 살펴보자.(p127) Logistic모형은 P(X) 가정이 들어가있지 않다. 대신 P(G|X)를 피팅시켜서 conditional likelihood를 고르는 것이다.

```python
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns

## 데이터가 들어오면 해당 데이터 근처 lambda 범위만큼 탐색한다.
def kernel(x,y,x0,lamb,type=['quad','tri']):
    dist = np.where(abs(x-x0)>1,lamb,abs(x-x0))/lamb
    quad = (3/4*(1-dist**2)); tri=((1-abs(dist)**3)**3)
    message = 'You have to choose type'
    dens = {type=='quad':quad, type=='tri':tri}.get(True, message)
    hat = (dens*y).sum(axis=1)/(dens).sum(axis=1)
    return(hat)
    
dataset = load_boston()
x = dataset['data'].transpose()[-1]
y = dataset['target']
x0 = np.reshape(np.linspace(1,35,100),(-1,1)) #min(x)근처로 주자
y_hat = kernel(x=x,y=y,x0=x0,lamb=3, type='quad')

fig, ax = plt.subplots()
sns.lineplot(x=x,y=y)
sns.lineplot(x=x0.reshape(1,-1)[0],y=y_hat)

def kde(x,x0,lamb,type=['quad','tri'])
    dist = np.where(abs(x-x0)>1,lamb,abs(x-x0))/lamb
    quad = (3/4*(1-dist**2)); tri=((1-abs(dist)**3)**3)
    message = 'You have to choose type'
    dens = {type=='quad':quad, type=='tri':tri}.get(True, message)
    hat = (dens).sum(axis=1) / len(x)*lamb
    return(hat)

fig, ax = plt.subplots()
sns.lineplot(x=x,y=y)
sns.lineplot(x=x0.reshape(1,-1)[0],y=y_hat)
```

![](<../../.gitbook/assets/image (13).png>)



