# 비모수 로지스틱 회귀

패널 방식을 적용한다는 이야기는 X매트릭스 대신에 범위함수가 들어간 N매트릭스를 사용한다는 것이다.&#x20;



&#x20;**penalized log-likelihood criterion**&#x20;

$$
\begin{split}
l(f;\lambda) & =\sum^N_{i=1}[y_i logp(x_i)+(1-y_i)log(1-p(x_i))]-\dfrac{1}{2} \lambda \int \{f''(t)\}^2dt \\
& =  \sum^N_{i=1}[y_i f(x_i)-log(1+e^{f(x_i)}\}-\dfrac{1}{2} \lambda \int \{f''(t)\}^2dt

\end{split}
$$

$$
\dfrac{\partial l(\theta)}{\partial \theta}=N^T(y-p)-\lambda \Omega \theta \\
\dfrac{\partial^2 l(\theta)}{\partial \theta \partial \theta^T} = -N^TWN-\lambda \Omega
$$

$$
\begin{split}
\theta^{new}& =(N^TWN+\lambda \Omega)^{-1}N^TW(N \theta^{old}+W^{-1}(y-p)) \\
& = (N^TWN+\lambda \Omega)^{-1}N^TWz

\end{split}
$$
