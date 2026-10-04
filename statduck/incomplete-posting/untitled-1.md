# Ch6(editing)

#### FCB 6.6

한 쪽에 고여서 샘플링이 더 되어서 반복횟수를 늘림. 그 결과 Naive SE가 줄었음. 또한 반복을 진행하다보니 음수 부분에도 샘플링이 많이 되었다. Trace plot을 통해 추출이 제대로 되었는지 평가할 수 있다.



시각적인 평가 이외에 BGR Statistics을 이용해서 MCMC가 제대로 수렴했는지를 평가한다. Chain들의 within variance, between variance comparison!



$$
W = \frac{1}{M}\Sigma^M_{j=1}s^2_j  \\
B = \frac{N}{M-1}\Sigma^M_{j=1} (\bar{\theta}_j-\bar{\theta})^2
$$

이 때 B는 overestiamte을 시키는 주범이다. 다양한 체인들간의 평균인데 각 초기값이 다르므로, 다른 초기값에 대한 분산이 고려되어 그 분산보다 큰 값이 나옴. W의 경우 한 초기값이 설정된 체인 내에서의 분산이므로 전체 분산보다 작다.

$$
\hat{var}(\theta)=(1-\frac{1}{N})W+\frac{1}{N}B \\
\hat{R}=\sqrt{\frac{\hat{var}(\theta)}{W}}
$$

&#x20;위의 통계량이 unbiasedness를 만족한다. Shrinkage factor를 나눠서 값을 만든다.

Autocorrelation이 존재하는데, iid가정에 비해 얼마나 영향을 가지는지 보고싶다. 그래서 ACF(Autocorrelation Function)이 나옴.  Correlation이 빨리 작아지면 좋은 것이고 함수 식은 다음과 같다.&#x20;

$$
acf(\theta)=\dfrac{\Sigma^{n-k}_{i=1}(\theta_i-\bar{\theta})(\theta_{i+k}-\bar{\theta})}{\Sigma^n_{i=1}(\theta_i-\bar{\theta})^2}
$$



ESS(Effective Sample Size)를 구하기 위해서 다음의 식을 이용한다.

$$
lim_{N->\infty}MNvar(\bar{\bar{{\theta}}})=(1+2\Sigma^\infty_{k=1}acf_k)var(\theta)
$$

Correlated 된 샘플을 버리고 쓰는것은 thinning이라고 하는데 샘플한 것의 정보를 버리게 된다.&#x20;

실제 수렴 여부는 알 수 없기에, trace plot으로 모양이 이상한지만 판단한다.



HW

MCMC 직접 샘플링 하는 과제. 책의 가이드라인에 따라 accept/reject 비율 고려하고 여러 진단 플랏 그리기.&#x20;

