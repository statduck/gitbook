# Some problems

## Week1

![](<../../.gitbook/assets/image (64).png>)

$$
\hat{y}_i=x_i\dfrac{\sum x_jy_j}{\sum x_j^2}=\sum_m \dfrac{x_ix_m}{\sum_j x_j^2} y_m=a_my_m
$$



![](<../../.gitbook/assets/image (65).png>)

$$
\begin{split}
\hat{\beta} &= (X^TX)^{-1}X^Ty \\
&=(R^TQ^TQR)^{-1}R^TQ^Ty \\
&=R^{-1}Q^Ty  
\end{split}
$$

![](<../../.gitbook/assets/image (66).png>)

```python
import ssl
import pandas as pd
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context
data = pd.read_csv('https://github.com/YonseiESC/ESC-21SUMMER/blob/main/week1/HW/week1_data.csv?raw=True')
y = data['mpg']
x = data.drop(['mpg'],axis=1)
x['horsepower'] = x['horsepower'].replace('?','0').astype(float)

class MyOwnRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def training(self):
        X_t = np.transpose(self.x)
        try:
            beta = np.linalg.inv(X_t@self.x)@X_t@self.y
        except:
            beta = np.linalg.pinv(X_t@self.x)@X_t@self.y
        self.beta = beta
        y_hat = x.values @ beta
        self.y_hat = y_hat
    
    def beta(self):
        return(self.beta)
        
    def predict(self):
        return(self.y_hat)
        
lr = MyOwnRegression(x,y)
lr.training()

lr.beta
lr.predict()
```



## Week2

![](<../../.gitbook/assets/image (69).png>)

```r
install.packages('MASS')
library(MASS)
data <- Boston
summary(data) 
step(lm(medv~1,  data=data), scope=~crim+zn+indus+chas+nox+rm+
age+dis+rad+tax+ptratio+black+lstat,direction = "forward")
```

![](<../../.gitbook/assets/image (72).png>)

![](<../../.gitbook/assets/image (73).png>)

Four variables are included: lstat, rm, ptratio, dis

$$
\hat{y}=-0.67lstat+4.22rm-ptratio-0.55dis+24.47
$$

Adjusted R-squared is 0.6878.



![](<../../.gitbook/assets/image (70).png>)



![](<../../.gitbook/assets/image (74).png>)

![](<../../.gitbook/assets/image (75).png>)

> ref: [https://stats.stackexchange.com/questions/88912/optimism-bias-estimates-of-prediction-error](https://stats.stackexchange.com/questions/88912/optimism-bias-estimates-of-prediction-error)

