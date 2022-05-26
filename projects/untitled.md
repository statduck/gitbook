---
description: 2020 Predicting Winning Rates, AVG, and ERA by team in KBO
---

# KBO Prediction

![](<../.gitbook/assets/image (112).png>)

&#x20;   The competition is held by [Big-contest](https://www.bigcontest.or.kr/index.php)

* **Problem**: 2020 Prediction of winning rates, avg, and era by team in KBO(Korea Baseball Organization) regular season. The several records until 2020.07 are used to predict several indicators during new season starting from 2020.09.
* **Challenge**: We don't have any record in future(2020.09), so prediction of $$X$$ values is required.&#x20;
* **Approach**: Because the duplicated prediction with $$\hat{X}$$=>$$\hat{Y}$$ shows a high error, so we just predict $$\hat{Y}$$ directly without $$\hat{X}$$. We used univariate time series forecasting model. However, it was hard to find the linear relation between past data and present data, so that we adjust our ARIMA model to fit a non-linear structure by using random forest. This model captures the behavior occurred in the past. Baseball has a tendency in winning, we also make a regression model on data by year(tendency of whole year) and by 10 days(tendency of recent plays). By averaging theses three results, we predicted the indicators for baseball game,
* **Results**: Our model shows the team NC has 0.25 winning rate even it had the highest winning rate 0.628 before 20.09.28. In this period this Team really shows low winning rate: 0.428. Our team advanced to final in big contest.
* **Evaluation**: We could't explain the reason why our model works clearly. To work out this issue, the deeper understanding of time-series analysis was needed.
* **Period**: 2020.07-2020.11
* **Prize**: Advanced to the finals

****







****







****
