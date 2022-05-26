# Parking Demand

![](<../.gitbook/assets/image (189).png>)

[code explanation](https://dacon.io/competitions/official/235745/codeshare/3025?page=1\&dtype=recent)

* **Problem**: We need to predict parking demand for new apartments.
* **Question**: How can we add more variables to make our model more accurately?
* **Approach**: I crawl the data of how many people lives in each apartment. To crawl the data we need to click on each block for apartments, so I used a selenium crawling method to have my computer automatically click on and crawl the data.
* **Model**: We used a polynomial lasso regression model & catboost model. We first appended polynomial features onto our data matrix, including interaction term. It automatically controls the multicollinearity issue by adding the interaction term. After that, with using lasso regression we filtered unnecessary features. We also used a catboost model (it showed a best accuracy score in AutoML with pycaret.) However, this method overfitted our data and lasso model showed better accuracy score. **Because our data has a low number of sample, the lasso model using regularization term well works by reducing the model variance.**
* **Evaluation**: In test cases, there are situations of us not knowing how many people lives in each apartment. For these cases, we need to predict these values leading to a bigger error.&#x20;
* **Period**: 2021.06-2021.08
* **Prize**: Excellence Award





