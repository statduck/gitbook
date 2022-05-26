# Profiling of Y\&Z Investors

![](<../.gitbook/assets/image (78).png>)

[Presentation File](https://statkwon.github.io/projects/nh/) / [Competition Explanation](https://dacon.io/competitions/official/235663/overview/description)

* **Task**: Visualizing the investment pattern of Y\&Z investors by profiling them.
* **Summary**: First our team suggests why we need to focus on Y\&Z investors. We regard commission as target data and find important variables that well explain our target data in random forest model. Using these variables, we use factor analysis to bind similar features together and naming combined factors as **transaction frequency** and **economic** **power**. With these two factors we cluster the investment data of Y\&Z investor as four clusters by k-means clustering. We compare characteristics for each cluster through frequency analysis, survival analysis, word cloud, and market basket analysis. Based on the result of our analysis, we label each clusters. In final process, we customized marketing strategies considering each characteristics for each cluster.
* **SubProblem**: Clustering investors and Estimating attrition rate with survival analysis
  * Question: How can we capture the attrition rate of original customers.
  * Approach: We made three variables to discriminate dormant customers.

1. Trade break period: The period from the last trade to 30/06/21
2. Total trade period: The period from the first trade to the last trade
3. Expected number of trades: Total trade period / personal trade cycle

&#x20;Using these variables, we can check which customers are dormant customers. I compare the attrition rate by group estimating survival function using Kaplan-Meier method.

* **Evaluation**: The result of cluster is not separated enough. One cluster should be apart from another cluster, but it isn't.&#x20;
* **Period**: 2020.11-2021.02
* **Prize**: Winning prize(입선상)

