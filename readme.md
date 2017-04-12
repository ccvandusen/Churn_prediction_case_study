# Rideshare Churn Analysis
This repo cleans, engineers features, plots, and models rideshare data given for a case study in Galvanize's Data Science Immersive course.

## Introduction
A ride-sharing company is interested in predicting rider retention, and has given us a dataset of a cohort of users who signed up for an account in Janurary 2014. We are tasked with finding the factors that are the best predictors for retention, and to build a predictive model for churn analysis. Our final model uses sklearn's Gradient Boost and our test results had 85% accuracy, 87% recall and 81% precision.

## The Data
![Hist Plot](https://github.com/ccvandusen/churn-prediction-case-study/blob/master/images/hist-plot.png)
The data used for analysis in this repo is from a subset of users who signed up for the ride sharing in Janurary 2014, and ends July 1, 2014. It contains 12 variables and the training set provided contained 40000 rows. The variables were mostly averaged data, such as average driver/passenger rating, average distance travelled, and number of trips in first 30 days. Unforunately, the rideshare data given for this code is proprietary so it cannot be shared.

## The Model
For prediction, we attempted logistic regression, random forest, and boosting. We threw out several variables and dummied and engineered several more for a final feature set of 10 variables. After some initial analysis and throwing out some of the least impactful variables we found it difficult to produce a model with much better validation accuracy than about 89%. Our ROC curve for the predictions on the test set is shown below: ![ROC Curve](https://github.com/ccvandusen/churn-prediction-case-study/blob/master/images/roc-curve.png)

## Findings & What Next
![Feature Importance](https://github.com/ccvandusen/churn-prediction-case-study/blob/master/images/feature-importances.png)
Because the data was mostly aggregated, it was difficult to get much signal from the data. Our feature importance graph shows that the average distance and rating of the driver were the most important variables for the boosted tree predictons. Our suggestions for future analysis would be to more granular data and more features to predict upon. From our inference analysis from our logistic regression model we found one of the cities to enourage churn while the other two to discourage churn, so maybe getting more data from each city and building a model for each one would also improve future analysis and understanding.

