# Cricket Match Outcome Prediction

This project investigates the possibility of predicting the winner during the second innings of a cricket match. By applying different machine learning methods and innovative shrinkage strategies, the project aims to provide accurate predictions that could be useful for captains, coaches, and broadcasting teams to enhance strategic decisions during live matches. This repository only contains the code for the web application.

## Table of Contents
1. [Motivation](#motivation)
2. [Objectives](#objectives)
3. [Data Modeling and Preprocessing](#data-modeling-and-preprocessing)
4. [Summary of Results](#summary-of-results)
5. [Technologies Used](#technologies-used)
6. [Usage](#usage)

## Motivation

Real-time winning predictions during cricket matches keep the broadcast engaging for viewers. This inspired us to create a model that predicts the winner as the second innings progresses, allowing viewers to better understand the evolving match dynamics. For team management, this tool can offer insights to improve on-field decision-making. The availability of ball-by-ball datasets for public use further motivated us to utilize this data to develop and compare different machine learning models.

## Objectives

The project aims to achieve the following:
1. Identify key factors that influence the outcome of a match during the second innings.
2. Develop machine learning models based on these predictors.
3. Deploy the models in a web-based dashboard to predict match outcomes in real time.

## Data Modeling and Preprocessing

### Dataset Overview

The dataset, sourced from [Kaggle (Jamie Welsh)](https://www.kaggle.com/datasets/jamiewelsh2/ball-by-ball-it20), includes ball-by-ball data from T20 cricket matches between 2005 and 2023. The dataset contains 425,119 records with 34 attributes describing each delivery. After filtering for second-innings data, 200,304 observations were used in the analysis.

### Problem Definition

The prediction task is modeled as a classification problem, where the target variable (*Chased Successfully*) is binary:
- **0**: Chasing team lost.
- **1**: Chasing team won.

### Predictor Selection

Thirteen key predictors were chosen based on their practical relevance to the match outcome:
1. Runs Required
2. Balls Remaining
3. Current Score
4. Balls Delivered
5. Wickets Remaining
6. Target Score
7. Current Run Rate (CRR)
8. Required Run Rate (RRR)
9. Striker's Score
10. Balls Faced by Striker
11. Non-Striker's Score
12. Balls Faced by Non-Striker
13. Runs Conceded by Bowler

### Feature Engineering

Derived predictors such as **Balls Delivered**, **Wickets Remaining**, **CRR**, and **RRR** were included. Missing values in CRR and RRR were imputed with their mean and maximum values, respectively.

### Tools and Libraries

The following Python libraries were used:
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For machine learning model development.
- **Streamlit**: For building the web-based dashboard.

## Summary of Results

The table below summarizes the performance of each classifier based on accuracy when applied to the test dataset:

| **Classifier**                   | **Accuracy (%)**       |
|-----------------------------------|------------------------|
| Logistic Regression               | 81.89                  |
| Penalized Logistic Regression (LASSO) | 81.89              |
| Shrinkage Estimation              | 81.94                  |
| Positive Shrinkage Estimation     | 81.94                  |
| Linear Shrinkage Estimation       | 81.89                  |
| Pretest Estimation                | 81.96                  |
| Shrinkage Pretest Estimation      | 81.96                  |
| Gradient Boosting Machine         | 91.66                  |

The **Gradient Boosting Machine** model performed the best, achieving an accuracy of 91.66%, while the other classifiers showed accuracy between 81.89% and 81.96%.

## Technologies Used

- **Python**: Programming language used for model development.
- **NumPy**: Numerical computation.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Machine learning library.
- **Streamlit**: Framework for building the web-based dashboard.

## Usage

The Streamlit app is available in the **Web Application** folder, and the model notebook can be found in the **Model** folder. You can access the app by visiting [https://match-predictor.onrender.com/](https://match-predictor.onrender.com/).