# Home Price Prediction & Analysis Using Machine Learning

This project includes a complete pipeline for predicting home prices, starting from data preprocessing to model evaluation. A special focus is placed on regularization techniques to prevent overfitting and enhance model performance.

## Key Features

- **Data Cleaning & Preprocessing**: Addressed missing values and encoded categorical data to prepare the dataset for model training.
  
- **Linear Regression Model**: Built a baseline model using linear regression.
  - **Test Accuracy**: 67.4%
  - **Train Accuracy**: 67.9%
  
- **Regularization Techniques**:
  - **Ridge**: Utilized L2 regularization to improve generalization.
    - **Test Accuracy**: 67.1%
    - **Train Accuracy**: 66.3%
  - **Lasso**: Applied L1 regularization, promoting sparsity in the feature set.
    - **Test Accuracy**: 67.8%
    - **Train Accuracy**: 67.5%
  
- **Model Evaluation**: Compared model performance to assess their accuracy and capability in predicting home prices.

## Why Regularization?

Regularization is applied to reduce overfitting by penalizing large coefficients, ensuring that the model generalizes better to unseen data. Both Ridge and Lasso techniques were employed to demonstrate their effects on the model's accuracy and feature selection.

## What You Will Learn

- End-to-end data preprocessing for machine learning.
- Implementing and evaluating regression models.
- Understanding the impact of regularization techniques on model performance.
- Practical insights into handling real-world datasets with missing values and categorical data.

## Conclusion

In this project, Linear Regression served as the baseline model for predicting home prices. Regularization techniques, such as Ridge and Lasso, were crucial in preventing overfitting and improving generalization on unseen data. While the base model achieved reasonable accuracy, applying Lasso slightly enhanced performance by balancing accuracy with feature selection. This project highlights the importance of data preprocessing, model selection, and the application of regularization in building robust predictive models.

