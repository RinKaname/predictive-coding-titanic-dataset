# Machine Learning Models Comparison for Titanic Dataset

This document compares the performance of several machine learning models on the Titanic dataset, using 5-fold cross-validation on the training set.

| Model | Mean Accuracy | Standard Deviation |
|---|---|---|
| Gradient Boosting | 82.83% | &plusmn; 2.17% |
| Support Vector Machine | 82.72% | &plusmn; 1.68% |
| Random Forest | 80.59% | &plusmn; 3.42% |
| Logistic Regression | 79.01% | &plusmn; 2.16% |

## Conclusion
Based on the cross-validation results, **Gradient Boosting** is the best performing model among the ones tested.

## Experimental: Predictive Coding Network
We also experimented with the custom `PredictiveCodingNetwork` algorithm found in the accompanying Jupyter Notebook (`titanic-predictivecodingnetwork.ipynb`).

The experimental parameters we tested included:
- Baseline (Notebook Config)
- More Epochs (10)
- High Epochs (50)
- Higher Learning Rates (Weights=0.01, Activities=0.1)
- Lower Learning Rates (Weights=0.001, Activities=0.01)
- Wider Network ([Features, 64, 32, 1])
- Narrower Network ([Features, 16, 8, 1])

**Findings:**
In all hyperparameter configurations tested, the Predictive Coding Network achieved a training accuracy of exactly **61.62%**.

This accuracy corresponds exactly to the proportion of passengers in the training set who did not survive (majority class prediction). This indicates that the custom algorithm, in its current form, suffers from mode collapse and fails to learn meaningful patterns from the features, reverting instead to predicting the most frequent class (0). It is significantly outperformed by standard models like Gradient Boosting (~82.8%).

## Option 2 Experiment: Logistic Regression on PCN Hidden Activities
We also explored using a proper linear classifier (Logistic Regression) on top of the last hidden layer activities (`r[-2]`) instead of relying on the PCN's internal output inference.

**Findings:**
Even with Logistic Regression applied to the extracted hidden features, the training accuracy remained firmly at **61.62%**. This further confirms that the Predictive Coding Network is failing to extract linearly separable or informative features from the dataset, and its internal representations have entirely collapsed to the majority class bias.

## Option 3 Experiment: Precision-Weighted Predictive Coding Network
Following up on the initial mode-collapse, we integrated a "Precision-Weighted" Predictive Coding Network architecture (Version 2). This architecture introduces dynamic precision units (`log_pi`) that learn to weight errors based on reliability, effectively acting as an attention mechanism.

**Findings:**
- With the precision-weighted architecture, the network successfully escapes the mode-collapse (it no longer predicts entirely 0s).
- However, its training accuracy with standard scaling and features hovers around **~45-48%**.
- The user's external tests achieved a Kaggle score of **0.68660** using this architecture, suggesting that while it can learn, it's highly sensitive to feature preprocessing, scaling choices, and hyperparameter tuning compared to robust tree-based models like Random Forests.
