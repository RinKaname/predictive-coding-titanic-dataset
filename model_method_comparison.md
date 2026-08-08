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

## Option 4 Experiment: Precision-Weighted PCN (Reduced Inference Steps)
Building on the Version 2 architecture, we tested a "Version 3" configuration where the number of iterative inference steps (`n_inference_steps`) was reduced from 50 to 30.

**Findings:**
- Reducing the inference steps acts as a form of early stopping during the inference phase, preventing the network from overfitting its internal representations to noisy or difficult samples.
- While local training accuracy remained relatively low (~43%), the user's external tests demonstrated a noticeable boost in generalization, achieving a Kaggle score of **0.70334**. This highlights that finding the right balance of inference iterations is crucial for the generalization of Predictive Coding Networks.
