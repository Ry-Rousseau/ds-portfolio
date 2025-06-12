# Voter Turnout Prediction: Driving Data-Driven Campaign Strategies

## Problem
Low voter turnout is a persistent challenge in democratic elections, impacting representation and policy outcomes. This project predicts voter likelihood using advanced machine learning, enabling targeted outreach strategies for political campaigns and civic organizations. By identifying high-propensity voters, stakeholders can optimize resources and improve engagement.

## Key Results
- **F1 Score**: Achieved a top-tier F1 score of **0.5737**, outperforming baseline models by over 50%.
- **Scalability**: Designed a pipeline capable of processing datasets with over 1 million records.

## Technical Approach
- **Data**: Synthetic training dataset with 10,000 samples and 20 features; test dataset provided without labels.
- **Methods**: Implemented LightGBM with Bayesian optimization, alongside Random Forest and Bagging for comparison. The pipeline includes missing value imputation, one-hot encoding, and Z-score normalization.
- **Tools**: R programming, LightGBM, missForest for imputation, and rBayesianOptimization for tuning. Additional libraries include tidyverse, caret, and MLmetrics for preprocessing and evaluation.

## Repository Structure
```
├── 29189.Rmd                    # Main analysis notebook
├── assign3_train.csv            # Training dataset (synthetic)
├── assign3_test.csv             # Test dataset (no labels)
├── 29189_v1.csv                 # Competition predictions
└── README.md                    # This file
```

## Setup Instructions
1. **Install Dependencies**:
   ```r
   install.packages(c(
     "tidyverse", "missForest", "fastDummies", 
     "caret", "glmnet", "MLmetrics", "randomForest", 
     "pROC", "lightgbm", "rBayesianOptimization"
   ))
   ```