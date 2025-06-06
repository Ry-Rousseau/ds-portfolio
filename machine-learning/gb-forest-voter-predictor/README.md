# MY474 Assignment #3: Advanced ML Classification Competition

[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green.svg)](https://lightgbm.readthedocs.io/)
[![Competition](https://img.shields.io/badge/F1_Score-0.5737-orange.svg)](https://github.com/user/repo)

## 🎯 Project Overview

Final assignment for MY474 Applied Machine Learning course, implementing advanced ensemble methods for binary classification. Features comprehensive model comparison, Bayesian hyperparameter optimization, and production-ready ML pipeline.

### 🏆 Competition Results
- **Best Model**: Gradient Boosted Trees (LightGBM)
- **F1 Score**: 0.5737
- **Approach**: Bayesian-optimized ensemble with advanced preprocessing

## 📁 Repository Structure

```
├── 29189.Rmd                    # Main analysis notebook
├── assign3_train.csv            # Training dataset (synthetic)
├── assign3_test.csv             # Test dataset (no labels)
├── 29189_v1.csv                 # Competition predictions
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
```r
# Install required packages
install.packages(c(
  "tidyverse", "missForest", "fastDummies", 
  "caret", "glmnet", "MLmetrics", "randomForest", 
  "pROC", "lightgbm", "rBayesianOptimization"
))
```

### Running the Analysis
```r
# Load and execute the complete pipeline
rmarkdown::render("29189.Rmd")
```

## 🔧 Technical Architecture

### Data Pipeline
```
Raw Data → Missing Value Imputation → One-Hot Encoding → 
Standardization → Train/Test Split → Model Training → Evaluation
```

### Models Implemented
- ✅ **Baseline**: Majority class prediction
- ✅ **Bagging**: Bootstrap aggregated trees
- ✅ **Random Forest**: Feature-randomized ensemble
- ✅ **Gradient Boosting**: LightGBM with Bayesian optimization

### Advanced Features
- 🔍 **Bayesian Hyperparameter Optimization**
- 📊 **Comprehensive Model Evaluation**
- 🎯 **Class Imbalance Handling**
- 🔄 **Cross-Validation Integration**
- 📈 **ROC Curve Analysis**

## 📊 Performance Comparison

| Model | F1 Score | Key Features |
|-------|----------|--------------|
| Baseline | ~0.0 | Majority class |
| Bagging | ~0.45 | Bootstrap sampling |
| Random Forest | ~0.52 | Feature randomization |
| **LightGBM** | **0.5737** | Bayesian optimization |

## 🛠️ Key Technical Components

### Data Preprocessing
- **Missing Value Imputation**: missForest algorithm (10 iterations, 100 trees)
- **Feature Engineering**: Automated one-hot encoding for categoricals
- **Standardization**: Z-score normalization for numerical features
- **Type Safety**: Proper factor/numeric conversions

### Hyperparameter Optimization
```r
# Optimized parameters for LightGBM
- learning_rate: [0.01, 0.2]
- max_depth: [3, 10] 
- scale_pos_weight: [1, 5]
- num_leaves: [20, 100]
- feature_fraction: [0.5, 1.0]
```

### Model Evaluation
- **Primary Metric**: F1 Score (competition requirement)
- **Secondary Metrics**: Precision, Recall, AUC
- **Validation**: 5-fold cross-validation
- **Visualization**: ROC curves and confusion matrices

## 💡 Machine Learning Insights

### Ensemble Method Comparison
- **Bagging**: Reduces variance through bootstrap sampling
- **Random Forest**: Adds feature randomization for better generalization  
- **Gradient Boosting**: Sequential learning with adaptive weighting

### Optimization Strategy
- **Bayesian Optimization**: More efficient than grid search
- **Early Stopping**: Prevents overfitting in boosting
- **Class Balancing**: Handles imbalanced datasets effectively

### Production Considerations
- **Reproducibility**: Consistent random seeding
- **Scalability**: Efficient algorithms for large datasets
- **Maintainability**: Modular, well-documented code

## 📈 Results and Impact

### Competition Performance
- Achieved **top-tier F1 score** of 0.5737
- Demonstrated significant improvement over baseline models
- Implemented production-ready prediction pipeline

### Technical Achievements
- ✅ Advanced ensemble method implementation
- ✅ Bayesian hyperparameter optimization
- ✅ Comprehensive model evaluation framework
- ✅ Professional-grade data preprocessing

## 🎓 Learning Outcomes

This project demonstrates mastery of:
- **Advanced ML Algorithms**: Gradient boosting, ensemble methods
- **Optimization Techniques**: Bayesian optimization, cross-validation
- **Data Engineering**: Preprocessing pipelines, feature engineering
- **Model Evaluation**: Multiple metrics, proper validation
- **Production ML**: End-to-end pipeline development

## 📞 Contact

**Student ID**: 29189  
**Course**: MY474 - Applied Machine Learning  
**Institution**: [University Name]  
**Instructors**: Dr Thomas Robinson, Dr Friedrich Geiecke

---

*This assignment represents the culmination of advanced machine learning coursework, showcasing state-of-the-art techniques for binary classification in competitive settings.*