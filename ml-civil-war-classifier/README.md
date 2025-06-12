# MY474 Machine Learning Assignment #2

[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![RMarkdown](https://img.shields.io/badge/RMarkdown-2.0+-green.svg)](https://rmarkdown.rstudio.com/)

## Overview

This repository contains the complete solution for MY474 Assignment #2, implementing machine learning algorithms from scratch and applying them to real-world datasets. The assignment covers gradient descent optimization, regularized regression, and binary classification.

## 📁 Repository Structure

```
├── MY474_Assignment2_Final.Rmd    # Main R Markdown document
├── paste.txt                      # Assignment instructions
├── civil_wars.RData              # Dataset for Exercise 3
└── README.md                      # This file
```

## 🎯 Assignment Objectives

### Exercise 1: Gradient Descent Algorithm Analysis (20 marks)
- **Objective**: Compare Stochastic Gradient Descent (SGD) with SGD + Momentum
- **Implementation**: Custom algorithms built from scratch
- **Analysis**: Performance comparison, convergence behavior, and theoretical discussion

### Exercise 2: Custom Regularized Regression Function (35 marks)
- **Objective**: Build an automated model selection function for polynomial regression
- **Features**: Ridge regularization, cross-validation, hyperparameter optimization
- **Output**: Optimal polynomial degree, lambda value, and trained model

### Exercise 3: Civil War Prediction Model (45 marks)
- **Objective**: Binary classification on real-world political science data
- **Method**: LASSO logistic regression with feature selection
- **Dataset**: 53 socioeconomic/political predictors across country-years

## 🛠️ Dependencies

```r
# Required packages
library(tidyverse)    # Data manipulation and visualization
library(glmnet)       # LASSO/Ridge regression
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd MY474-Assignment2
   ```

2. **Install dependencies**
   ```r
   install.packages(c("tidyverse", "glmnet"))
   ```

3. **Run the analysis**
   ```r
   # Open in RStudio and knit to HTML
   rmarkdown::render("MY474_Assignment2_Final.Rmd")
   ```

## 📊 Key Results

### Exercise 1 Findings
- **SGD with Momentum**: 🚀 Faster convergence but potential oscillation
- **Traditional SGD**: 🎯 More stable but slower convergence
- **Recommendation**: Nesterov momentum for optimal balance

### Exercise 2 Implementation
- **Grid Search**: 210 hyperparameter combinations tested
- **Cross-Validation**: 5-fold CV for robust model selection
- **Features**: Polynomial degrees 1-7, ridge regularization

### Exercise 3 Performance
- **Test Accuracy**: 99.16% 🎯
- **Feature Selection**: Reduced from 53 to 34 predictors
- **Model Type**: LASSO logistic regression

## 🔧 Technical Features

### Custom Implementations
- ✅ Stochastic Gradient Descent from scratch
- ✅ SGD with Momentum algorithm
- ✅ Ridge regression with polynomial features
- ✅ K-fold cross-validation
- ✅ Grid search optimization

### Advanced Techniques
- 📈 Regularization (L1 and L2 penalties)
- 🔄 Cross-validation strategies
- 🎛️ Hyperparameter tuning
- 📊 Model comparison and evaluation

## 📈 Visualizations

The analysis includes:
- Convergence plots comparing SGD algorithms
- LASSO coefficient paths showing feature selection
- Performance tables and confusion matrices
- Algorithm comparison summaries

## 📝 Academic Context

**Course**: MY474 - Applied Machine Learning  
**Institution**: [University Name]  
**Term**: Winter Term 2024  
**Weight**: 40% of final grade

## 🤝 Contributing

This is an academic assignment submission. For questions about the methodology or implementation, please refer to the course materials or contact the instructor.

## 📄 License

This project is submitted for academic evaluation. Please respect academic integrity policies when referencing this work.

## 📞 Contact

For questions about this implementation:
- Student ID: 29189
- Course: MY474
- Instructor: Dr Thomas Robinson

---

*Note: This assignment demonstrates proficiency in machine learning fundamentals, algorithm implementation, and practical model development skills.*