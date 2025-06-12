# LSTM Neural Network for Multi-Stock Price Prediction

A comprehensive deep learning approach to intraday stock price forecasting using Long Short-Term Memory (LSTM) neural networks with systematic hyperparameter optimization via grid search.

## 📈 Project Overview

This project implements a sophisticated multi-stock prediction system that forecasts intraday stock prices using historical time series data. The model employs LSTM neural networks to capture complex temporal dependencies in financial markets and uses automated hyperparameter tuning to optimize prediction accuracy.

### 🎯 Key Achievements

- **Multi-Stock Prediction**: Simultaneous forecasting of 30 stock prices using a single unified model
- **61% Directional Accuracy**: Successfully predicts price movement direction with 61% accuracy (significantly above 50% random chance)
- **Temporal Architecture**: Utilizes 10-step lookback windows to predict prices 3 steps ahead (configurable)
- **Optimized Performance**: Grid search across 27 hyperparameter combinations using 3-fold cross-validation
- **Fast Inference**: 13-second prediction runtime enables real-time trading applications

## 🏗️ Model Architecture

### LSTM Network Structure

The neural network consists of:

- **Input Layer**: Sequences of historical stock prices (shape: 10 timesteps × 30 features)
- **LSTM Layer 1**: 60 neurons with sigmoid activation, return sequences enabled
- **Dropout Layer**: 0.2 dropout rate for regularization and overfitting prevention
- **LSTM Layer 2**: 60 neurons with sigmoid activation, no return sequences
- **Dense Output Layer**: Fully connected layer outputting predictions for all 30 stocks

![Network Architecture](images/network_arhcitecture.png)

### Model Configuration (Optimized)

Based on comprehensive grid search results:
- **LSTM Neurons**: 60 (optimal from [50, 60, 70])
- **Activation Function**: Sigmoid (optimal from ['relu', 'tanh', 'sigmoid'])
- **Optimizer**: Adam (optimal from ['adam', 'rmsprop', 'sgd'])
- **Training Epochs**: 200 with early stopping potential
- **Batch Size**: 32
- **Loss Function**: Mean Squared Error (MSE)

## 📊 Data Pipeline & Preprocessing

### Data Processing Steps

1. **Data Loading**: Supports CSV format stock price datasets
2. **Temporal Split**: 80/20 train/test split maintaining chronological order
3. **Normalization**: MinMax scaling to [0,1] range for stable training
4. **Sequence Generation**: Sliding window approach creating supervised learning samples
5. **Inverse Transformation**: Converts normalized predictions back to original price scale

### Dataset Structure

```python
# Example data structure for 30 stocks
dataframe.shape  # (time_periods, 30_stocks)
# Features: Historical closing prices for each stock
# Target: Future prices 3 steps ahead
```

## 🔬 Experimental Results

### Model Performance Metrics

- **Overall MSE**: 72.82 across all 30 stocks
- **Direction Prediction**: 61% accuracy in price movement direction
- **Training Convergence**: Clear loss reduction over 200 epochs
- **Model Parameters**: 2,432 trainable parameters

### Visual Analysis

#### Training Performance
![Loss Over Time](images/loss_over_time.png)
*Training and validation loss curves showing model convergence*

#### Prediction Quality
![Actual vs Predicted Prices](images/actual_vs_predicted_prices.png)
*Comparison of actual vs predicted stock prices demonstrating trend capture*

#### Architecture Comparison
![LSTM vs RNN Comparison](images/compare_lstm_rnn_predictions.png)
*Performance comparison showing LSTM superiority over traditional RNN*

## 🛠️ Technical Implementation

### Dependencies

```python
# Core deep learning framework
keras==2.14.0
tensorflow

# Model optimization and evaluation
scikeras
scikit-learn

# Data manipulation and visualization
pandas
numpy
matplotlib

# Data preprocessing
sklearn.preprocessing.MinMaxScaler
sklearn.model_selection.GridSearchCV
```

### Grid Search Configuration

```python
parameters_grid = {
    'model__lstm_neurons': [50, 60, 70],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'model__optimizer': ['adam', 'rmsprop', 'sgd']
}
# Total combinations: 27 (3×3×3)
# Cross-validation: 3-fold
# Scoring metric: Negative MSE
```

## 🚀 Usage Instructions

### Quick Start

```python
# 1. Data Preparation
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load your stock data (replace with actual dataset)
# dataframe = pd.read_csv('your_stock_data.csv')
np.random.seed(42)
data = np.random.rand(100, 30)  # Demo data
dataframe = pd.DataFrame(data, columns=[f'stock{i}' for i in range(1, 31)])

# 2. Train/Test Split (temporal)
train_ratio = 0.8
training_set_size = int(dataframe.shape[0] * train_ratio)
train_data = dataframe[:training_set_size]
test_data = dataframe[training_set_size:]

# 3. Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
normalize_train_data = scaler.fit_transform(train_data)
normalize_test_data = scaler.transform(test_data)
```

### Model Training

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def create_lstm_model(lstm_neurons=60, dropout=0.2, activation='sigmoid', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(lstm_neurons, input_shape=(10, 30), 
                   activation=activation, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_neurons, activation=activation, return_sequences=False))
    model.add(Dense(30))  # Output for 30 stocks
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Train with optimal hyperparameters
model = create_lstm_model()
history = model.fit(train_X, train_y, epochs=200, batch_size=32, 
                   validation_data=(test_X, test_y), verbose=1)
```

### Hyperparameter Optimization

```python
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Define parameter grid
parameters_grid = {
    'model__lstm_neurons': [50, 60, 70],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'model__optimizer': ['adam', 'rmsprop', 'sgd']
}

# Grid search with cross-validation
model = KerasRegressor(model=create_lstm_model, epochs=100, batch_size=32, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=parameters_grid, 
                   scoring='neg_mean_squared_error', cv=3)
grid_result = grid.fit(train_X, train_y)

print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_}")
```

## 📁 Project Structure

```
LSTM-net-stock-predictor/
├── final_Project_code.ipynb          # Complete implementation with grid search
├── project_report.md                 # Detailed technical report and analysis
├── grid_result_combined.xlsx         # Hyperparameter optimization results
├── images/                           # Visualization assets
│   ├── network_arhcitecture.png      # LSTM architecture diagram
│   ├── loss_over_time.png           # Training/validation loss curves
│   ├── actual_vs_predicted_prices.png # Prediction quality visualization
│   └── compare_lstm_rnn_predictions.png # Architecture comparison
└── README.md                         # This documentation
```

## 🔍 Model Validation & Insights

### Theoretical Foundation

The model addresses key challenges in financial time series:

1. **Vanishing Gradient Problem**: LSTM's gated architecture maintains long-term dependencies
2. **Non-linear Patterns**: Deep learning captures complex market relationships
3. **Multi-variate Dependencies**: Simultaneous modeling of multiple stocks captures market correlations
4. **Temporal Dynamics**: Recurrent structure preserves sequential information

### Performance Analysis

- **Trend Capture**: Successfully identifies general market direction and patterns
- **Practical Feasibility**: 13-second inference time enables real-time applications
- **Statistical Significance**: 61% directional accuracy significantly exceeds random chance
- **Convergence**: Clear training stability with proper regularization

### Limitations & Considerations

1. **Sample Scope**: Limited to IT sector stocks (20-30 stocks)
2. **Computational Constraints**: Grid search optimization limited by available computing resources
3. **Feature Engineering**: Uses only price data; could benefit from technical indicators
4. **Market Efficiency**: Acknowledges random walk theory challenges in high-frequency prediction

## 🔮 Future Enhancements

### Immediate Improvements
- **Real-time Data Integration**: API connections for live stock feeds
- **Technical Indicators**: RSI, moving averages, volume, and momentum features
- **Advanced Optimization**: Bayesian or genetic algorithm hyperparameter search
- **Ensemble Methods**: Combining multiple models for improved robustness

### Research Extensions
- **Attention Mechanisms**: Transformer-based architectures for enhanced temporal modeling
- **Convolutional Networks**: Processing candlestick chart images as alternative input
- **Risk Metrics**: Portfolio optimization and risk-adjusted return calculations
- **Market Regime Detection**: Adaptive models for different market conditions

## 📚 Academic Context

This project demonstrates practical applications of:

- **Financial Machine Learning**: LSTM networks for time series forecasting
- **Hyperparameter Optimization**: Systematic grid search methodology
- **Deep Learning for Finance**: Non-linear pattern recognition in market data
- **Temporal Modeling**: Sequence-to-sequence prediction architectures

### Research Contribution

The study provides empirical evidence for LSTM effectiveness in multi-stock prediction while acknowledging the challenges posed by efficient market hypothesis and random walk behavior in high-frequency financial data.

## 🤝 Usage Notes

- **Data Replacement**: Current implementation uses synthetic data for demonstration
- **Real Implementation**: Replace demo data with actual stock price CSV files
- **Scalability**: Architecture supports different numbers of stocks and prediction horizons
- **Validation**: Maintains temporal integrity in all train/test procedures

## 📄 Additional Resources

- **Technical Report**: `project_report.md` - Comprehensive academic analysis
- **Implementation**: `final_Project_code.ipynb` - Complete code with grid search
- **Results**: `grid_result_combined.xlsx` - Detailed hyperparameter optimization outcomes

---

*This project serves as a foundation for advanced stock prediction research and demonstrates the potential of deep learning in financial forecasting while maintaining awareness of market efficiency principles.*
