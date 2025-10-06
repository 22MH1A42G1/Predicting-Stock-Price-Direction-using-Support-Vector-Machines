# Predicting Stock Price Direction using Support Vector Machines

## Overview
Predicting stock price direction is a key goal for traders and analysts. This project uses Support Vector Machines (SVM), a powerful machine learning algorithm, to classify whether a stock's price will rise or fall. The project analyzes historical stock data from Reliance Industries and builds predictive models to forecast price movements.

## Dataset
The project uses the `RELIANCE.csv` dataset containing historical stock price data from **January 2, 2009 to September 16, 2019** with **2,634 trading days**.

### Dataset Features:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Closing price
- **Adj Close**: Adjusted closing price
- **Volume**: Trading volume

## Features Engineering
The model uses two engineered features for prediction:
1. **Open-Close**: Difference between opening and closing prices
2. **High-Low**: Difference between highest and lowest prices

### Target Variable:
- **Binary Classification**: 1 if next day's closing price is higher than today's closing price, 0 otherwise

## Methodology

### 1. Data Preprocessing
- Loaded historical stock data from CSV file
- Set Date as index column
- Created predictor variables (Open-Close, High-Low)
- Generated target variable based on next day's price movement

### 2. Data Split
- **Training Set**: 80% of the data (first 2,107 days)
- **Testing Set**: 20% of the data (remaining 527 days)

### 3. Model Training
Implemented Support Vector Classifier (SVC) with multiple kernel functions:
- **RBF Kernel** (default)
- **Linear Kernel**
- **Polynomial Kernel** (degree=3)
- **Sigmoid Kernel**

### 4. Model Evaluation
Evaluated models using:
- Accuracy Score
- ROC AUC Score
- F1 Score

## Technologies Used
- **Python 3**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning (SVC, accuracy metrics)
- **matplotlib**: Data visualization
- **Google Colab**: Development environment

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/22MH1A42G1/Predicting-Stock-Price-Direction-using-Support-Vector-Machines.git
cd Predicting-Stock-Price-Direction-using-Support-Vector-Machines
```

## Usage

1. Open `Project.ipynb` in Jupyter Notebook or Google Colab
2. Ensure the `RELIANCE.csv` file is in the same directory
3. Run all cells sequentially to:
   - Load and preprocess the data
   - Train the SVM models
   - Evaluate model performance
   - Visualize results

## Results

### Model Performance

#### Initial SVM Model (RBF Kernel):
- **Training Accuracy**: 53.58%
- **Testing Accuracy**: 47.25%
- **ROC AUC**: 0.472
- **F1 Score**: 0.502

#### Kernel Comparison (Test Set Accuracy):
| Kernel | Accuracy |
|--------|----------|
| Linear | 49.15% |
| Polynomial (degree=3) | 48.96% |
| RBF | 47.25% |
| Sigmoid | 50.47% |

### Key Findings
- All tested kernels achieved accuracy around 47-50%, which is close to random guessing (50%)
- The Sigmoid kernel performed slightly better with 50.47% accuracy
- Training accuracy (53.58%) is higher than testing accuracy (47.25%), suggesting possible overfitting
- The model shows limited predictive power with the current feature set

### Strategy Returns
The project also calculates cumulative returns based on the predicted signals:
- **Daily Returns**: Percentage change in closing price
- **Strategy Returns**: Returns based on model predictions
- **Cumulative Returns**: Accumulated returns over time

## Insights and Limitations

### Insights:
1. Stock price prediction using only two features (Open-Close, High-Low) is challenging
2. Different SVM kernels show minimal variation in performance
3. The relatively low accuracy suggests that stock price movements are complex and require more sophisticated features

### Limitations:
1. Limited feature set - only using price-based features
2. No incorporation of external factors (market sentiment, news, economic indicators)
3. No hyperparameter tuning performed
4. Binary classification oversimplifies price movement prediction

## Future Improvements

1. **Feature Engineering**:
   - Add technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
   - Include volume-based features
   - Incorporate sentiment analysis from news and social media
   - Add market indices and sector performance

2. **Hyperparameter Tuning**:
   - Optimize C parameter (regularization)
   - Tune gamma for RBF kernel
   - Adjust degree for polynomial kernel
   - Use GridSearchCV or RandomizedSearchCV

3. **Advanced Models**:
   - Try ensemble methods (Random Forest, XGBoost)
   - Explore deep learning models (LSTM, GRU)
   - Implement multi-class classification for more granular predictions

4. **Cross-Validation**:
   - Implement time-series cross-validation
   - Use rolling window validation
   - Test on multiple stock symbols

5. **Risk Management**:
   - Add stop-loss and take-profit strategies
   - Implement position sizing
   - Calculate Sharpe ratio and other risk metrics

## Project Structure
```
Predicting-Stock-Price-Direction-using-Support-Vector-Machines/
│
├── Project.ipynb          # Main Jupyter notebook with analysis and models
├── RELIANCE.csv          # Historical stock price data
└── README.md             # Project documentation
```

## License
This project is available for educational and research purposes.

## Author
22MH1A42G1

## Acknowledgments
- Dataset: Reliance Industries stock data (2009-2019)
- Machine Learning Framework: scikit-learn
- Development Environment: Google Colab
