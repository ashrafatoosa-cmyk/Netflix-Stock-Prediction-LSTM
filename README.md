# Netflix Stock Price Prediction using LSTM

This repository features a Deep Learning project that forecasts Netflix (NFLX) stock prices using a Long Short-Term Memory (LSTM) neural network built with TensorFlow and Keras. The project demonstrates a full machine learning pipeline, including data preprocessing, time-series sequence creation, hyperparameter tuning, and statistical evaluation.

## Project Overview
Predicting stock prices is a complex time-series challenge due to market volatility. This project utilizes a historical dataset of Netflix stock prices to train an LSTM network, which excels at capturing long-term temporal dependencies and trend momentum. 

The dataset is chronologically split into **80% for training** and **20% for testing** to ensure realistic evaluation on unseen data.

## Key Features & Workflow
- **Data Source:** Historical stock data fetched from Kaggle.
- **Time-Series Preparation:** Transformed regular tabular data into 3D historical sequences using a 60-day sliding window approach.
- **Data Normalization:** Scaled prices between 0 and 1 using `MinMaxScaler` to improve gradient descent stability during training.
- **Architecture:** A deep stacked LSTM network with Dropout layers to prevent overfitting.
- **Metrics:** Evaluated model performance using both absolute scale metrics and standardized percentage metrics.

## Model Performance & Evaluation Results
After tuning the architecture by increasing the unit depth and stacking LSTM layers, the model achieved highly accurate results on the testing dataset:

- **Mean Absolute Error (MAE):** $19.18
- **Weighted Absolute Percentage Error (WAPE):** 3.44%

A WAPE of 3.44% means the model operates at an approximate 96.56% overall accuracy relative to the volume scale of the stock price, making it highly reliable for trend forecasting.

## Technologies Used
- Python 3
- Google Colab
- TensorFlow / Keras
- Scikit-Learn
- Pandas & NumPy
- Matplotlib (for data visualization)

## How to Run the Project
1. Open the `.ipynb` notebook file directly in Google Colab or Jupyter Notebook.
2. Upload the `NFLX.csv` dataset into your workspace directory.
3. Run the cells sequentially to preprocess data, train the model, and view the evaluation visualizations.
