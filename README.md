# MT5_FOREX_ML_Regression
# **Forex Trading Strategy Automation**

This repository contains a Python script designed to automate trading strategy implementation for various Forex symbols using historical data and predictive modeling. The script connects to **MetaTrader 5 (MT5)** to fetch real-time data, process historical data, train machine learning models, and place orders based on predictions.

---

## **Features**

- **Automates trading for Forex pairs** using predictive models.
- **Historical Data Management**:
  - Downloads and processes historical data from MT5 and saves it as CSV files.
  - Generates technical indicators:
    - Moving Averages (MA2, MA5, MA10, MA42)
    - Relative Strength Index (RSI)
    - Bollinger Bands, MACD, ATR
    - Parabolic SAR, Money Flow Index (MFI), Commodity Channel Index (CCI), Stochastic Oscillator.
- **Predictive Modeling**:
  - Logistic Regression with Cross-Validation (LogisticRegressionCV).
  - "BUY" and "SELL" signal predictions optimized for high precision.
- **Trading Execution**:
  - Places BUY or SELL orders in MT5 based on predictions.
  - Configurable take-profit (`tp_delta`) levels.
- **Reporting**:
  - Outputs results, including predictions and model metrics, into a CSV file.

---

## **Code Overview**

### **1. Data Management**
- **Historical Data**: Fetches historical data for Forex pairs using the MT5 terminal.
- **Processing**: Converts data into CSV format and calculates various technical indicators.

### **2. Predictive Modeling**
- **Target Variables**:
  - "BUY" and "SELL" targets are created based on configurable thresholds.
- **Logistic Regression**:
  - Uses Logistic Regression with Cross-Validation to train models for both "BUY" and "SELL" signals.
- **Precision Focused**:
  - Ensures only high-precision predictions are used for trading decisions.

### **3. Trading Logic**
- **Connection to MT5**: Establishes a secure connection to the MT5 terminal.
- **Order Placement**:
  - Executes BUY or SELL trades based on predictions.
  - Configurable take-profit (`tp_delta`) levels and risk management.

### **4. Results and Reporting**
- **Evaluation**:
  - Outputs classification reports, confusion matrices, and model accuracy.
- **Results File**:
  - Saves results for each symbol in `results_30_2days.csv`.

---

## **Usage Instructions**

### **1. Prerequisites**
- **MetaTrader 5 (MT5)** installed and configured.
- Python 3.8+ with the following libraries installed:
  - `MetaTrader5`, `pandas`, `numpy`, `scikit-learn`, `pytz`.

## **2. Setup**

- **Configure MT5 Credentials**:  
  Replace `xxxxxx` with your **login ID**, `xxxxxxx` with your **password**, and `"Forex.com-Demo 535"` with your **broker server name**.

- **Modify Symbols**:  
  Edit the `symbols` list in the script to include the Forex pairs you wish to trade.

---

## **3. Execution**

- **Run the Script**:  
  Use the following command to execute the script:
  ```bash
  python MT5_v10-2_30_2days_Regression.py

Install dependencies using pip:
```bash
pip install MetaTrader5 pandas numpy scikit-learn pytz
