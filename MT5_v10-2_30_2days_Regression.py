import numpy as np
import pandas as pd
import json
import csv
import os
import MetaTrader5 as mt5
import sys
from datetime import datetime, timedelta
import pytz
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV



symbols = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "AUDCAD", "AUDJPY", "AUDNZD", "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "NZDJPY", "GBPDKK", "USDDKK", "EURCNH", "EURHKD", "GBPHKD", "HKDJPY", "USDCNH", "USDSGD", "EURCZK", "USDCZK", "USDPLN", "USDMXN", "XAUUSD"]
#symbols = ["USDJPY"]


def connect_to_mt5(login, password, server):
    """Connect to the MetaTrader 5 terminal."""
    if not mt5.initialize(login=login, password=password, server=server):
        print("initialize() failed, error code =", mt5.last_error())
        sys.exit()

def download_and_process_history(symbol, timeframe, start_date, end_date, filename):
    """Download historical data, process it, and save it to a CSV file."""
    timezone = pytz.timezone("Etc/UTC")
    
    # Create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone)
    utc_to = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone)
    
    # Load historical data from MetaTrader 5
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    
    # Shut down the connection to the MetaTrader 5 terminal
    #mt5.shutdown()
    
    # Proceed if data was retrieved
    if rates is not None and len(rates) > 0:
        # Create a DataFrame from the obtained data
        df = pd.DataFrame(rates)
        # Convert time in seconds to the 'datetime' format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename 'tick_volume' column to 'Volume'
        df.rename(columns={'tick_volume': 'Volume'}, inplace=True)
        df.rename(columns={'time': 'Date'}, inplace=True)
        df.rename(columns={'open': 'Open'}, inplace=True)
        df.rename(columns={'high': 'High'}, inplace=True)
        df.rename(columns={'low': 'Low'}, inplace=True)
        df.rename(columns={'close': 'Close'}, inplace=True)
        
        # Drop the 7th and 8th columns (assuming 0-based indexing)
        df.drop(df.columns[[6, 7]], axis=1, inplace=True)
        
        # Save data to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print(f"Failed to get historical data for {symbol}")        
        

def place_order(symbol, order_type, tp_delta):
    """Place an order with a take-profit (TP) level."""
    symbol_info = mt5.symbol_info(symbol)
    # Parameters
    lot_size = 0.01
    current_price = mt5.symbol_info_tick(symbol).bid
    
    if symbol_info is None:
        print(f"{symbol} not found, can not call order_check()")
        mt5.shutdown()
        sys.exit()

    if not symbol_info.visible:
        print(f"{symbol} is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed, exit")
            mt5.shutdown()
            sys.exit()

    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    tp_price = price * (1 + tp_delta) if order_type == mt5.ORDER_TYPE_BUY else price * (1 - tp_delta)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": 0.0,
        "tp": tp_price,
        "deviation": 20,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    # Send a trading request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Failed to send order :(")
        print("retcode={}".format(result.retcode))
        result_dict = result._asdict()
        for field in result_dict.keys():
            print("   {}={}".format(field, result_dict[field]))
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_field in traderequest_dict:
                    print("       traderequest: {}={}".format(tradereq_field, traderequest_dict[tradereq_field]))
    else:
        print("Order placed successfully!")





# Step 1: Load historical data from CSV file
def fetch_stock_data(csv_path):
    df = pd.read_csv(csv_path)
    df.set_index('Date', inplace=True)  # Assuming 'Date' is a column in your CSV. If not, adjust this accordingly.
    df.index = pd.to_datetime(df.index)  # Convert the Date strings to datetime objects
    return df


# Step 2: Create target variable
def create_target_variable(df, direction="BUY", forecast_days=5, buy_threshold=1.005, sell_threshold=0.995):
    df['Target'] = 0
    if direction == "BUY":
        for i in range(len(df) - forecast_days):
            if max(df['High'][i+1:i+1+forecast_days]) >= buy_threshold * df['Close'][i]:
                df.loc[df.index[i], 'Target'] = 1
    elif direction == "SELL":
        for i in range(len(df) - forecast_days):
            if min(df['Low'][i+1:i+1+forecast_days]) <= sell_threshold * df['Close'][i]:
                df.loc[df.index[i], 'Target'] = 1
    return df



def prepare_data(df):
    # Calculate moving averages
    df['MA2'] = df['Close'].rolling(window=2).mean()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA42'] = df['Close'].rolling(window=42).mean()

    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 10 - (10 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=2).min()
    high_max = df['High'].rolling(window=2).max()
    df['Stochastic'] = 10 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stochastic_Signal'] = df['Stochastic'].rolling(window=3).mean()
    
    # Compute shifted close outside of the lambda function
    df['Close_shifted'] = df['Close'].shift(1)

    # Use the computed shifted close inside the lambda function
    df['TR'] = df.apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Close_shifted']), abs(x['Low'] - x['Close_shifted'])), axis=1)

    df['ATR'] = df['TR'].rolling(window=32).mean()

    # On-Balance Volume
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()

    df = df.copy()

    # Parabolic SAR
    df['PSAR'] = df['Close'].copy()  # Use copy() to ensure you don't modify original data unintentionally
    df['PSAR_trend'] = 1
    df['PSAR_EP'] = df['Low']
    df['PSAR_AF'] = 0.02

    for i in range(1, len(df)):
        if df['PSAR_trend'].iloc[i - 1] == 1:
            df.iloc[i, df.columns.get_loc('PSAR')] = df.iloc[i - 1, df.columns.get_loc('PSAR')] + df.iloc[i - 1, df.columns.get_loc('PSAR_AF')] * (df.iloc[i - 1, df.columns.get_loc('PSAR_EP')] - df.iloc[i - 1, df.columns.get_loc('PSAR')])
            if df['Close'].iloc[i] < df['PSAR'].iloc[i]:
                df.iloc[i, df.columns.get_loc('PSAR_trend')] = -1
                df.iloc[i, df.columns.get_loc('PSAR')] = df['High'].iloc[i]
                df.iloc[i, df.columns.get_loc('PSAR_EP')] = df['High'].iloc[i]
                df.iloc[i, df.columns.get_loc('PSAR_AF')] = 0.02
        else:
            df.iloc[i, df.columns.get_loc('PSAR')] = df.iloc[i - 1, df.columns.get_loc('PSAR')] - df.iloc[i - 1, df.columns.get_loc('PSAR_AF')] * (df.iloc[i - 1, df.columns.get_loc('PSAR')] - df.iloc[i - 1, df.columns.get_loc('PSAR_EP')])
            if df['Close'].iloc[i] > df['PSAR'].iloc[i]:
                df.iloc[i, df.columns.get_loc('PSAR_trend')] = 1
                df.iloc[i, df.columns.get_loc('PSAR')] = df['Low'].iloc[i]
                df.iloc[i, df.columns.get_loc('PSAR_EP')] = df['Low'].iloc[i]
                df.iloc[i, df.columns.get_loc('PSAR_AF')] = 0.02


    # Money Flow Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 7
    money_flow = typical_price * df['Volume']
    positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=df.index)  # Convert to Series
    negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=df.index)  # Convert to Series
    positive_flow_sum = positive_flow.rolling(window=54).sum()
    negative_flow_sum = negative_flow.rolling(window=54).sum()
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    df['MFI'] = 10 - (10 / (1 + money_flow_ratio))

    # Commodity Channel Index
    mean_deviation = (df['Close'] - df['Close'].rolling(window=2).mean()).abs().rolling(window=2).mean()
    df['CCI'] = (df['Close'] - df['Close'].rolling(window=2).mean()) / (0.045 * mean_deviation)

    # Update X with the new columns
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA42', 'CCI', 'MA2','Stochastic','Stochastic_Signal']].dropna()
    y = df['Target'].loc[X.index] if 'Target' in df.columns else None

    return X, y if y is not None else X



def time_series_train_test_split(X, y, test_size=0.2):
    test_length = int(len(X) * test_size)
    X_train, X_test = X[:-test_length], X[-test_length:]
    y_train, y_test = y[:-test_length], y[-test_length:]
    return X_train, X_test, y_train, y_test

# Train and Evaluate with a focus on precision
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Using Logistic Regression with Cross-validation and L2 regularization
    clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000, class_weight='balanced', penalty='l2')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # Generate confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    
    return clf, accuracy, report, matrix


def predict_next_2_days(clf, df_latest):
    X_today = df_latest[['Open', 'High', 'Low', 'Close', 'Volume', 'MA42', 'CCI', 'MA2','Stochastic','Stochastic_Signal']].iloc[-1:].reset_index(drop=True)
    prediction = clf.predict(X_today)
    return prediction[0]

def parse_report(report_string):
    lines = [line.strip() for line in report_string.split("\n") if line.strip()]
    
    class_data = {}
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        
        # Check the first part of the line to decide how to process it
        first_part = parts[0]
        
        if first_part.isdigit():  # This is a class line (e.g., "0" or "1")
            class_data[first_part] = {
                'precision': float(parts[1]),
                'recall': float(parts[2]),
                'f1-score': float(parts[3]),
                'support': int(parts[4])
            }
        elif first_part == 'accuracy':
            class_data['accuracy'] = float(parts[1])
        elif first_part == 'macro':
            class_data['macro avg'] = {
                'precision': float(parts[2]),
                'recall': float(parts[3]),
                'f1-score': float(parts[4]),
                'support': int(parts[5])
            }
        elif first_part == 'weighted':
            class_data['weighted avg'] = {
                'precision': float(parts[2]),
                'recall': float(parts[3]),
                'f1-score': float(parts[4]),
                'support': int(parts[5])
            }

    return class_data




if __name__ == "__main__":
    
    results = []
    
    # Connect to MetaTrader 5
    connect_to_mt5(login=xxxxxx, password="xxxxxxx", server="Forex.com-Demo 535")

    # Define parameters
    timeframe = mt5.TIMEFRAME_D1  # Daily timeframe
    start_date = datetime(2015, 1, 1)  # Start date

    # Get current time in UTC and add 2 hours to it
    timezone = pytz.timezone("Etc/UTC")
    current_utc_time = datetime.now(timezone)
    end_date = current_utc_time + timedelta(hours=2)  # End date: GMT time plus 2 hours
    
    for symbol in symbols:
        filename = f"{symbol}_Daily_History.csv"  # CSV filename
        download_and_process_history(symbol, timeframe, start_date, end_date, filename)
        csv_path = f"{symbol}_Daily_History.csv"
        
        
        print(f"\nProcessing {symbol}...")
        df = fetch_stock_data(csv_path)
        df_latest = df[-10000:].copy()

        # For BUY:
        df_buy = create_target_variable(df.copy(), direction="BUY")
        X_buy, y_buy = prepare_data(df_buy)
        X_train_buy, X_test_buy, y_train_buy, y_test_buy = time_series_train_test_split(X_buy, y_buy, test_size=0.2)

        clf_buy, accuracy_buy, report_buy, matrix_buy = train_and_evaluate(X_train_buy, y_train_buy, X_test_buy, y_test_buy)
        print("BUY Model Results:")
        print(f"Accuracy: {accuracy_buy}")
        print("Classification Report:")
        print(report_buy)
        print("Confusion Matrix:")
        print(matrix_buy)

        X_latest, _ = prepare_data(df_latest)
        # Buy prediction for next 2 days
        buy_prediction = predict_next_2_days(clf_buy, X_latest)
        action = "BUY" if buy_prediction == 1 else "NOT BUY"
        print(f"Suggested to {action} for the next 2 days")
        
        buy_report_string = report_buy       
        
        # Replace the next line with your code to get the report_string for the symbol
        buy_report_dict = parse_report(buy_report_string)

        # For SELL:
        df_sell = create_target_variable(df.copy(), direction="SELL")
        X_sell, y_sell = prepare_data(df_sell)
        X_train_sell, X_test_sell, y_train_sell, y_test_sell = time_series_train_test_split(X_sell, y_sell, test_size=0.2)

        clf_sell, accuracy_sell, report_sell,matrix_sell = train_and_evaluate(X_train_sell, y_train_sell, X_test_sell, y_test_sell)
        print("\nSELL Model Results:")
        print(f"Accuracy: {accuracy_sell}")
        print("Classification Report:")
        print(report_sell)
        print("Confusion Matrix:")
        print(matrix_sell)


        # Sell prediction for next 2 days
        sell_prediction = predict_next_2_days(clf_sell, X_latest)  
        action = "SELL" if sell_prediction == 1 else "NOT SELL"
        print(f"Suggested to {action} for the next 2 days")

        
        sell_report_string = report_sell

        
        sell_report_dict = parse_report(sell_report_string)
        # Replace the next line with your code to get the report_string for the symbol
        buy_signal=0
        sell_signal=0
        tp_delta = 0.003
        if (buy_prediction==1) and (buy_report_dict["1"]["precision"] >= 0.7): #and (buy_report_dict["0"]["precision"] > 0.01):
            buy_signal=1
            order_type_buy = mt5.ORDER_TYPE_BUY
            place_order(symbol, order_type_buy, tp_delta)
        if (sell_prediction==1) and (sell_report_dict["1"]["precision"] >= 0.7): #and (sell_report_dict["0"]["precision"] > 0.01):
            sell_signal=1
            order_type_sell = mt5.ORDER_TYPE_SELL
            place_order(symbol, order_type_sell, tp_delta)
        
        #sell_report_dict = parse_report(sell_report_string)
        results.append({
            "Symbol": symbol,
            "BUY_Prediction": buy_prediction,
            "SELL_Prediction": sell_prediction,
            "BUY_1_Precision": buy_report_dict["1"]["precision"],
            "BUY_0_Precision": buy_report_dict["0"]["precision"],
            "SELL_1_Precision": sell_report_dict["1"]["precision"],
            "SELL_0_Precision": sell_report_dict["0"]["precision"],
            "BUY_SIGNAL": buy_signal,
            "SELL_SIGNAL": sell_signal
            })

    mt5.shutdown()
    # This part remains outside the loop, so you write to the CSV only once after processing all symbols
    keys = results[0].keys()
    with open('results_30_2days.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
