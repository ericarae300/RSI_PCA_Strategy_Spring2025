from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def calculate_rsi(price, period):
    #calculate price changes, avg gains and losses, and return the rsis
    price_change = price.diff()
    gains = (price_change.where(price_change > 0, 0))
    losses = (-price_change.where(price_change < 0, 0))
    avg_gains = gains.rolling(window = period, min_periods = 1).mean()
    avg_losses = losses.rolling(window = period, min_periods = 1).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100/(1 + rs))
    
    return rsi


def conduct_pca(data, num_components=None):
    # Standardize the RSI data
    data = data.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Apply PCA and store principal components (PCs) into a DataFrame
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i + 1}' for i in range(pca_result.shape[1])], index=data.index)
    
    # Calculate explained variance ratio for each PC
    exp_var_ratio = pca.explained_variance_ratio_

    return pca_df, exp_var_ratio, pca


def walk_forward_pca(data, rsis, pca_df, window_size, test_size, step_size):
    results = []
    
    # Precompute signals outside the loop to avoid recalculation in every iteration
    buy_signals = (pca_df['combined'] > 0) & (rsis['RSI2'] < 30)
    sell_signals = (pca_df['combined'] < 0) & (rsis['RSI2'] > 70)
    
    for i in range(0, len(data) - window_size - test_size, step_size):
        # Set training and testing slices
        training_data = data.iloc[i:i + window_size]
        test_data = data.iloc[i + window_size : i + window_size + test_size]
        training_pca = pca_df.iloc[i:i + window_size]
        test_pca = pca_df.iloc[i + window_size : i + window_size + test_size]
        
        # Use precomputed signals
        buy_sig_train = buy_signals.iloc[i:i + window_size].astype(int)
        sell_sig_train = sell_signals.iloc[i:i + window_size].astype(int)
        training_signals = buy_sig_train - sell_sig_train
        
        buy_sig_test = buy_signals.iloc[i + window_size:i + window_size + test_size].astype(int)
        sell_sig_test = sell_signals.iloc[i + window_size:i + window_size + test_size].astype(int)
        testing_signals = buy_sig_test - sell_sig_test
        
        # Determine true price movement
        true_signal = test_data['close'].pct_change().shift(-1).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).dropna()
        true_signal = true_signal.iloc[:len(testing_signals)]
        true_signal.index = testing_signals.index
        # Calculate accuracy
        correct_predictions = (testing_signals == true_signal).sum()
        num_predictions = len(true_signal)
        
        score = correct_predictions / num_predictions
        results.append({'start': training_data.index[0], 'end': test_data.index[-1], 'score': score})
        
    return pd.DataFrame(results)


def backtest(data, results_df, initial_capital = 100000, risk = 0.02):
    #initialize portfolio value, no position, and lists of trades taken and portfolio value over time
    capital = initial_capital
    position = 0
    trades = []
    portfolio_value_list = []
    portfolio_value = capital
    portfolio_value_list.append(portfolio_value)

    results_df = results_df.drop_duplicates(subset='end', keep='last')
    #iterate through results_df and create series, row, to store the start date, end date, and score created from the walk forward function 
    for i, row in results_df.iterrows():
        start_date = row['start'] 
        end_date = row['end'] 
        score = row['score'] 

        end_date = pd.to_datetime(end_date)

        # Check if 'end_date' exists in data index
        if end_date in data.index:
            # Access the closing price at the given 'end_date'
            buy_price = data.loc[end_date, 'close']
        else:
            print(f"Warning: {end_date} not found in data index.")
            continue  
            
        #if score of date is greater than chance and no position has been taken yet:
        if score > 0.5 and position == 0: 
            #buy at closing price
            position = 1
            capital -= buy_price * risk
            trades.append({'date': end_date, 'action': 'buy', 'price': buy_price})
        elif score < 0.5 and position == 1: 
            #sell
            position = 0
            sell_price = data.loc[end_date, 'close']
            capital += sell_price * risk 
            trades.append({'date': end_date, 'action': 'sell', 'price': sell_price})

        if position == 1:
            position = 1 
            hold_price = data.loc[end_date, 'close']
            portfolio_value = capital + (risk * hold_price)
            portfolio_value_list.append(portfolio_value)
        elif position == 0: 
            portfolio_value = capital
            portfolio_value_list.append(portfolio_value)

    portfolio_value_list = pd.Series(portfolio_value_list, index=results_df['end'])
    # total_returns = (portfolio_value_list[-1] - portfolio_value_list[0]) / portfolio_value_list[0]
    # max_value = portfolio_value_list.cummax()
    # drawdown = (portfolio_value_list - max_value) / max_value
    # max_drawdown = drawdown.max()
    # profitable_trades = sum(1 for trade in trades if trade['action'] == 'sell' and trade['price'] > trades[trades.index(trade) - 1]['price'])
    # win_rate = profitable_trades / len(trades) if trades else 0
    # daily_returns = portfolio_value_list.pct_change().dropna()
    # sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) 

    # print(f"Total Return: {total_returns*100:.2f}%")
    # print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    # print(f"Win Rate: {win_rate*100:.2f}%")
    # print(f"Sharpe Ratio: {sharpe:.2f}")
    
    return portfolio_value_list, trades
            

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def backtest2(data, results_df, initial_capital=100000, risk=0.02):
    # Ensure data has a unique index
    if data.index.duplicated().sum() > 0:
        data = data[~data.index.duplicated(keep='last')]
    
    # Initialize variables
    capital = float(initial_capital)  # Explicitly set as float
    position = 0
    trades = []
    portfolio_value_dict = {}

    # Drop duplicates from 'end' column
    results_df = results_df.drop_duplicates(subset='end', keep='last').reset_index(drop=True)
    
    # Iterate through results_df
    for i, row in results_df.iterrows():
        start_date = row['start']
        end_date = pd.to_datetime(row['end'])
        score = row['score']

        # Check if 'end_date' exists in data index
        if end_date not in data.index:
            print(f"Warning: {end_date} not found in data index.")
            continue
        
        # Get the closing price as a scalar
        current_price = float(data.loc[end_date, 'close'])  # Ensure scalar

        # Trading logic
        if score > 0.5 and position == 0:  # Buy
            position = 1
            capital -= current_price * risk
            trades.append({'date': end_date, 'action': 'buy', 'price': current_price})
        
        elif score < 0.5 and position == 1:  # Sell
            position = 0
            capital += current_price * risk
            trades.append({'date': end_date, 'action': 'sell', 'price': current_price})

        # Update portfolio value
        if position == 1:
            portfolio_value = capital + (risk * current_price)
        else:
            portfolio_value = capital
        
        portfolio_value_dict[end_date] = portfolio_value

    # Convert to Series
    portfolio_value_series = pd.Series(portfolio_value_dict)

    # Calculate metrics
    total_returns = (portfolio_value_series.iloc[-1] - initial_capital) / initial_capital
    max_value = portfolio_value_series.cummax()
    drawdown = (portfolio_value_series - max_value) / max_value
    max_drawdown = drawdown.min()
    daily_returns = portfolio_value_series.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0

    # Calculate profitable trades and win rate
    profitable_trades = 0
    for i in range(1, len(trades), 2):
        if trades[i]['action'] == 'sell' and trades[i]['price'] > trades[i-1]['price']:
            profitable_trades += 1
    win_rate = profitable_trades / (len(trades) // 2) if trades else 0

    # Print results
    print(f"Total Return: {total_returns * 100:.2f}%")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    return portfolio_value_series, trades

# Example usage (assuming data and results_df are defined):
# portfolio_value, trades = backtest2(data, results_df)
    
                          
