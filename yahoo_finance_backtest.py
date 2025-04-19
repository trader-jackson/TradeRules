import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import importlib
import sys

# Import ticker lists and date configurations
from config_tickers import SSE_50_TICKER, NAS_100_TICKER, DOW_30_TICKER
from config import TEST_START_DATE, TEST_END_DATE

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules from pgportfolio
from pgportfolio.tdagent.algorithms.olmar import OLMAR
from pgportfolio.tdagent.algorithms.pamr import PAMR
from pgportfolio.tdagent.algorithms.cwmr_std import CWMR_STD
from pgportfolio.tdagent.algorithms.cwmr_var import CWMR_VAR
from pgportfolio.tdagent.algorithms.rmr import RMR
from pgportfolio.tdagent.algorithms.ons import ONS
from pgportfolio.tdagent.algorithms.up import UP
from pgportfolio.tdagent.algorithms.eg import EG
from pgportfolio.tdagent.algorithms.bk import BK
from pgportfolio.tdagent.algorithms.corn_deprecated import CORN
from pgportfolio.tdagent.algorithms.m0 import M0
from pgportfolio.tdagent.algorithms.best import BEST
from pgportfolio.tdagent.algorithms.bcrp import BCRP
from pgportfolio.tdagent.algorithms.crp import CRP
from pgportfolio.tdagent.algorithms.anticor1 import ANTICOR1
from pgportfolio.tdagent.algorithms.anticor2 import ANTICOR2
from pgportfolio.tdagent.algorithms.olmar2 import OLMAR2
from pgportfolio.tdagent.algorithms.wmamr import WMAMR
from pgportfolio.tdagent.algorithms.sp import SP
from pgportfolio.tdagent.algorithms.ubah import UBAH
from pgportfolio.tdagent.algorithms.min_var import MinVar


def download_yahoo_finance_data(tickers, start_date, end_date, interval='1d'):
    """
    Download data from Yahoo Finance for the given tickers and date range.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d', '1wk', '1mo')
        
    Returns:
        pd.Panel: Panel with items=features, major_axis=tickers, minor_axis=dates
    """
    print(f"Downloading data for {tickers} from {start_date} to {end_date}")
    
    # Download data for each ticker
    data_dict = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if not data.empty:
                data_dict[ticker] = data
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    
    if not data_dict:
        raise ValueError("No data was downloaded for any ticker")
    
    # Create a panel-like structure (since pandas Panel is deprecated)
    # We'll use a dictionary of DataFrames instead
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    dates = sorted(set().union(*[df.index for df in data_dict.values()]))
    
    # Convert to lowercase to match the expected format in pgportfolio
    panel_dict = {}
    for feature in features:
        feature_lower = feature.lower()
        panel_dict[feature_lower] = pd.DataFrame(index=dates, columns=tickers)
        
        for ticker in tickers:
            if ticker in data_dict:
                df = data_dict[ticker]
                if feature in df.columns:
                    panel_dict[feature_lower].loc[df.index, ticker] = df[feature].values.squeeze()
    
    # Forward fill missing values
    for feature in panel_dict:
        panel_dict[feature] = panel_dict[feature].fillna(method='ffill')
        panel_dict[feature] = panel_dict[feature].fillna(method='bfill')  # For any initial missing values
    
    return panel_dict


def prepare_relative_price_data(panel_dict, window_size=10):
    """
    Prepare relative price data for the algorithms.
    
    Args:
        panel_dict (dict): Dictionary of DataFrames with price data
        window_size (int): Window size for the algorithms
        
    Returns:
        tuple: (X, y) where X is the history matrix and y is the price relative vector
    """
    # Use close prices for relative price calculation
    close_prices = panel_dict['close']
    
    # Calculate price relatives (today's price / yesterday's price)
    price_relatives = close_prices / close_prices.shift(1)
    price_relatives = price_relatives.dropna()
    
    # Prepare X and y for the algorithms
    X = []
    y = []
    
    for i in range(len(price_relatives) - window_size):
        X.append(price_relatives.iloc[i:i+window_size].values.T)  # Transpose to get (assets, window_size)
        y.append(price_relatives.iloc[i+window_size].values)  # Next day's price relative
    
    return np.array(X), np.array(y)


def backtest_algorithm(algorithm_class, X, y, **kwargs):
    """
    Backtest a trading algorithm.
    
    Args:
        algorithm_class: The algorithm class to backtest
        X (np.array): History matrix with shape (num_periods, assets, window_size)
        y (np.array): Price relative vector with shape (num_periods, assets)
        **kwargs: Additional arguments for the algorithm
        
    Returns:
        tuple: (portfolio_values, weights_history)
    """
    # Initialize the algorithm
    algorithm = algorithm_class(**kwargs)
    
    # Initialize portfolio
    num_assets = X[0].shape[0]
    weights = np.ones(num_assets) / num_assets  # Equal weight initially
    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    weights_history = [weights.copy()]
    
    # Run the backtest
    for i in range(len(X)):
        # Format the input for the algorithm
        x = X[i].reshape(1, X[i].shape[0], X[i].shape[1])  # Shape: (1, assets, window_size)
        
        # Get the new weights from the algorithm
        try:
            weights = algorithm.decide_by_history(x, weights)
            weights = np.array(weights)
            
            # Ensure weights sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(num_assets) / num_assets
                
        except Exception as e:
            print(f"Error in algorithm: {e}")
            weights = np.ones(num_assets) / num_assets
        
        # Calculate portfolio return
        portfolio_return = np.sum(weights * y[i])
        portfolio_value *= portfolio_return
        
        # Record results
        portfolio_values.append(portfolio_value)
        weights_history.append(weights.copy())
    
    return np.array(portfolio_values), np.array(weights_history)


def run_all_algorithms(X, y, tickers):
    """
    Run all traditional algorithms and compare their performance.
    
    Args:
        X (np.array): History matrix
        y (np.array): Price relative vector
        tickers (list): List of ticker symbols
        
    Returns:
        dict: Dictionary with algorithm names as keys and performance metrics as values
    """
    # Define the algorithms to test
    algorithms = {
        'OLMAR': (OLMAR, {'window': 5, 'eps': 10}),
        'PAMR': (PAMR, {'eps': 0.5}),
        'CWMR_STD': (CWMR_STD, {'eps': 0.5}),
        'CWMR_VAR': (CWMR_VAR, {'eps': 0.5}),
        'RMR': (RMR, {'eps': 0.5, 'window': 5}),
        'ONS': (ONS, {}),
        'EG': (EG, {}),
        'UP': (UP, {}),
        'ANTICOR1': (ANTICOR1, {'window': 5}),
        'ANTICOR2': (ANTICOR2, {'window': 5}),
        'OLMAR2': (OLMAR2, {'window': 5, 'eps': 10}),
        'WMAMR': (WMAMR, {'window': 5}),
        'CRP': (CRP, {}),
        'UBAH': (UBAH, {}),
        'BK': (BK, {'w': 5}),
        'M0': (M0, {}),
        'BEST': (BEST, {}),
        'SP': (SP, {}),
        'MinVar': (MinVar, {'window': 30}),
    }
    
    # Run each algorithm
    results = {}
    for name, (algo_class, params) in algorithms.items():
        try:
            print(f"Running {name}...")
            portfolio_values, weights = backtest_algorithm(algo_class, X, y, **params)
            
            # Calculate performance metrics
            total_return = portfolio_values[-1] / portfolio_values[0] - 1
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
            
            # Calculate annualized return (assuming 252 trading days per year)
            trading_days = len(portfolio_values) - 1
            years = trading_days / 252
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate Calmar ratio (annualized return / maximum drawdown)
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            results[name] = {
                'portfolio_values': portfolio_values,
                'weights': weights,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'annualized_return': annualized_return,
                'calmar_ratio': calmar_ratio
            }
            
            print(f"{name} - Total Return: {total_return:.2%}, Annualized Return: {annualized_return:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}, Calmar Ratio: {calmar_ratio:.2f}")
            
        except Exception as e:
            print(f"Error running {name}: {e}")
    
    return results


def plot_results(results, tickers, start_date, end_date, dataset_name=""):
    """
    Plot the performance of all algorithms.
    
    Args:
        results (dict): Results from run_all_algorithms
        tickers (list): List of ticker symbols
        start_date (str): Start date
        end_date (str): End date
        dataset_name (str): Name of the dataset for file naming
    
    Returns:
        pd.DataFrame: Summary table of algorithm performance
    """
    # Create file prefix based on dataset name
    file_prefix = f"{dataset_name}_" if dataset_name else ""
    
    # Plot portfolio values
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['portfolio_values'], label=name)
    
    title_tickers = ", ".join(tickers[:5])
    if len(tickers) > 5:
        title_tickers += f"... (+{len(tickers)-5} more)"
    
    plt.title(f'Portfolio Value - {dataset_name} ({start_date} to {end_date})')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{file_prefix}portfolio_values.png')
    plt.close()
    
    # Plot performance metrics
    names = list(results.keys())
    total_returns = [results[name]['total_return'] for name in names]
    annualized_returns = [results[name]['annualized_return'] for name in names]
    sharpe_ratios = [results[name]['sharpe_ratio'] for name in names]
    max_drawdowns = [results[name]['max_drawdown'] for name in names]
    calmar_ratios = [results[name]['calmar_ratio'] for name in names]
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))
    
    # Total Returns
    axes[0].bar(names, total_returns)
    axes[0].set_title(f'Total Returns - {dataset_name}')
    axes[0].set_ylabel('Return')
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=90)
    
    # Annualized Returns
    axes[1].bar(names, annualized_returns)
    axes[1].set_title(f'Annualized Returns - {dataset_name}')
    axes[1].set_ylabel('Annualized Return')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=90)
    
    # Sharpe Ratios
    axes[2].bar(names, sharpe_ratios)
    axes[2].set_title(f'Sharpe Ratios - {dataset_name}')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].grid(True)
    axes[2].tick_params(axis='x', rotation=90)
    
    # Max Drawdowns
    axes[3].bar(names, max_drawdowns)
    axes[3].set_title(f'Maximum Drawdowns - {dataset_name}')
    axes[3].set_ylabel('Drawdown')
    axes[3].grid(True)
    axes[3].tick_params(axis='x', rotation=90)
    
    # Calmar Ratios
    axes[4].bar(names, calmar_ratios)
    axes[4].set_title(f'Calmar Ratios - {dataset_name}')
    axes[4].set_ylabel('Calmar Ratio')
    axes[4].grid(True)
    axes[4].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'{file_prefix}performance_metrics.png')
    plt.close()
    
    # Create a summary table
    summary = pd.DataFrame({
        'Algorithm': names,
        'Total Return': [f"{r:.2%}" for r in total_returns],
        'Annualized Return': [f"{r:.2%}" for r in annualized_returns],
        'Sharpe Ratio': [f"{s:.2f}" for s in sharpe_ratios],
        'Max Drawdown': [f"{d:.2%}" for d in max_drawdowns],
        'Calmar Ratio': [f"{c:.2f}" for c in calmar_ratios]
    })
    
    summary = summary.sort_values('Total Return', ascending=False)
    summary.to_csv(f'{file_prefix}algorithm_performance.csv', index=False)
    print(f"\nPerformance Summary for {dataset_name}:")
    print(summary)
    
    return summary


def run_backtest_for_dataset(tickers, dataset_name, start_date, end_date, window_size=10):
    """
    Run backtest for a specific dataset and save results in dataset-specific files.
    
    Args:
        tickers (list): List of ticker symbols
        dataset_name (str): Name of the dataset (e.g., 'SSE50', 'NASDAQ100', 'DOW30')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        window_size (int): Window size for the algorithms
        
    Returns:
        tuple: (results, summary)
    """
    print(f"\n\n{'='*80}\nRunning backtest for {dataset_name} dataset\n{'='*80}")
    print(f"Tickers: {tickers[:5]}... (total {len(tickers)} tickers)")
    print(f"Period: {start_date} to {end_date}\n")
    
    # Create a directory for results if it doesn't exist
    results_dir = f"results_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Download data from Yahoo Finance
        panel_dict = download_yahoo_finance_data(tickers, start_date, end_date)
        print(panel_dict)
        # Prepare data for the algorithms
        X, y = prepare_relative_price_data(panel_dict, window_size)
        
        # Run all algorithms
        results = run_all_algorithms(X, y, tickers)
        
        # Plot and save results with dataset-specific filenames
        plt.figure(figsize=(12, 8))
        for name, result in results.items():
            plt.plot(result['portfolio_values'], label=name)
        
        plt.title(f'Portfolio Value - {dataset_name} ({start_date} to {end_date})')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_dir}/portfolio_values_{dataset_name}.png")
        plt.close()
        
        # Plot performance metrics
        names = list(results.keys())
        total_returns = [results[name]['total_return'] for name in names]
        annualized_returns = [results[name]['annualized_return'] for name in names]
        sharpe_ratios = [results[name]['sharpe_ratio'] for name in names]
        max_drawdowns = [results[name]['max_drawdown'] for name in names]
        calmar_ratios = [results[name]['calmar_ratio'] for name in names]
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 25))
        
        # Total Returns
        axes[0].bar(names, total_returns)
        axes[0].set_title('Total Returns')
        axes[0].set_ylabel('Return')
        axes[0].grid(True)
        axes[0].tick_params(axis='x', rotation=90)
        
        # Annualized Returns
        axes[1].bar(names, annualized_returns)
        axes[1].set_title('Annualized Returns')
        axes[1].set_ylabel('Annualized Return')
        axes[1].grid(True)
        axes[1].tick_params(axis='x', rotation=90)
        
        # Sharpe Ratios
        axes[2].bar(names, sharpe_ratios)
        axes[2].set_title('Sharpe Ratios')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].grid(True)
        axes[2].tick_params(axis='x', rotation=90)
        
        # Max Drawdowns
        axes[3].bar(names, max_drawdowns)
        axes[3].set_title('Maximum Drawdowns')
        axes[3].set_ylabel('Drawdown')
        axes[3].grid(True)
        axes[3].tick_params(axis='x', rotation=90)
        
        # Calmar Ratios
        axes[4].bar(names, calmar_ratios)
        axes[4].set_title('Calmar Ratios')
        axes[4].set_ylabel('Calmar Ratio')
        axes[4].grid(True)
        axes[4].tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/performance_metrics_{dataset_name}.png")
        plt.close()
        
        # Create a summary table
        summary = pd.DataFrame({
            'Algorithm': names,
            'Total Return': [f"{r:.2%}" for r in total_returns],
            'Annualized Return': [f"{r:.2%}" for r in annualized_returns],
            'Sharpe Ratio': [f"{s:.2f}" for s in sharpe_ratios],
            'Max Drawdown': [f"{d:.2%}" for d in max_drawdowns],
            'Calmar Ratio': [f"{c:.2f}" for c in calmar_ratios]
        })
        
        summary = summary.sort_values('Total Return', ascending=False)
        summary.to_csv(f"{results_dir}/algorithm_performance_{dataset_name}.csv", index=False)
        print(f"\nPerformance Summary for {dataset_name}:")
        print(summary)
        
        # Save detailed results to JSON
        results_json = {}
        for algo_name, result in results.items():
            results_json[algo_name] = {
                'total_return': float(result['total_return']),
                'annualized_return': float(result['annualized_return']),
                'sharpe_ratio': float(result['sharpe_ratio']),
                'max_drawdown': float(result['max_drawdown']),
                'calmar_ratio': float(result['calmar_ratio'])
            }
        
        with open(f"{results_dir}/detailed_results_{dataset_name}.json", 'w') as f:
            json.dump(results_json, f, indent=4)
        
        return results, summary
    
    except Exception as e:
        print(f"Error running backtest for {dataset_name}: {e}")
        return None, None


def main():
    # Use test dates from config.py
    start_date = TEST_START_DATE
    end_date = TEST_END_DATE
    window_size = 10  # Window size for the algorithms
    
    # Dictionary to store results for each dataset
    all_results = {}
    
    # Run backtest for SSE50 dataset
    # For Chinese stocks, we need to handle the ticker format
    sse50_tickers = [ticker for ticker in SSE_50_TICKER if ticker.strip()]
    #sse50_results, sse50_summary = run_backtest_for_dataset(sse50_tickers, "SSE50", start_date, end_date, window_size)
    #all_results["SSE50"] = sse50_results
    
    # Run backtest for NASDAQ100 dataset
    nasdaq100_tickers = [ticker for ticker in NAS_100_TICKER if ticker.strip()]
    nasdaq100_results, nasdaq100_summary = run_backtest_for_dataset(nasdaq100_tickers, "NASDAQ100", start_date, end_date, window_size)
    all_results["NASDAQ100"] = nasdaq100_results
    
    # Run backtest for DOW30 dataset
    dow30_tickers = [ticker for ticker in DOW_30_TICKER if ticker.strip()]
    #dow30_results, dow30_summary = run_backtest_for_dataset(dow30_tickers, "DOW30", start_date, end_date, window_size)
    #all_results["DOW30"] = dow30_results
    
    print("\nAll backtests completed. Results saved in dataset-specific directories.")
    
    return all_results


if __name__ == "__main__":
    main()