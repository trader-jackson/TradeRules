# Rule-Based Portfolio Backtesting Framework

## Overview

This is a comprehensive backtesting framework for evaluating rule-based trading strategies, with a focus on mean reversion and trend following approaches. The framework implements a wide range of portfolio optimization algorithms and provides tools for performance analysis and visualization.

## Features

- **Multiple Trading Algorithms**: Implementation of 19+ traditional portfolio optimization algorithms including:
  - Mean Reversion: OLMAR, PAMR, CWMR, RMR
  - Trend Following: Follow-the-Winner, Follow-the-Loser
  - Universal Portfolios: UP, EG, ONS
  - Benchmark Strategies: UBAH (Buy and Hold), CRP (Constant Rebalanced Portfolio)

- **Multi-Market Support**: Backtest on various market indices:
  - US Markets: NASDAQ 100, DOW 30
  - Chinese Markets: SSE 50, CSI 300

- **Performance Metrics**: Comprehensive performance evaluation with metrics such as:
  - Total Return
  - Annualized Return
  - Sharpe Ratio
  - Maximum Drawdown
  - Calmar Ratio

- **Data Integration**: Yahoo Finance API integration for easy data acquisition

- **Visualization**: Automated generation of performance charts and comparison tables

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run a backtest on the default datasets (NASDAQ 100, DOW 30, SSE 50):

```bash
python yahoo_finance_backtest.py
```

This will:
1. Download historical price data from Yahoo Finance
2. Run all implemented algorithms on the data
3. Generate performance metrics and visualizations
4. Save results to dataset-specific directories

### Configuration

You can modify the following configuration files to customize your backtest:

- `config.py`: Configure date ranges and technical indicators
- `config_tickers.py`: Modify the ticker lists for different market indices

### Implementing Custom Algorithms

To implement a custom trading algorithm, create a new class in the `pgportfolio/tdagent/algorithms/` directory that inherits from the `TDAgent` base class and implements the required methods.

## Implemented Algorithms

### Mean Reversion
- **OLMAR**: Online Moving Average Reversion
- **PAMR**: Passive Aggressive Mean Reversion
- **CWMR**: Confidence Weighted Mean Reversion
- **RMR**: Robust Median Reversion

### Follow-the-Winner
- **BEST**: Best Constant Rebalanced Portfolio
- **BCRP**: Best Constant Rebalanced Portfolio (in hindsight)
- **UP**: Universal Portfolio

### Follow-the-Loser
- **ANTICOR**: Anti-correlation strategy
- **WMAMR**: Weighted Moving Average Mean Reversion

### Pattern-Matching
- **BK**: Bk strategy
- **CORN**: Correlation-driven Nonparametric Learning

### Others
- **ONS**: Online Newton Step
- **EG**: Exponentiated Gradient
- **SP**: Switching Portfolio
- **CRP**: Constant Rebalanced Portfolio
- **UBAH**: Uniform Buy and Hold
- **MinVar**: Minimum Variance Portfolio

## Results

Results are saved in dataset-specific directories (e.g., `results_NASDAQ100`, `results_DOW30`, `results_SSE50`) and include:

- CSV files with performance metrics
- JSON files with detailed results
- Visualizations of portfolio values and performance metrics

## License

This project is available for academic and research purposes.

## Acknowledgements

This framework builds upon the pgportfolio library and implements algorithms from various academic papers in the field of online portfolio selection.