# region imports
from AlgorithmImports import *
import numpy as np
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
# endregion

class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Define 7-year backtest window
        start_year = 2013
        self.SetStartDate(start_year, 1, 1)
        self.SetEndDate(start_year + 7, 1, 1)
        self.SetCash(100000)

        # 1. Define the Trading Pairs from the image
        # Format: (Stock 1, Stock 2)
        trading_pair_tickers = [
            ("AAP", "IBM"),
            ("EWA", "EWC"),
            ("NKE", "AES"),
            ("MRK", "VZ"),
            ("GS", "CVX"),
            ("MRK", "DD"),   # Note: DD (DuPont) has complex merger history
            ("AES", "XOM"),
            ("CVX", "AMG"),
            ("VZ", "KO"),
            ("VZ", "JNJ"),
            ("VZ", "PG"),
            ("KO", "WMT"),
            ("WMT", "V"),
            ("GE", "IBM"),
            ("WMT", "XOM"),
            ("TRV", "XOM"),
            ("PG", "DD"),
            ("NKE", "VZ"),
            ("CSCO", "XOM"),
            ("ACN", "VZ")    # "CAN" in table likely typo for ACN (Accenture)
        ]

        self.pairs = []
        self.lookback = 252 # History window for Cointegration Test
        
        # Calculate position size based on number of pairs to avoid margin calls
        # We leave some buffer (0.9 instead of 1.0)
        self.pos_size_per_pair = 0.9 / len(trading_pair_tickers)

        # 2. Loop to Initialize each pair
        for ticker1, ticker2 in trading_pair_tickers:
            # Add Equities
            s1 = self.AddEquity(ticker1, Resolution.Daily).Symbol
            s2 = self.AddEquity(ticker2, Resolution.Daily).Symbol
            
            # Create a 'Manager' for this specific pair
            pair_manager = CointegratedPair(self, s1, s2, self.pos_size_per_pair)
            self.pairs.append(pair_manager)

        self.SetWarmUp(self.lookback)

    def OnData(self, data: Slice):
        if self.IsWarmingUp: return

        # Loop through every pair and update their specific logic
        for pair in self.pairs:
            pair.OnData(data)


class CointegratedPair:
    '''
    Helper class to manage the State, Kalman Filter, and Logic 
    for a single pair of assets.
    '''
    def __init__(self, algorithm, symbol1, symbol2, max_allocation):
        self.algo = algorithm
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.allocation = max_allocation # Max % of portfolio for this pair
        
        # Hyperparameters
        self.entry_zscore = 1.09
        self.exit_zscore = 0.38
        self.coint_threshold = 0.05 # P-value threshold for statistical significance
        
        self.is_invested = None
        
        # Kalman Filter Setup (Same as your original code)
        trans_cov = 1e-5 / (1 - 1e-5) * np.eye(2)
        self.kf = KalmanFilter(
            n_dim_obs=1, 
            n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_covariance=1.0,
            transition_covariance=trans_cov
        )
        self.state_mean = np.zeros(2)
        self.state_cov = np.ones((2, 2))
        
        # Rolling buffer for Z-score calculation
        self.spread_buffer = RollingWindow[float](8)

    def OnData(self, data):
        # 1. Data Check
        if not data.Bars.ContainsKey(self.symbol1) or not data.Bars.ContainsKey(self.symbol2):
            return

        # 2. Cointegration Test
        # We fetch history to check if the pair is ACTUALLY cointegrated right now.
        #history = self.algo.History([self.symbol1, self.symbol2], 252, Resolution.Daily)
        #if history.empty or len(history) < 252: return
        
        # Extract closing prices
        #try:
        #    h1 = history.loc[self.symbol1].close
        #    h2 = history.loc[self.symbol2].close
        #except KeyError:
        #    return

        # Align series (drop missing dates)
        #df = pd.concat([h1, h2], axis=1).dropna()
        #if len(df) < 100: return
        
        #s1_prices = df.iloc[:, 0]
        #s2_prices = df.iloc[:, 1]

        # Run Engle-Granger Cointegration Test
        # Returns: (t-stat, p-value, crit_values)
        #score, pvalue, _ = coint(s1_prices, s2_prices)
        
        # If p-value is too high, the relationship is broken. Do not trade.
        # If we are invested, we might want to exit, or just hold. 
        # Here we block NEW entries if not cointegrated.
        #is_cointegrated = pvalue < self.coint_threshold
        is_cointegrated = True

        # 3. Kalman Filter Update
        y = data[self.symbol1].Close
        x = data[self.symbol2].Close
        
        obs_mat = np.array([[x, 1.0]])
        
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean,
            self.state_cov,
            observation=y,
            observation_matrix=obs_mat
        )

        # 4. Calculate Spread & Z-Score
        beta, intercept = self.state_mean
        predicted_y = beta * x + intercept
        spread = y - predicted_y
        
        self.spread_buffer.Add(spread)
        if not self.spread_buffer.IsReady: return
        
        spreads = np.array([v for v in self.spread_buffer])
        std_dev = np.std(spreads)
        
        if std_dev == 0: return
        z_score = spread / std_dev

        # 5. Trading Logic
        
        # Define Targets based on allocation
        # Long Spread: Buy Symbol 1, Sell Symbol 2
        # We split the allocation: 50% of the pair's budget to S1, 50% to S2
        half_alloc = self.allocation / 2
        
        long_targets = [
            PortfolioTarget(self.symbol1, half_alloc), 
            PortfolioTarget(self.symbol2, -half_alloc)
        ]
        short_targets = [
            PortfolioTarget(self.symbol1, -half_alloc), 
            PortfolioTarget(self.symbol2, half_alloc)
        ]

        # Entry Logic (Only if cointegrated)
        if not self.is_invested and is_cointegrated:
            if z_score > self.entry_zscore:
                #self.algo.Debug(f"[{self.symbol1}/{self.symbol2}] Entry Short (Z:{z_score:.2f} P:{pvalue:.3f})")
                self.algo.SetHoldings(short_targets)
                self.is_invested = 'short'
            
            elif z_score < -self.entry_zscore:
                #self.algo.Debug(f"[{self.symbol1}/{self.symbol2}] Entry Long (Z:{z_score:.2f} P:{pvalue:.3f})")
                self.algo.SetHoldings(long_targets)
                self.is_invested = 'long'

        # Exit Logic (Mean Reversion)
        elif self.is_invested:
            # We exit if Z-score reverts OR if cointegration breaks significantly (optional safety)
            # Here we stick to Mean Reversion for exit
            if self.is_invested == 'long' and z_score > -self.exit_zscore:
                self.algo.Liquidate(self.symbol1)
                self.algo.Liquidate(self.symbol2)
                self.is_invested = None
                
            elif self.is_invested == 'short' and z_score < self.exit_zscore:
                self.algo.Liquidate(self.symbol1)
                self.algo.Liquidate(self.symbol2)
                self.is_invested = None