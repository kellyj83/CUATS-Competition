# region imports
from AlgorithmImports import *
import numpy as np
from statsmodels.regression.linear_model import OLS
from pykalman import KalmanFilter
# endregion

'''
https://medium.com/analytics-vidhya/implementing-a-simple-mean-reverting-pairs-trading-algorithm-in-the-quantconnect-platform-part-1-6f39c99e1233

https://github.com/QuantConnect/Research/blob/master/Analysis/05%20Pairs%20Trading%20Strategy%20Based%20on%20Cointegration.ipynb
'''

class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Define 7-year backtest window
        start_year = 2017

        self.SetStartDate(start_year, 1, 1)
        self.SetEndDate(start_year + 7, 1, 1)

        # Define portfolio cash
        self.SetCash(100000)
        
        # Add pair of assets to trade
        self.symbol1 = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbol2 = self.AddEquity("IVV", Resolution.Daily).Symbol

        # Hyperparameters
        self.lookback = 500      # 200 trading days 
        self.total_spread_history = RollingWindow[float](252)
        self.entry_zscore = 1.5  # Enter trade at 1.5 SD
        self.exit_zscore = 0.2   # Exit trade at 0.2 SD (mean reversion)
        
        self.is_invested = None

        #self.Train(self.DateRules.MonthStart(), self.TimeRules.At(0,0), self.OptimizeThreshold)

        # Objectives (long=buy symbol1 and sell symbol2)
        self.long_targets = [PortfolioTarget(self.symbol1, -0.9), PortfolioTarget(self.symbol2, 0.9)]
        self.short_targets = [PortfolioTarget(self.symbol1, 0.9), PortfolioTarget(self.symbol2, -0.9)]
        
        # Ensure data is available
        self.SetWarmUp(self.lookback)

        # Kalman Filter Setup
        # We track two states: [slope (beta), intercept]
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

        # Initial state estimates
        self.state_mean = np.zeros(2)
        self.state_cov = np.ones((2, 2))
        
        # To calculate Z-score, we track the rolling error (spread)
        self.spread_buffer = RollingWindow[float](10)

    #def OptimizeThreshold(self):
    #        if not self.total_spread_history.IsReady: return
    #        
    #        # Convert buffer to absolute normalized (Z-score) values
    #        spreads = np.array([x for x in self.total_spread_history])
    #        normalized_spread = np.abs((spreads - np.mean(spreads)) / np.std(spreads))
    #        
    #        # Your optimization logic
    #        s0 = np.linspace(0, max(normalized_spread), 50)
    #        f_bar = np.array([len(normalized_spread[normalized_spread > s]) / len(normalized_spread) for s in s0])
    #        
    #        D = np.zeros((49, 50))
    #        for i in range(49):
    #            D[i, i], D[i, i+1] = 1, -1
    #            
    #        l = 1.0
    #        f_star = np.linalg.inv(np.eye(50) + l * D.T @ D) @ f_bar.reshape(-1, 1)
    #        s_star = [f_star[i] * s0[i] for i in range(50)]
    #        
    #        self.entry_zscore = float(s0[np.argmax(s_star)])
    #        self.Debug(f"New Optimized Entry Threshold: {self.entry_zscore}")

    def OnData(self, data: Slice):
        # Ensure data is present
        if not data.Bars.ContainsKey(self.symbol1) or not data.Bars.ContainsKey(self.symbol2) or self.IsWarmingUp:
            return

        # Fetch historical prices
        #history = self.History([self.symbol1, self.symbol2], self.lookback, Resolution.Daily)
        #if history.empty: return
        #prices = np.log(history['close'].unstack(level=0))
        #y = prices[self.symbol1]
        #x = prices[self.symbol2]

        # y = beta * x + c -> y - beta * x stationary and x, y are cointegrated
        #model = OLS(y, x).fit()
        #beta = model.params[0]
#
        ## spread = y - beta * x
        #spread_series = y - (beta * x)
        #current_spread = spread_series.iloc[-1]
        
        # Z-score for current spread
        #spread_mean = np.mean(spread_series)
        #spread_std = np.std(spread_series)
        
        #if spread_std == 0: return
        #z_score = (current_spread - spread_mean) / spread_std

        y = data[self.symbol1].Close
        x = data[self.symbol2].Close

        # 1. Kalman Filter Update Step
        # The observation matrix links our states [beta, intercept] to y: y = beta*x + intercept
        obs_mat = np.array([[x, 1.0]])
        
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean,
            self.state_cov,
            observation=y,
            observation_matrix=obs_mat
        )

        # 2. Calculate the Spread (Error)
        # Spread = Actual Y - Predicted Y
        beta, intercept = self.state_mean
        predicted_y = beta * x + intercept
        spread = y - predicted_y
        
        #self.total_spread_history.Add(spread)

        self.spread_buffer.Add(spread)
        if not self.spread_buffer.IsReady:
            return

        # 3. Calculate Z-Score
        spreads = np.array([x for x in self.spread_buffer])
        std_dev = np.std(spreads)
        z_score = spread / std_dev if std_dev > 0 else 0

        # If it is not invested, see if there is an entry point
        if not self.is_invested:
            # Current spread is too high -> enter short
            if z_score > self.entry_zscore:
                self.Debug(f"Entering Short: Z-Score {z_score}")
                self.SetHoldings(self.short_targets)
                self.is_invested = 'short'
            
            # Current spread is too low -> enter long
            elif z_score < -self.entry_zscore:
                self.Debug(f"Entering Long: Z-Score {z_score}")
                self.SetHoldings(self.long_targets)
                self.is_invested = 'long'

        # If it is invested in something, check the exiting signal 
        elif self.is_invested == 'long':
            if abs(z_score) < self.exit_zscore:
                self.Liquidate()
                self.Debug('Exiting Long')
                self.is_invested = None
                
        elif self.is_invested == 'short':
            if abs(z_score) < self.exit_zscore:
                self.Liquidate()
                self.Debug('Exiting Short')
                self.is_invested = None

    
