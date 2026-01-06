# region imports
from AlgorithmImports import *
import numpy as np
from statsmodels.regression.linear_model import OLS
# endregion

'''
https://medium.com/analytics-vidhya/implementing-a-simple-mean-reverting-pairs-trading-algorithm-in-the-quantconnect-platform-part-1-6f39c99e1233

https://github.com/QuantConnect/Research/blob/master/Analysis/05%20Pairs%20Trading%20Strategy%20Based%20on%20Cointegration.ipynb
'''

class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Define 7-year backtest window
        start_year = 2006

        self.SetStartDate(start_year, 1, 1)
        self.SetEndDate(start_year + 7, 1, 1)

        # Define portfolio cash
        self.SetCash(100000)
        
        # Add pair of assets to trade
        self.symbol1 = self.AddEquity("EWA", Resolution.Daily).Symbol
        self.symbol2 = self.AddEquity("EWC", Resolution.Daily).Symbol

        # Hyperparameters
        self.lookback = 500      # 200 trading days 
        self.entry_zscore = 1.5  # Enter trade at 1.5 SD
        self.exit_zscore = 0.2   # Exit trade at 0.2 SD (mean reversion)
        
        self.is_invested = None

        # Objectives (long=buy symbol1 and sell symbol2)
        self.long_targets = [PortfolioTarget(self.symbol1, 0.9), PortfolioTarget(self.symbol2, -0.9)]
        self.short_targets = [PortfolioTarget(self.symbol1, -0.9), PortfolioTarget(self.symbol2, 0.9)]
        
        # Ensure data is available
        self.SetWarmUp(self.lookback)

    def OnData(self, data: Slice):
        # Ensure data is present
        if not data.Bars.ContainsKey(self.symbol1) or not data.Bars.ContainsKey(self.symbol2) or self.IsWarmingUp:
            return

        # Fetch historical prices
        history = self.History([self.symbol1, self.symbol2], self.lookback, Resolution.Daily)
        if history.empty: return
        prices = history['close'].unstack(level=0)
        y = prices[self.symbol1]
        x = prices[self.symbol2]

        # y = beta * x + c -> y - beta * x stationary and x, y are cointegrated
        model = OLS(y, x).fit()
        beta = model.params[0]

        # spread = y - beta * x
        spread_series = y - (beta * x)
        current_spread = spread_series.iloc[-1]
        
        # Z-score for current spread
        spread_mean = np.mean(spread_series)
        spread_std = np.std(spread_series)
        
        if spread_std == 0: return
        z_score = (current_spread - spread_mean) / spread_std

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

    
