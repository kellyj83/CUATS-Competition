# region imports
from AlgorithmImports import *
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
# endregion

'''
https://medium.com/analytics-vidhya/implementing-a-simple-mean-reverting-pairs-trading-algorithm-in-the-quantconnect-platform-part-1-6f39c99e1233

https://github.com/QuantConnect/Research/blob/master/Analysis/05%20Pairs%20Trading%20Strategy%20Based%20on%20Cointegration.ipynb
'''

class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Define 7-year backtest window
        start_year = 2004

        self.SetStartDate(start_year, 1, 1)
        self.SetEndDate(start_year + 21, 1, 1)

        # Define portfolio cash
        self.SetCash(100000)
        
        # Add pair of assets to trade
        self.symbol1 = self.AddEquity("EWA", Resolution.Daily).Symbol
        self.symbol2 = self.AddEquity("EWC", Resolution.Daily).Symbol

        self.symbol3 = self.AddEquity("PEP", Resolution.Daily).Symbol
        self.symbol4 = self.AddEquity("KO", Resolution.Daily).Symbol

        self.symbol5 = self.AddEquity("XOM", Resolution.Daily).Symbol
        self.symbol6 = self.AddEquity("CVX", Resolution.Daily).Symbol

        # Hyperparameters
        self.lookback = 90     # 36 trading days 
        self.entry_zscore = 1.5  # Enter trade at 1.5 SD
        self.exit_zscore = 0.2   # Exit trade at 0.2 SD (mean reversion)
        
        self.is_invested = None
        self.is_invested2 = None
        self.is_invested3 = None
        

        # Objectives (long=buy symbol1 and sell symbol2)
        self.long_targets = [PortfolioTarget(self.symbol1, 0.15), PortfolioTarget(self.symbol2, -0.15)]
        self.short_targets = [PortfolioTarget(self.symbol1, -0.15), PortfolioTarget(self.symbol2, 0.15)]
        
        self.long_targets2 = [PortfolioTarget(self.symbol3, 0.15), PortfolioTarget(self.symbol4, -0.15)]
        self.short_targets2 = [PortfolioTarget(self.symbol3, -0.15), PortfolioTarget(self.symbol4, 0.15)]
        
        self.long_targets3 = [PortfolioTarget(self.symbol5, 0.15), PortfolioTarget(self.symbol6, -0.15)]
        self.short_targets3 = [PortfolioTarget(self.symbol5, -0.15), PortfolioTarget(self.symbol6, 0.15)]
        
        # Ensure data is available
        self.SetWarmUp(self.lookback)

    def OnData(self, data: Slice):
        # Ensure data is present
        if not data.Bars.ContainsKey(self.symbol1) or not data.Bars.ContainsKey(self.symbol2) or self.IsWarmingUp:
            return

        if not data.Bars.ContainsKey(self.symbol3) or not data.Bars.ContainsKey(self.symbol4) or self.IsWarmingUp:
            return

        if not data.Bars.ContainsKey(self.symbol5) or not data.Bars.ContainsKey(self.symbol6) or self.IsWarmingUp:
            return

        # Fetch historical prices
        history = self.History([self.symbol1, self.symbol2], self.lookback, Resolution.Daily)
        history2 = self.History([self.symbol3, self.symbol4], self.lookback, Resolution.Daily)
        history3 = self.History([self.symbol5, self.symbol6], self.lookback, Resolution.Daily)
        
        if history.empty: return
        if history2.empty: return
        if history3.empty: return
        
        prices = history['close'].unstack(level=0)
        prices2 = history2['close'].unstack(level=0)
        prices3 = history3['close'].unstack(level=0)

        y = prices[self.symbol1]
        x = prices[self.symbol2]

        y2 = prices2[self.symbol3]
        x2 = prices2[self.symbol4]        

        y3 = prices3[self.symbol5]
        x3 = prices3[self.symbol6]

        # spread = y - (beta * x + alpha)
        # y = beta * x + c -> y - beta * x stationary and x, y are cointegrated
        
        model = OLS(y, x).fit()

        model2 = OLS(y2, x2).fit()

        model3 = OLS(y3, x3).fit()

        beta = model.params[0]

        beta2 = model2.params[0]

        beta3 = model3.params[0]

        # spread = y - beta * x
        spread_series = y - (beta * x)
        current_spread = spread_series.iloc[-1]

        spread_series2 = y2 - (beta2 * x2)
        current_spread2 = spread_series2.iloc[-1]

        spread_series3 = y3 - (beta3 * x3)
        current_spread3 = spread_series3.iloc[-1]
        
        # Z-score for current spread
        spread_mean = np.mean(spread_series)
        spread_std = np.std(spread_series)

        spread_mean2 = np.mean(spread_series2)
        spread_std2 = np.std(spread_series2)

        spread_mean3 = np.mean(spread_series3)
        spread_std3 = np.std(spread_series3)
        
        if spread_std == 0: return
        z_score = (current_spread - spread_mean) / spread_std

        if spread_std2 == 0: return
        z_score2 = (current_spread2 - spread_mean2) / spread_std2

        if spread_std3 == 0: return
        z_score3 = (current_spread3 - spread_mean3) / spread_std3


        # If it is not invested, see if there is an entry point
        if not self.is_invested:
            score, pvalue, _ = coint(y, x)
            if pvalue <= 0.05: 
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
                self.SetHoldings(self.symbol1, 0)
                self.SetHoldings(self.symbol2, 0)
                self.Debug('Exiting Long')
                self.is_invested = None
        
        
        elif self.is_invested == 'short':
            if abs(z_score) < self.exit_zscore:
                self.SetHoldings(self.symbol1, 0)
                self.SetHoldings(self.symbol2, 0)
                self.Debug('Exiting Short')
                self.is_invested = None


        # If it is not invested, see if there is an entry point
        if not self.is_invested2:
            score, pvalue2, _ = coint(y2, x2)
            if pvalue2 <= 0.05: 
                # Current spread is too high -> enter short
                if z_score2 > self.entry_zscore:
                    self.Debug(f"Entering Short: Z-Score {z_score2}")
                    self.SetHoldings(self.short_targets2)
                    self.is_invested2 = 'short'
                
                # Current spread is too low -> enter long
                elif z_score2 < -self.entry_zscore:
                    self.Debug(f"Entering Long: Z-Score {z_score2}")
                    self.SetHoldings(self.long_targets2)
                    self.is_invested2 = 'long'

        # If it is invested in something, check the exiting signal 
        elif self.is_invested2 == 'long':
            if abs(z_score2) < self.exit_zscore:
                self.SetHoldings(self.symbol3, 0)
                self.SetHoldings(self.symbol4, 0)
                self.Debug('Exiting Long')
                self.is_invested2 = None
        
        
        elif self.is_invested2 == 'short':
            if abs(z_score2) < self.exit_zscore:
                self.SetHoldings(self.symbol3, 0)
                self.SetHoldings(self.symbol4, 0)
                self.Debug('Exiting Short')
                self.is_invested2 = None


        # If it is not invested, see if there is an entry point
        if not self.is_invested3:
            score, pvalue3, _ = coint(y3, x3)
            if pvalue3 <= 0.05: 
                # Current spread is too high -> enter short
                if z_score3 > self.entry_zscore:
                    self.Debug(f"Entering Short: Z-Score {z_score3}")
                    self.SetHoldings(self.short_targets3)
                    self.is_invested3 = 'short'
                
                # Current spread is too low -> enter long
                elif z_score3 < -self.entry_zscore:
                    self.Debug(f"Entering Long: Z-Score {z_score3}")
                    self.SetHoldings(self.long_targets3)
                    self.is_invested3 = 'long'

        # If it is invested in something, check the exiting signal 
        elif self.is_invested3 == 'long':
            if abs(z_score3) < self.exit_zscore:
                self.SetHoldings(self.symbol5, 0)
                self.SetHoldings(self.symbol6, 0)
                self.Debug('Exiting Long')
                self.is_invested3 = None
        
        
        elif self.is_invested3 == 'short':
            if abs(z_score3) < self.exit_zscore:
                self.SetHoldings(self.symbol5, 0)
                self.SetHoldings(self.symbol6, 0)
                self.Debug('Exiting Short')
                self.is_invested3 = None


    
