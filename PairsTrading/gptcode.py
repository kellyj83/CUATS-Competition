
# region imports
from AlgorithmImports import *
import numpy as np
from statsmodels.regression.linear_model import OLS
# endregion

class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        # Settings
        self.lookback = 500
        self.entry_zscore = 1.8  # Increased for higher conviction
        self.exit_zscore = 0.2
        self.allocation_per_pair = 0.30 # 30% per pair

        # Initialize Pairs
        self.pair_list = [
            Pair(self, "EWA", "EWC"),
            Pair(self, "XOM", "CVX"),
            Pair(self, "SPY", "IVV")
        ]

        self.SetWarmUp(self.lookback)

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        for pair in self.pair_list:
            pair.Update(data)

class Pair:
    def __init__(self, algo, sym1, sym2):
        self.algo = algo
        self.s1 = algo.AddEquity(sym1, Resolution.Daily).Symbol
        self.s2 = algo.AddEquity(sym2, Resolution.Daily).Symbol
        self.state = None # 'long', 'short', or None

    def Update(self, data):
        # Ensure data exists for both symbols
        if not (data.Bars.ContainsKey(self.s1) and data.Bars.ContainsKey(self.s2)):
            return

        # Fetch history
        history = self.algo.History([self.s1, self.s2], self.algo.lookback, Resolution.Daily)
        if history.empty or 'close' not in history.columns:
            return

        # Prepare Price Series
        prices = history['close'].unstack(level=0)
        if self.s1 not in prices.columns or self.s2 not in prices.columns:
            return
            
        y = prices[self.s1]
        x = prices[self.s2]

        # Calculate Hedge Ratio (Beta) and Spread
        # y = beta * x + alpha
        model = OLS(y, x).fit()
        beta = model.params[0]
        spread_series = y - (beta * x)
        
        current_spread = spread_series.iloc[-1]
        mean = spread_series.mean()
        std = spread_series.std()
        
        if std == 0: return
        z_score = (current_spread - mean) / std

        # Trading Logic
        if self.state is None:
            if z_score > self.algo.entry_zscore:
                # Spread too high: Sell S1, Buy S2
                self.algo.SetHoldings(self.s1, -self.algo.allocation_per_pair / 2)
                self.algo.SetHoldings(self.s2, self.algo.allocation_per_pair / 2)
                self.state = 'short'
                self.algo.Debug(f"Shorting Spread {self.s1}/{self.s2} at {z_score}")

            elif z_score < -self.algo.entry_zscore:
                # Spread too low: Buy S1, Sell S2
                self.algo.SetHoldings(self.s1, self.algo.allocation_per_pair / 2)
                self.algo.SetHoldings(self.s2, -self.algo.allocation_per_pair / 2)
                self.state = 'long'
                self.algo.Debug(f"Longing Spread {self.s1}/{self.s2} at {z_score}")

        elif self.state == 'short':
            if z_score <= self.algo.exit_zscore:
                self.algo.SetHoldings(self.s1, 0)
                self.algo.SetHoldings(self.s2, 0)
                self.state = None
                self.algo.Debug(f"Exiting Short {self.s1}/{self.s2}")

        elif self.state == 'long':
            if z_score >= -self.algo.exit_zscore:
                self.algo.SetHoldings(self.s1, 0)
                self.algo.SetHoldings(self.s2, 0)
                self.state = None
                self.algo.Debug(f"Exiting Long {self.s1}/{self.s2}")