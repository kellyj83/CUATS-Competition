from AlgorithmImports import *
import numpy as np
from pykalman import KalmanFilter

class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        start_year = 2004
        self.SetStartDate(start_year, 1, 1)
        self.SetEndDate(start_year + 22, 1, 1)
        self.SetCash(100000)

        # 1. Define hyperparameters
        self.entry_zscore = 1.09
        self.exit_zscore = 0.38
        
        # 2. Define the pairs you want to trade
        # Format: (Symbol1, Symbol2, Weight1, Weight2)
        pair_configs = [
            ("WMT", "V", 0.35, 0.65), # Good Pair
            ("EWA", "EWC", 0.5, 0.5), # Good Pair
            ("VZ", "KO", 0.45, 0.55) # Decent Pair
            #("RDS.A", "RDS.B", 0.25, 0.2), # OK
            #("GE", "IBM", 0.2, 0.2), # bad during 2008
            #("MRK", "VZ", 0.2, 0.2), # Not Great
            #("CAN", "VZ", 0.2, 0.2),
            #("CL", "BRN", 0.14, 0.2) # Very Bad at 2021 for some reason, but good overall
            #("EWA", "EWC", 0.25, 0.65), Good Pair
            #("RDS.A", "RDS.B", 0.34, 0.65), # OK
            #("CAN", "VZ", 0.34, 0.65),
            # ("CL", "BRN", 0.34, 0.65) # Very Bad at 2021 for some reason, but good overall
        ]

        # 3. Initialize Pair Managers
        self.pair_managers = []
        for s1, s2, w1, w2 in pair_configs:
            sym1 = self.AddEquity(s1, Resolution.Daily).Symbol
            sym2 = self.AddEquity(s2, Resolution.Daily).Symbol
            
            manager = PairManager(sym1, sym2, w1, w2, self.entry_zscore, self.exit_zscore)
            self.pair_managers.append(manager)

        self.SetWarmUp(500)

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        for pair in self.pair_managers:
            # Check if data exists for both symbols in the pair
            if data.Bars.ContainsKey(pair.s1) and data.Bars.ContainsKey(pair.s2):
                pair.Update(self, data)


class PairManager:
    """Helper class to encapsulate Kalman Filter and Trading Logic for a single pair."""
    def __init__(self, s1, s2, w1, w2, entry_z, exit_z):
        self.s1 = s1
        self.s2 = s2
        self.w1 = w1
        self.w2 = w2
        self.entry_zscore = entry_z
        self.exit_zscore = exit_z
        
        self.is_invested = None # None, 'long', or 'short'
        self.spread_buffer = RollingWindow[float](20)

        # Kalman Filter Setup
        trans_cov = 1e-5 / (1 - 1e-5) * np.eye(2)
        self.kf = KalmanFilter(
            n_dim_obs=1, n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_covariance=1.0,
            transition_covariance=trans_cov
        )
        self.state_mean = np.zeros(2)
        self.state_cov = np.ones((2, 2))

    def Update(self, algo, data):
        y = data[self.s1].Close
        x = data[self.s2].Close

        # Kalman Filter Update
        obs_mat = np.array([[x, 1.0]])
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, observation=y, observation_matrix=obs_mat
        )

        # Calculate Spread and Z-Score
        beta, intercept = self.state_mean
        spread = y - (beta * x + intercept)
        self.spread_buffer.Add(spread)

        if not self.spread_buffer.IsReady:
            return

        spreads = np.array([s for s in self.spread_buffer])
        std_dev = np.std(spreads)
        z_score = spread / std_dev if std_dev > 0 else 0

        # Trading Logic
        if not self.is_invested:
            if z_score > self.entry_zscore:
                algo.SetHoldings([PortfolioTarget(self.s1, -self.w1), PortfolioTarget(self.s2, self.w2)])
                self.is_invested = 'short'
            elif z_score < -self.entry_zscore:
                algo.SetHoldings([PortfolioTarget(self.s1, self.w1), PortfolioTarget(self.s2, -self.w2)])
                self.is_invested = 'long'

        elif self.is_invested == 'long':
            if abs(z_score) < self.exit_zscore:
                algo.Liquidate(self.s1)
                algo.Liquidate(self.s2)
                self.is_invested = None
                
        elif self.is_invested == 'short':
            if abs(z_score) < self.exit_zscore:
                algo.Liquidate(self.s1)
                algo.Liquidate(self.s2)
                self.is_invested = None
