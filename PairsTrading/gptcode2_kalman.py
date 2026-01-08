# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class KalmanPairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)

        # 1. Define Pairs
        # Format: {'y': SymbolY, 'x': SymbolX, 'kf': KalmanFilter, 'status': PositionState}
        self.pair_data = []
        symbols = [
            ("EWA", "EWC"), # Australia vs Canada (Commodity heavy)
            ("AAPL", "MSFT"), # S&P 500 tracking
            ("XOM", "CVX")  # Energy giants
        ]

        for y_ticker, x_ticker in symbols:
            pair = {
                "y": self.AddEquity(y_ticker, Resolution.Daily).Symbol,
                "x": self.AddEquity(x_ticker, Resolution.Daily).Symbol,
                "kf": MyKalmanFilter(delta=1e-4, R=1e-3),
                "invested": None,
                "spread_history": []
            }
            self.pair_data.append(pair)

        # 2. Strategy Hyperparameters
        self.entry_zscore = 1.5
        self.exit_zscore = 0.5
        self.lookback = 30     # Window for calculating Z-Score of the residuals
        self.burn_in = 60      # Days to train Kalman before trading
        self.pair_allocation = 0.20 # Max 20% of capital per pair

        # Use Warmup to ensure we have data immediately
        self.SetWarmUp(self.burn_in)

    def OnData(self, data: Slice):
        for pair in self.pair_data:
            symbol_y = pair["y"]
            symbol_x = pair["x"]

            # Ensure both symbols have data in this slice
            if not (data.Bars.ContainsKey(symbol_y) and data.Bars.ContainsKey(symbol_x)):
                continue

            price_y = data[symbol_y].Close
            price_x = data[symbol_x].Close

            # 3. Update Kalman Filter
            # The filter estimates: price_y = beta * price_x + alpha
            beta, alpha, y_pred = pair["kf"].step_forward(price_y, price_x)
            
            # 4. Calculate Spread (Residual)
            residual = price_y - y_pred
            pair["spread_history"].append(residual)
            
            # Only keep 'lookback' number of residuals
            if len(pair["spread_history"]) > self.lookback:
                pair["spread_history"].pop(0)
            
            # 5. Skip trading if warming up or not enough history
            if self.IsWarmingUp or len(pair["spread_history"]) < self.lookback:
                continue

            # 6. Z-Score Calculation
            mean_res = np.mean(pair["spread_history"])
            std_res = np.std(pair["spread_history"])
            if std_res == 0: continue
            
            z_score = (residual - mean_res) / std_res

            # 7. Execute Trade Logic
            self.ExecuteTrade(pair, z_score, beta)

    def ExecuteTrade(self, pair, z_score, beta):
        y = pair["y"]
        x = pair["x"]

        # Market Neutral Weighting: 
        # Total weight = Weight_Y + Weight_X
        # Weight_X = Weight_Y * Beta
        weight_y = self.pair_allocation / (1 + abs(beta))
        weight_x = weight_y * beta

        if pair["invested"] is None:
            if z_score > self.entry_zscore:
                # Spread high: Y is expensive, X is cheap. 
                # Short Y, Long X
                self.SetHoldings([PortfolioTarget(y, -weight_y), PortfolioTarget(x, weight_x)])
                pair["invested"] = 'short_spread'
                self.Debug(f"Entry Short Spread {y}/{x} | Z: {z_score:.2f} | Beta: {beta:.2f}")

            elif z_score < -self.entry_zscore:
                # Spread low: Y is cheap, X is expensive
                # Long Y, Short X
                self.SetHoldings([PortfolioTarget(y, weight_y), PortfolioTarget(x, -weight_x)])
                pair["invested"] = 'long_spread'
                self.Debug(f"Entry Long Spread {y}/{x} | Z: {z_score:.2f} | Beta: {beta:.2f}")

        elif pair["invested"] == 'long_spread' and z_score > -self.exit_zscore:
            self.Liquidate(y)
            self.Liquidate(x)
            pair["invested"] = None
            self.Debug(f"Exit Long Spread {y}/{x}")

        elif pair["invested"] == 'short_spread' and z_score < self.exit_zscore:
            self.Liquidate(y)
            self.Liquidate(x)
            pair["invested"] = None
            self.Debug(f"Exit Short Spread {y}/{x}")

class MyKalmanFilter:
    """
    An online Kalman Filter for recursive least squares regression.
    """
    def __init__(self, delta=1e-4, R=1e-3):
        # R: Measurement noise (Trust in new data)
        # Q: Process noise (How fast the beta/alpha can change)
        self.R = R
        self.Q = delta / (1 - delta) * np.eye(2)
        
        # State: [Slope (Beta), Intercept (Alpha)]
        self.x = np.zeros((2, 1)) 
        self.P = np.eye(2) # Initial uncertainty    

    def step_forward(self, y_val, x_val):
        # Observation matrix: [price_x, 1]
        H = np.array([x_val, 1])[None]
        
        # 1. Prediction (Time Update)
        # We assume x_k = x_{k-1} (Random walk model)
        x_hat = self.x 
        P_hat = self.P + self.Q

        # 2. Measurement Update (Correction)
        y_pred = H.dot(x_hat)
        residual = y_val - y_pred
        
        # Kalman Gain
        S = H.dot(P_hat).dot(H.T) + self.R
        K = P_hat.dot(H.T) / S

        # Update State and Covariance
        self.x = x_hat + K.dot(residual)
        self.P = (np.eye(2) - K.dot(H)).dot(P_hat)

        # Return Beta, Alpha, and the predicted Y value
        return self.x[0, 0], self.x[1, 0], y_pred[0, 0]