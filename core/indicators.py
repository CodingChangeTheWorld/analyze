import pandas as pd
from typing import Optional, Tuple


class Indicators:
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(pd.NA)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def last_swing_low(high: pd.Series, low: pd.Series, lookback: int = 2) -> Optional[Tuple[int, float]]:
        n = len(low)
        if n < lookback * 2 + 1:
            return None
        for i in range(n - lookback - 1, lookback - 1, -1):
            lv = low.iloc[i]
            ok = True
            for k in range(1, lookback + 1):
                if not (lv <= low.iloc[i - k] and lv <= low.iloc[i + k]):
                    ok = False
                    break
            if ok:
                return i, float(lv)
        return None

    @staticmethod
    def close_position(close: pd.Series, high: pd.Series, low: pd.Series, eps: float = 1e-12) -> pd.Series:
        rng = (high - low) + float(eps)
        return (close - low) / rng

    @staticmethod
    def rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
        window = int(window)
        if window <= 1:
            return series
        return series.rolling(window, min_periods=window).quantile(float(q))
