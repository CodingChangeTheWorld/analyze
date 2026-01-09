from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from core.datasource import BinanceDataSource


class BaseStrategy(ABC):
    @abstractmethod
    def analyze(self, symbol: str, ds: BinanceDataSource, **kwargs: object) -> Optional[Dict[str, object]]:
        pass

    def build_signal(self, symbol: str, time_ms: int, rr: float, score: int, **fields: object) -> Dict[str, object]:
        out: Dict[str, object] = {
            "symbol": symbol,
            "strategy": self.__class__.__name__,
            "time": int(time_ms),
            "rr": float(rr),
            "score": int(score),
        }
        out.update(fields)
        return out

    @staticmethod
    def is_level_rejection(
        candle: pd.Series,
        level: float,
        side: str,
        price_tolerance_pct: float = 0.001,
        wick_ratio_min: float = 0.5,
        close_pos_max_for_short: float = 0.4,
        close_pos_min_for_long: float = 0.6,
        require_reclaim: bool = True,
        eps: float = 1e-12,
    ) -> bool:
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])

        level = float(level)
        price_tolerance_pct = float(price_tolerance_pct)
        wick_ratio_min = float(wick_ratio_min)
        close_pos_max_for_short = float(close_pos_max_for_short)
        close_pos_min_for_long = float(close_pos_min_for_long)

        if level <= 0 or price_tolerance_pct <= 0:
            return False
        if wick_ratio_min <= 0 or wick_ratio_min > 1:
            return False

        rng = h - l
        if rng <= 0:
            return False

        tol = level * price_tolerance_pct

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_ratio_u = upper_wick / (rng + eps)
        wick_ratio_l = lower_wick / (rng + eps)

        close_pos = (c - l) / (rng + eps)

        if side == "short":
            touched = h >= level - tol
            shape_ok = (wick_ratio_u >= wick_ratio_min) and (close_pos <= close_pos_max_for_short)
            if not (touched and shape_ok):
                return False
            if require_reclaim:
                return c <= level
            return True

        if side == "long":
            touched = l <= level + tol
            shape_ok = (wick_ratio_l >= wick_ratio_min) and (close_pos >= close_pos_min_for_long)
            if not (touched and shape_ok):
                return False
            if require_reclaim:
                return c >= level
            return True

        raise ValueError(f"unsupported side: {side}")
