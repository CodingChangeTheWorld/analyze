from typing import Dict, Optional, Tuple

import pandas as pd

from config import Config
from core.datasource import BinanceDataSource
from core.indicators import Indicators
from core.strategy.base import BaseStrategy
from core.strategy.config import StrategyConfig


class PullbackShortStrategy(BaseStrategy):
    def __init__(self):
        self.cfg = StrategyConfig.PULLBACK_SHORT

    def analyze(self, symbol: str, ds: BinanceDataSource, **kwargs: object) -> Optional[Dict[str, object]]:
        h4 = ds.get_klines_df(symbol, "4h", 200)
        h1 = ds.get_klines_df(symbol, "1h", 200)
        m15 = ds.get_klines_df(symbol, "15m", 200)
        return self.analyze_frames(symbol, h4=h4, h1=h1, m15=m15)

    def analyze_frames(self, symbol: str, h4: pd.DataFrame, h1: pd.DataFrame, m15: pd.DataFrame) -> Optional[Dict[str, object]]:
        if len(h4) < 60 or len(h1) < 60 or len(m15) < 60:
            return None

        if bool(getattr(Config, "TREND_FILTER_ENABLED", False)):
            if not self._check_trend(h4["close"]) or not self._check_trend(h1["close"]):
                return None

        breakdown = self._find_breakdown(h1)
        if not breakdown:
            return None
        level, _ = breakdown

        m15_atr = Indicators.atr(m15["high"], m15["low"], m15["close"], 14)
        rsi_vals = Indicators.rsi(m15["close"], 14)
        rej_idx = self._check_rejection(m15, level, m15_atr)

        if rej_idx is None:
            return None

        rsi_val = rsi_vals.iloc[rej_idx]
        rsi_ok = (rsi_val < 50) if pd.notna(rsi_val) else False
        if not rsi_ok:
            return None

        return self._calculate_risk(symbol, level, m15_atr, rej_idx, m15["close_time"].iloc[rej_idx], rsi_ok, h4, h1)

    def _check_trend(self, close: pd.Series) -> bool:
        e20 = Indicators.ema(close, 20)
        e50 = Indicators.ema(close, 50)
        return bool(e20.iloc[-1] < e50.iloc[-1])

    def _find_breakdown(self, df: pd.DataFrame) -> Optional[Tuple[float, int]]:
        sw = Indicators.last_swing_low(df["high"], df["low"], 2)
        if not sw:
            return None
        idx, lvl = sw
        if df["close"].iloc[-1] < lvl:
            return lvl, idx
        return None

    def _check_rejection(self, df: pd.DataFrame, level: float, atr_vals: pd.Series) -> Optional[int]:
        n = len(df)
        if n < 20:
            return None
        price_tolerance_pct = 0.02
        wick_ratio_min = 0.4

        for i in range(n - 5, n):
            if self.is_level_rejection(df.iloc[i], level, side="short", price_tolerance_pct=price_tolerance_pct, wick_ratio_min=wick_ratio_min) or (
                i + 1 < n and self.is_level_rejection(df.iloc[i + 1], level, side="short", price_tolerance_pct=price_tolerance_pct, wick_ratio_min=wick_ratio_min)
            ):
                return i
        return None

    def _calculate_risk(
        self, symbol: str, level: float, atr_vals: pd.Series, idx: int, time_ms: int, rsi_ok: bool, h4: pd.DataFrame, h1: pd.DataFrame
    ) -> Dict[str, object]:
        last_atr = atr_vals.iloc[-1]
        atr_small = (last_atr * 0.1) if pd.notna(last_atr) else 0
        atr_large = (last_atr * 0.5) if pd.notna(last_atr) else 0

        entry = level - max(atr_small, level * 0.0005)
        stop = level + max(atr_large, level * 0.0015)
        risk = stop - entry
        target = entry - risk * 2
        rr = (entry - target) / risk if risk > 0 else 0

        score = 1
        score += 1 if self._check_trend(h4["close"]) else 0
        score += 1 if self._check_trend(h1["close"]) else 0
        score += 1 if rsi_ok else 0
        score += 1 if rr >= 2 else 0

        return self.build_signal(
            symbol=symbol,
            time_ms=int(time_ms),
            rr=round(rr, 2),
            score=score,
            level=level,
            entry=entry,
            stop=stop,
            target=target,
        )
