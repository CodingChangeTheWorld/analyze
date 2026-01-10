from typing import Dict, Optional

import pandas as pd

from core.datasource import BinanceDataSource
from core.indicators import Indicators
from core.strategy.base import BaseStrategy
from core.strategy.config import StrategyConfig


class SqueezeBreakoutLongStrategy(BaseStrategy):
    def __init__(self):
        self._last_signal_time: Dict[str, int] = {}
        self.cfg = StrategyConfig.SQUEEZE_BREAKOUT_LONG

    def analyze(self, symbol: str, ds: BinanceDataSource, **kwargs: object) -> Optional[Dict[str, object]]:
        m15 = ds.get_klines_df(symbol, "15m", self.cfg["kline_15m_limit"])
        if m15 is None or m15.empty or len(m15) < self.cfg["min_15m_candles"]:
            return None

        squeeze = self._squeeze_box_15m(m15)
        if not squeeze:
            return None

        box_high = float(squeeze["box_high"])
        box_low = float(squeeze["box_low"])
        box_width = float(squeeze["box_width"])
        atr15 = float(squeeze["atr15"])

        m5 = ds.get_klines_df(symbol, "5m", self.cfg["kline_5m_limit"])
        if m5 is None or m5.empty or len(m5) < self.cfg["min_5m_candles"]:
            return None

        now_ms = kwargs.get("now_ms")
        if now_ms is None:
            now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        else:
            now_ms = int(now_ms)

        idx = self._last_closed_idx(m5, now_ms)
        if idx is None:
            return None

        abs_idx = idx if idx >= 0 else len(m5) + idx
        if abs_idx < 0 or abs_idx >= len(m5):
            return None

        close_time_ms = int(m5["close_time"].iloc[abs_idx])
        last_sig = self._last_signal_time.get(symbol)
        if last_sig == close_time_ms:
            return None

        atr5_series = Indicators.atr(m5["high"], m5["low"], m5["close"], self.cfg["atr_period"])
        atr5 = float(atr5_series.iloc[abs_idx]) if pd.notna(atr5_series.iloc[abs_idx]) else 0.0
        if atr5 <= 0:
            return None

        level = box_high

        c = float(m5["close"].iloc[abs_idx])
        h = float(m5["high"].iloc[abs_idx])
        l = float(m5["low"].iloc[abs_idx])
        v = float(m5["volume"].iloc[abs_idx])

        candle_range = max(h - l, 0.0)
        close_pos = (c - l) / candle_range if candle_range > 0 else 0.0

        vol_ma = m5["volume"].rolling(self.cfg["vol_ma_period"], min_periods=self.cfg["vol_ma_period"]).mean()

        vol_base = (
            float(vol_ma.iloc[abs_idx])
            if abs_idx >= self.cfg["vol_ma_period"] - 1 and pd.notna(vol_ma.iloc[abs_idx])
            else 0.0
        )
        vol_ratio = (v / vol_base) if vol_base > 0 else 0.0

        cond_close_break = c > level + self.cfg["k1"] * atr5
        cond_range_expand = (candle_range / atr5) > self.cfg["k2"] if atr5 > 0 else False
        cond_vol_shock = vol_ratio > self.cfg["k3"]
        cond_close_pos = close_pos > self.cfg["close_pos_min"]
        conds = [cond_close_break, cond_range_expand, cond_vol_shock, cond_close_pos]
        cond_count = sum(1 for x in conds if x)

        # 保持原策略结构/含义不变：站稳逻辑仍然强制 True
        stand_ok = True

        # ✅ FIX: 突破策略的核心条件是价格必须突破箱体顶部
        # 即使满足其他条件，如果没有价格突破，也不应该产生信号
        if cond_close_break and cond_count >= self.cfg["min_conditions"] and stand_ok:
            entry = c
            stop = level - self.cfg["ksl"] * atr5
            risk = max(entry - stop, 0.0)
            tp1 = entry + self.cfg["tp1_r"] * risk if risk > 0 else 0.0
            tp2 = entry + self.cfg["tp2_r"] * risk if risk > 0 else 0.0

            m15_idx = self._last_closed_idx(m15, now_ms)
            if m15_idx is None:
                # fallback：尽量用倒数第二根（假设最后一根可能在形成）
                m15_abs_idx = len(m15) - 2
            else:
                m15_abs_idx = m15_idx if m15_idx >= 0 else len(m15) + m15_idx

            if m15_abs_idx < 0 or m15_abs_idx >= len(m15):
                return None

            end15 = m15_abs_idx + 1
            start15 = max(0, end15 - self.cfg["box_lookback"])
            highest_close_15m = float(m15["close"].iloc[start15:end15].max())
            trail_stop_15m = highest_close_15m - self.cfg["trail_m"] * atr15 if atr15 > 0 else 0.0

            score = int(cond_count) + (1 if vol_ratio >= self.cfg["k3"] * self.cfg["vol_shock_factor"] else 0)
            rr = (tp2 - entry) / risk if risk > 0 else 0.0

            self._last_signal_time[symbol] = close_time_ms
            return self.build_signal(
                symbol=symbol,
                time_ms=close_time_ms,
                rr=round(float(rr), 2),
                score=int(score),
                mode="breakout_chase",
                level=float(level),
                box_high=float(box_high),
                box_low=float(box_low),
                box_width=float(box_width),
                atr15=float(atr15),
                atr5=float(atr5),
                entry=float(entry),
                stop=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                trail_stop_15m=float(trail_stop_15m),
                vol_ratio=float(vol_ratio),
                close_pos=float(close_pos),
                cond_close_break=bool(cond_close_break),
                cond_range_expand=bool(cond_range_expand),
                cond_vol_shock=bool(cond_vol_shock),
                cond_close_pos=bool(cond_close_pos),
                stand_ok=bool(stand_ok),
            )



    def _last_closed_idx(self, df: pd.DataFrame, now_ms: int) -> Optional[int]:
        if df is None or df.empty or "close_time" not in df.columns:
            return None
        if len(df) < 2:
            return None
        last_ct = int(df["close_time"].iloc[-1])
        if last_ct <= now_ms - 2_000:
            return -1
        return -2

    def _squeeze_box_15m(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        if df is None or df.empty or len(df) < self.cfg["min_15m_candles"]:
            return None

        # Use -2 (last fully closed candle) instead of -1 (forming or just closed but potentially exploding)
        # to detect the squeeze state.
        # This prevents the breakout candle itself from ruining the squeeze metrics.
        check_idx = -2
        if len(df) + check_idx < 0:
            return None

        atr = Indicators.atr(df["high"], df["low"], df["close"], self.cfg["atr_period"])
        window = self.cfg["squeeze_box_window"]
        atr_percent = Indicators.rolling_quantile(atr, window, self.cfg["squeeze_quantile"])

        if pd.isna(atr_percent.iloc[check_idx]) or pd.isna(atr.iloc[check_idx]):
            return None
        atr_multi = self.cfg["atr_multi"]
        atr_ok = float(atr.iloc[check_idx]) <= float(atr_percent.iloc[check_idx]) * atr_multi

        n = self.cfg["box_n"]
        if len(df) < n:
            return None

        # Calculate box on the same check_idx
        box_high = float(df["high"].rolling(n, min_periods=n).quantile(0.95).iloc[check_idx])
        box_low = float(df["low"].rolling(n, min_periods=n).quantile(0.05).iloc[check_idx])
        mid = (float(box_high) + float(box_low)) / 2.0
        if mid <= 0:
            return None
        box_width = (box_high - box_low) / mid
    
        box_width_min = self.cfg["box_width_min"]
        box_width_max = self.cfg["box_width_max"]   
        box_ok = box_width_min <= box_width <= box_width_max
        if not (atr_ok and box_ok):
            return None

        return {
            "box_high": box_high,
            "box_low": box_low,
            "box_width": float(box_width),
            "atr15": float(atr.iloc[check_idx]),
        }
