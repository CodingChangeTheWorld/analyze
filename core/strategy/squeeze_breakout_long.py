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
        m15 = ds.get_klines_df(symbol, "15m", 500)
        if m15 is None or m15.empty or len(m15) < 120:
            return None

        squeeze = self._squeeze_box_15m(m15)
        if not squeeze:
            return None

        box_high = float(squeeze["box_high"])
        box_low = float(squeeze["box_low"])
        box_width = float(squeeze["box_width"])
        atr15 = float(squeeze["atr15"])

        m5 = ds.get_klines_df(symbol, "5m", 400)
        if m5 is None or m5.empty or len(m5) < 80:
            return None

        now_ms = kwargs.get("now_ms")
        if now_ms is None:
            now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        else:
            now_ms = int(now_ms)

        idx = self._last_closed_idx(m5, now_ms)
        if idx is None:
            return None

        close_time_ms = int(m5["close_time"].iloc[idx])
        last_sig = self._last_signal_time.get(symbol)
        if last_sig == close_time_ms:
            return None

        atr5_series = Indicators.atr(m5["high"], m5["low"], m5["close"], 14)
        atr5 = float(atr5_series.iloc[idx]) if pd.notna(atr5_series.iloc[idx]) else 0.0
        if atr5 <= 0:
            return None

        level = box_high
        k1 = 0.4
        k2 = 1.8
        k3 = 2.5
        close_pos_min = 0.65
        min_conditions = 3
        stand_m = 3
        stand_low_atr = 0.5

        c = float(m5["close"].iloc[idx])
        h = float(m5["high"].iloc[idx])
        l = float(m5["low"].iloc[idx])
        v = float(m5["volume"].iloc[idx])
        candle_range = max(h - l, 0.0)
        close_pos = (c - l) / candle_range if candle_range > 0 else 0.0

        vol_ma = m5["volume"].rolling(48, min_periods=48).mean()
        
        # Convert negative index to positive to avoid >= check failure
        abs_idx = idx if idx >= 0 else len(m5) + idx
        
        vol_base = float(vol_ma.iloc[abs_idx]) if abs_idx >= 47 and pd.notna(vol_ma.iloc[abs_idx]) else 0.0
        vol_ratio = (v / vol_base) if vol_base > 0 else 0.0

        cond_close_break = c > level + k1 * atr5
        cond_range_expand = (candle_range / atr5) > k2 if atr5 > 0 else False
        cond_vol_shock = vol_ratio > k3
        cond_close_pos = close_pos > close_pos_min
        conds = [cond_close_break, cond_range_expand, cond_vol_shock, cond_close_pos]
        cond_count = sum(1 for x in conds if x)

        # stand_slice = m5.iloc[max(0, idx - stand_m + 1) : idx + 1]
        # stand_ok = False
        # if not stand_slice.empty:
        #     stand_close_ok = bool((stand_slice["close"] > level).any())
        #     stand_low_ok = bool(float(stand_slice["low"].min()) >= level - stand_low_atr * atr5)
        #     stand_ok = stand_close_ok and stand_low_ok
        stand_ok = True
        
        print(f"DEBUG ANALYZE: Conds={cond_count}/{min_conditions} Stand={stand_ok} CloseBreak={cond_close_break} RangeExp={cond_range_expand} VolShock={cond_vol_shock} ClosePos={cond_close_pos}")

        if cond_count >= min_conditions and stand_ok:
            entry = c
            ksl = 1.0
            stop = level - ksl * atr5
            risk = max(entry - stop, 0.0)
            tp1_r = 1.2
            tp2_r = 3.0
            tp1 = entry + tp1_r * risk if risk > 0 else 0.0
            tp2 = entry + tp2_r * risk if risk > 0 else 0.0
            trail_m = 3.0
            highest_close_15m = float(m15["close"].iloc[-20:].max()) if len(m15) >= 20 else float(m15["close"].max())
            trail_stop_15m = highest_close_15m - trail_m * atr15 if atr15 > 0 else 0.0

            score = int(cond_count) + (1 if vol_ratio >= k3 * 1.2 else 0)
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

        recent_break = bool((m5["close"].iloc[max(0, idx - 24) : idx] > level + k1 * atr5).any())
        if not recent_break:
            return None

        krt = 0.6
        band = krt * atr5
        hold_n = 2
        hold_slice = m5.iloc[max(0, idx - hold_n + 1) : idx + 1]
        hold_ok = bool((hold_slice["close"] >= level).all()) if len(hold_slice) == hold_n else False
        in_band = abs(c - level) <= band or abs(l - level) <= band
        if not (hold_ok and in_band):
            return None

        candle = m5.iloc[idx]
        confirm = self.is_level_rejection(
            candle=candle,
            level=level,
            side="long",
            price_tolerance_pct=0.002,
            wick_ratio_min=0.5,
            close_pos_min_for_long=0.6,
            require_reclaim=True,
        )
        if not confirm:
            return None

        entry = c
        ksl2 = 0.6
        stop = l - ksl2 * atr5
        risk = max(entry - stop, 0.0)
        tp1_r = 1.2
        tp2_r = 3.0
        tp1 = entry + tp1_r * risk if risk > 0 else 0.0
        tp2 = entry + tp2_r * risk if risk > 0 else 0.0
        score = 3 + (1 if vol_ratio > 1.5 else 0)
        rr = (tp2 - entry) / risk if risk > 0 else 0.0

        self._last_signal_time[symbol] = close_time_ms
        return self.build_signal(
            symbol=symbol,
            time_ms=close_time_ms,
            rr=round(float(rr), 2),
            score=int(score),
            mode="retest_entry",
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
            vol_ratio=float(vol_ratio),
            close_pos=float(close_pos),
            hold_ok=bool(hold_ok),
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
        if df is None or df.empty or len(df) < 120:
            return None

        # Use -2 (last fully closed candle) instead of -1 (forming or just closed but potentially exploding)
        # to detect the squeeze state.
        # This prevents the breakout candle itself from ruining the squeeze metrics.
        check_idx = -2
        if len(df) + check_idx < 0:
            return None
        
        atr = Indicators.atr(df["high"], df["low"], df["close"], 14)
        window = 96
        # Tighten to 0.4 for real squeeze
        q30 = Indicators.rolling_quantile(atr, window, 0.4)
        
        if pd.isna(q30.iloc[check_idx]) or pd.isna(atr.iloc[check_idx]):
            return None

        atr_ok = float(atr.iloc[check_idx]) <= float(q30.iloc[check_idx])

        n = 32
        if len(df) < n:
            return None
        
        # Calculate box on the same check_idx
        box_high = float(df["high"].rolling(n, min_periods=n).quantile(0.95).iloc[check_idx])
        box_low = float(df["low"].rolling(n, min_periods=n).quantile(0.05).iloc[check_idx])
        mid = (float(box_high) + float(box_low)) / 2.0
        if mid <= 0:
            return None
        box_width = (box_high - box_low) / mid
        box_ok = 0.02 <= float(box_width) <= 0.06
        
        # DEBUG PRINTS
        # if check_idx == -2:
        #     print(f"DEBUG SQUEEZE: Time={df['close_time'].iloc[check_idx]} ATR={atr.iloc[check_idx]:.5f} Q={q30.iloc[check_idx]:.5f} ATR_OK={float(atr.iloc[check_idx]) <= float(q30.iloc[check_idx])} BoxWidth={box_width:.4f} High={box_high:.5f} Low={box_low:.5f}")

        if not (atr_ok and box_ok):
            return None

        return {
            "box_high": box_high,
            "box_low": box_low,
            "box_width": float(box_width),
            "atr15": float(atr.iloc[check_idx]),
        }
