from typing import Dict, Optional, List

import pandas as pd

from core.datasource import BinanceDataSource
from core.indicators import Indicators
from core.strategy.base import BaseStrategy
from core.strategy.config import StrategyConfig


class TopDistributionShortStrategy(BaseStrategy):
    def __init__(self):
        self._last_signal_time: Dict[str, int] = {}  # 记录每个交易对的最后信号时间
        self._utad_candles: Dict[str, Dict[str, object]] = {}  # 记录UTAD信号，包含candle和timestamp
        self.cfg = StrategyConfig.TOP_DISTRIBUTION_SHORT

    def analyze(self, symbol: str, ds: BinanceDataSource, **kwargs: object) -> Optional[Dict[str, object]]:
        # 获取15分钟K线数据
        m15 = ds.get_klines_df(symbol, "15m", self.cfg["kline_15m_limit"])
        if m15 is None or m15.empty or len(m15) < self.cfg["min_15m_candles"]:
            return None

        now_ms = kwargs.get("now_ms")
        if now_ms is None:
            now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        else:
            now_ms = int(now_ms)

        # 获取最后一根完整的K线索引
        last_idx = self._last_closed_idx(m15, now_ms)
        if last_idx is None:
            return None

        abs_idx = last_idx if last_idx >= 0 else len(m15) + last_idx
        if abs_idx < 0 or abs_idx >= len(m15):
            return None

        # 1. 识别顶部区域（派发区）- 使用abs_idx作为结束索引
        distribution_zone = self._identify_distribution_zone(m15, end_idx=abs_idx)
        if not distribution_zone:
            # 清空该交易对的UTAD记录
            self._utad_candles.pop(symbol, None)
            return None

        zone_high = float(distribution_zone["zone_high"])
        zone_low = float(distribution_zone["zone_low"])
        zone_mid = (zone_high + zone_low) / 2
        atr15 = float(distribution_zone["atr15"])

        # 2. 检测UTAD（假突破）信号
        current_candle = m15.iloc[abs_idx]
        utad_signal = self._detect_utad(current_candle, zone_high, atr15)

        if utad_signal:
            # 记录UTAD蜡烛及其所有相关参数（冻结参数，避免后续变化）
            current_close_time = current_candle["close_time"]
            if isinstance(current_close_time, pd.Timestamp):
                utad_time_ms = int(current_close_time.timestamp() * 1000)
            else:
                try:
                    utad_time_ms = int(current_close_time)
                except (TypeError, ValueError):
                    utad_time_ms = int(pd.Timestamp.now().timestamp() * 1000)
            
            self._utad_candles[symbol] = {
                "candle": current_candle,
                "timestamp": utad_time_ms,
                "index": abs_idx,
                "zone_high": zone_high,
                "zone_low": zone_low,
                "zone_mid": zone_mid,
                "zone_width": float(distribution_zone["zone_width"]),
                "atr15": atr15,
                "utad_high": float(current_candle["high"]),
                "utad_low": float(current_candle["low"]),
                "utad_close": float(current_candle["close"])
            }
            return None  # 等待确认信号

        # 3. 如果已有UTAD记录，检查确认入场条件
        if symbol in self._utad_candles:
            utad_record = self._utad_candles[symbol]
            utad_candle = utad_record["candle"]
            utad_index = utad_record["index"]
            utad_time = utad_record["timestamp"]
            
            # 使用存储的参数（冻结值），避免后续变化
            stored_zone_high = float(utad_record["zone_high"])
            stored_zone_low = float(utad_record["zone_low"])
            stored_zone_mid = float(utad_record["zone_mid"])
            stored_zone_width = float(utad_record["zone_width"])
            stored_atr15 = float(utad_record["atr15"])
            stored_utad_high = float(utad_record["utad_high"])
            stored_utad_low = float(utad_record["utad_low"])
            stored_utad_close = float(utad_record["utad_close"])
            
            # 检查UTAD失效条件：价格重新强势站上UTAD高点，UTAD假突破逻辑被推翻
            current_candle = m15.iloc[abs_idx]
            current_high = float(current_candle["high"])
            current_close = float(current_candle["close"])
            
            # 如果价格突破UTAD高点+ATR padding，清UTAD记录
            if current_high > stored_utad_high + 0.1 * stored_atr15:
                self._utad_candles.pop(symbol, None)
                return None
            
            # 检查UTAD是否超时 - 使用时间差计算，更稳健
            current_close_time = current_candle["close_time"]
            if isinstance(current_close_time, pd.Timestamp):
                current_close_time_ms = int(current_close_time.timestamp() * 1000)
            else:
                try:
                    current_close_time_ms = int(current_close_time)
                except (TypeError, ValueError):
                    current_close_time_ms = int(pd.Timestamp.now().timestamp() * 1000)
            
            # 计算时间差（毫秒），转换为15m K线数量
            time_diff_ms = current_close_time_ms - utad_time
            bars_since_utad = time_diff_ms // (15 * 60 * 1000)
            
            if bars_since_utad > self.cfg["confirm_wait_bars"]:
                # UTAD已超时，清空记录
                self._utad_candles.pop(symbol, None)
                return None
            
            entry_signal = self._check_entry_confirmation(
                m15, abs_idx, 
                utad_low=stored_utad_low, 
                zone_high=stored_zone_high, 
                zone_low=stored_zone_low, 
                zone_mid=stored_zone_mid, 
                atr15=stored_atr15
            )

            if entry_signal:
                # 计算风险收益比和入场参数
                entry = float(current_candle["close"])
                # 止损 = UTAD高点 + ATR padding，避免止损太紧
                stop = stored_utad_high + self.cfg["sl_pad_atr"] * stored_atr15
                risk = max(stop - entry, 0.000001)
                # 目标价格使用结构目标（更符合顶部策略）
                target1 = stored_zone_mid  # 第一目标：zone_mid
                target2 = stored_zone_low  # 第二目标：zone_low

                # 计算风险收益比 - 使用结构目标
                rr1 = (entry - target1) / risk if risk > 0 else 0.0
                rr2 = (entry - target2) / risk if risk > 0 else 0.0
                score = self._calculate_score(entry_signal)

                # 生成信号 - 处理不同类型的close_time
                current_close_time = current_candle["close_time"]
                if isinstance(current_close_time, pd.Timestamp):
                    close_time_ms = int(current_close_time.timestamp() * 1000)
                else:
                    try:
                        close_time_ms = int(current_close_time)
                    except (TypeError, ValueError):
                        return None
                
                self._last_signal_time[symbol] = close_time_ms
                self._utad_candles.pop(symbol, None)  # 清空UTAD记录

                return self.build_signal(
                    symbol=symbol,
                    time_ms=close_time_ms,
                    rr=round(float(rr1), 2),  # 主要使用zone_mid的RR值
                    score=int(score),
                    mode="top_distribution_short",
                    zone_high=stored_zone_high,
                    zone_low=stored_zone_low,
                    zone_mid=stored_zone_mid,
                    zone_width=stored_zone_width,
                    utad_time=utad_time,
                    utad_high=stored_utad_high,
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target1),  # 主要目标：zone_mid
                    target2=float(target2),  # 第二目标：zone_low
                    rr2=round(float(rr2), 2),  # zone_low的RR值
                    atr15=stored_atr15,
                    entry_signal_type=entry_signal["type"],
                )

        return None

    def _identify_distribution_zone(self, df: pd.DataFrame, end_idx: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        识别顶部区域（派发区）
        在15m上取最近N根K线，计算zone_high, zone_low, zone_width
        
        Args:
            df: K线数据
            end_idx: 计算的结束索引，使用该索引对应的K线作为最后一根
        """
        n = self.cfg["distribution_zone_n"]
        
        # 确定结束索引
        if end_idx is None:
            end_idx = len(df) - 1
        elif end_idx < 0:
            end_idx = len(df) + end_idx
        
        # 检查数据是否足够
        if end_idx < n - 1:
            return None

        # 获取计算窗口
        window_start = end_idx - n + 1
        window_end = end_idx + 1  # pandas切片是左闭右开
        window_df = df.iloc[window_start:window_end]

        # 计算zone_high和zone_low
        zone_high = float(window_df["high"].quantile(self.cfg["zone_high_quantile"], interpolation="linear"))
        zone_low = float(window_df["low"].quantile(self.cfg["zone_low_quantile"], interpolation="linear"))

        mid = (zone_high + zone_low) / 2
        zone_width = (zone_high - zone_low) / mid

        # 检查zone_width是否在合理范围
        if not (self.cfg["zone_width_min"] <= zone_width <= self.cfg["zone_width_max"]):
            return None

        # 计算触碰上沿的次数
        touch_count = 0
        for i in range(window_start, window_end):
            high = float(df["high"].iloc[i])
            # 计算价格与zone_high的接近程度
            if high >= zone_high - self.cfg["touch_tolerance_pct"] * zone_high:
                touch_count += 1

        # 检查触碰次数是否达到要求
        if touch_count < self.cfg["touch_min"]:
            return None

        # 计算ATR
        atr_series = Indicators.atr(df["high"], df["low"], df["close"], self.cfg["atr_period"])
        if len(atr_series) <= end_idx or pd.isna(atr_series.iloc[end_idx]):
            return None
        atr15 = float(atr_series.iloc[end_idx])

        # 可选：检查ATR是否下降
        if self.cfg["require_atr_decrease"]:
            # 计算前半段ATR
            prev_window_start = end_idx - n
            prev_window_end = end_idx - n//2
            if prev_window_start < 0:
                return None
                
            prev_atr_series = Indicators.atr(
                df["high"].iloc[prev_window_start:prev_window_end], 
                df["low"].iloc[prev_window_start:prev_window_end], 
                df["close"].iloc[prev_window_start:prev_window_end], 
                self.cfg["atr_period"]
            )
            
            if len(prev_atr_series) == 0:
                return None
                
            atr_prev_period = float(prev_atr_series.mean())
            
            if atr15 >= atr_prev_period * self.cfg["atr_decrease_ratio"]:
                return None

        return {
            "zone_high": zone_high,
            "zone_low": zone_low,
            "zone_width": zone_width,
            "atr15": atr15
        }

    def _detect_utad(self, candle: pd.Series, zone_high: float, atr15: float) -> bool:
        """
        检测假突破（UTAD）信号
        刺破上沿但收回区间内，长上影，收盘偏下
        """
        h = float(candle["high"])
        c = float(candle["close"])
        l = float(candle["low"])

        # 刺破上沿：high > zone_high + 0.15 * ATR
        pierce_upper = h > zone_high + self.cfg["utad_pierce_factor"] * atr15

        # 收回区间：close < zone_high
        reclaim = c < zone_high

        # 计算上影线比例
        candle_range = max(h - l, 0.000001)
        upper_wick = h - max(float(candle["open"]), c)
        upper_wick_ratio = upper_wick / candle_range

        # 上影明显：upper_wick_ratio > 0.50
        wick_ok = upper_wick_ratio > self.cfg["utad_upper_wick_ratio"]

        # 收盘偏下：close_pos < 0.45
        close_pos = (c - l) / candle_range
        close_pos_ok = close_pos < self.cfg["utad_close_pos_max"]

        return pierce_upper and reclaim and wick_ok and close_pos_ok

    def _check_entry_confirmation(self, df: pd.DataFrame, current_idx: int, 
                               utad_low: float, zone_high: float, zone_low: float, 
                               zone_mid: float, atr15: float) -> Optional[Dict[str, str]]:
        """
        检查入场确认条件
        1. 破坏结构：close < zone_mid
        2. 跌破UTAD低点：close < utad_low - 0.10*ATR
        3. 回抽不过：价格回到zone_high附近，但收盘仍 < zone_high且出现阴线/上影
        """
        current_candle = df.iloc[current_idx]
        c = float(current_candle["close"])
        l = float(current_candle["low"])
        h = float(current_candle["high"])

        # 1. 破坏结构：close < zone_mid
        if c < zone_mid:
            return {"type": "structure_break"}

        # 2. 跌破UTAD低点：close < utad_low - 0.10*ATR
        if c < utad_low - self.cfg["entry_break_utad_low_factor"] * atr15:
            return {"type": "break_utad_low"}

        # 3. 回抽不过：价格回到zone_high附近，但收盘仍 < zone_high
        if h >= zone_high - self.cfg["entry_retest_tolerance_pct"] * zone_high and c < zone_high:
            # 检查是否出现阴线或上影
            o = float(current_candle["open"])
            candle_range = max(h - l, 0.000001)
            upper_wick = h - max(o, c)
            upper_wick_ratio = upper_wick / candle_range

            # 阴线或上影明显
            if c < o or upper_wick_ratio > self.cfg["entry_retest_wick_ratio"]:
                return {"type": "retest_failure"}

        return None

    def _calculate_score(self, entry_signal: Dict[str, str]) -> int:
        """
        计算信号分数
        """
        base_score = 50
        score = base_score

        # 根据入场信号类型调整分数
        if entry_signal["type"] == "structure_break":
            score += 20
        elif entry_signal["type"] == "break_utad_low":
            score += 15
        elif entry_signal["type"] == "retest_failure":
            score += 10

        return min(100, score)

    def _last_closed_idx(self, df: pd.DataFrame, now_ms: int) -> Optional[int]:
        """
        获取最后一根完整的K线索引
        """
        if df is None or df.empty or "close_time" not in df.columns:
            return None
        if len(df) < 2:
            return None

        # 处理不同类型的close_time
        last_ct = df["close_time"].iloc[-1]
        if isinstance(last_ct, pd.Timestamp):
            last_ct = int(last_ct.timestamp() * 1000)
        else:
            try:
                last_ct = int(last_ct)
            except (TypeError, ValueError):
                return None
                
        # 如果最后一根K线已经收盘至少2秒，则返回-1
        if last_ct <= now_ms - 2000:
            return -1
        # 否则返回-2（倒数第二根）
        return -2
