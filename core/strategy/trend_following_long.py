from typing import Dict, Optional, List

import pandas as pd

from core.datasource import BinanceDataSource
from core.indicators import Indicators
from core.strategy.base import BaseStrategy
from core.strategy.config import StrategyConfig


class TrendFollowingLongStrategy(BaseStrategy):
    def __init__(self):
        self._last_signal_time: Dict[str, int] = {}  # 记录每个交易对的最后信号时间
        self.cfg = StrategyConfig.TREND_FOLLOWING_LONG

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
        m15_last_idx = self._last_closed_idx(m15, now_ms)
        
        if m15_last_idx is None:
            return None

        m15_abs_idx = m15_last_idx if m15_last_idx >= 0 else len(m15) + m15_last_idx
        
        if m15_abs_idx < 0 or m15_abs_idx >= len(m15):
            return None

        # 计算15min的EMA10
        m15_close = pd.to_numeric(m15["close"], errors="coerce")
        m15_ema10 = Indicators.ema(m15_close, self.cfg["m15_ema_period"])
        
        if len(m15_ema10) <= m15_abs_idx or pd.isna(m15_ema10.iloc[m15_abs_idx]):
            return None

        # 条件1：连续n根15min K线的最低价都在ema10之上
        if not self._check_m15_ema_above(m15, m15_ema10, m15_abs_idx):
            return None

        # 条件2：closeighow不断抬升
        if not self._check_prices_rising(m15, m15_abs_idx):
            return None

        # 条件3：K线实体需要占整根K线的60%以上
        if not self._check_candle_body_ratio(m15, m15_abs_idx):
            return None

        # 条件4：每根K线的量能要大于最近n=9根K线的SMA的值
        if not self._check_volume_condition(m15, m15_abs_idx):
            return None

        # 检查信号频率限制
        last_signal_time = self._last_signal_time.get(symbol, 0)
        if now_ms - last_signal_time < self.cfg["signal_cooldown_ms"]:
            return None

        # 生成信号
        entry = float(m15["close"].iloc[m15_abs_idx])
        
        # 使用15分钟K线的最低价作为止损（去掉1小时EMA20）
        stop = float(m15["low"].iloc[m15_abs_idx])
        
        # 目标价格设置
        risk = max(entry - stop, 0.000001)
        target1 = entry + self.cfg["tp1_r"] * risk
        target2 = entry + self.cfg["tp2_r"] * risk
        
        # 计算风险收益比
        rr1 = self.cfg["tp1_r"]
        rr2 = self.cfg["tp2_r"]
        
        # 计算信号分数
        score = self._calculate_score()

        # 生成信号 - 处理不同类型的close_time
        current_close_time = m15["close_time"].iloc[m15_abs_idx]
        if isinstance(current_close_time, pd.Timestamp):
            close_time_ms = int(current_close_time.timestamp() * 1000)
        else:
            try:
                close_time_ms = int(current_close_time)
            except (TypeError, ValueError):
                return None
        
        self._last_signal_time[symbol] = close_time_ms

        return self.build_signal(
            symbol=symbol,
            time_ms=close_time_ms,
            rr=round(float(rr1), 2),
            score=int(score),
            mode="trend_following_long",
            entry=float(entry),
            stop=float(stop),
            target=float(target1),
            target2=float(target2),
            rr2=round(float(rr2), 2),
            m15_ema10=float(m15_ema10.iloc[m15_abs_idx]),
            entry_signal_type="trend_following",
        )

    def _check_m15_ema_above(self, df: pd.DataFrame, ema10: pd.Series, current_idx: int) -> bool:
        """
        条件1：连续n根15min K线的最低价都在ema10之上
        """
        n = self.cfg["consecutive_bars"]
        
        if current_idx < n - 1:
            return False

        for i in range(current_idx - n + 1, current_idx + 1):
            low = float(df["low"].iloc[i])
            current_ema10 = float(ema10.iloc[i])
            
            if low < current_ema10:
                return False

        return True

    def _check_prices_rising(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        条件2：closeighow不断抬升
        """
        n = self.cfg["consecutive_bars"]
        
        if current_idx < n:
            return False

        # 检查close不断抬升
        for i in range(current_idx - n + 1, current_idx):
            close_current = float(df["close"].iloc[i])
            close_next = float(df["close"].iloc[i + 1])
            
            if close_next <= close_current:
                return False

        # 检查high不断抬升
        for i in range(current_idx - n + 1, current_idx):
            high_current = float(df["high"].iloc[i])
            high_next = float(df["high"].iloc[i + 1])
            
            if high_next <= high_current:
                return False

        # 检查low不断抬升
        for i in range(current_idx - n + 1, current_idx):
            low_current = float(df["low"].iloc[i])
            low_next = float(df["low"].iloc[i + 1])
            
            if low_next <= low_current:
                return False

        return True

    def _check_candle_body_ratio(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        条件3：K线实体需要占整根K线的60%以上
        """
        n = self.cfg["consecutive_bars"]
        body_ratio_min = self.cfg["body_ratio_min"]
        
        if current_idx < n - 1:
            return False

        for i in range(current_idx - n + 1, current_idx + 1):
            open_price = float(df["open"].iloc[i])
            close_price = float(df["close"].iloc[i])
            high_price = float(df["high"].iloc[i])
            low_price = float(df["low"].iloc[i])
            
            # 计算K线实体大小和总长度
            body_size = abs(close_price - open_price)
            candle_range = max(high_price - low_price, 0.000001)
            
            # 计算实体比例
            body_ratio = body_size / candle_range
            
            if body_ratio < body_ratio_min:
                return False

        return True

    def _check_volume_condition(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        条件4：任意一根K线的量能要大于最近n根K线的sma的值
        """
        n = self.cfg["consecutive_bars"]
        volume_sma_period = self.cfg["volume_sma_period"]
        
        if current_idx < max(n - 1, volume_sma_period):
            return False

        # 计算成交量SMA
        volume = pd.to_numeric(df["volume"], errors="coerce")
        volume_sma = Indicators.sma(volume, volume_sma_period)
        
        if len(volume_sma) <= current_idx:
            return False

        for i in range(current_idx - n + 1, current_idx + 1):
            current_volume = float(volume.iloc[i])
            current_sma = float(volume_sma.iloc[i])
            
            if current_volume > current_sma:
                return True

        return False



    def _calculate_score(self) -> int:
        """
        计算信号分数
        """
        base_score = 50
        score = base_score

        # 可以根据具体条件调整分数
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
