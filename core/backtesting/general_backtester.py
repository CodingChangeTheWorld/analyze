import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd

from core.datasource import RESTClient
from core.strategy import SqueezeBreakoutLongStrategy, TopDistributionShortStrategy, TrendFollowingLongStrategy


class MockDataSource:
    """
    模拟BinanceDataSource的类，用于回测
    """
    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self.data_map = data_map

    def get_klines_df(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """模拟获取K线数据"""
        df = self.data_map.get(interval)
        if df is None or df.empty:
            return None
        return df.iloc[max(0, len(df) - limit) :]


class GeneralBacktester:
    """
    通用回测类，支持指定代币、日期范围和策略进行回测
    
    功能特性：
    - 支持Squeeze突破策略和Pullback回踩策略
    - 自动拉取所需的K线数据（5m, 15m, 1h, 4h）
    - 支持多日连续回测
    - 自动分析信号结果（成功/止损/未入场）
    - 支持北京时间显示
    
    使用示例：
    >>> backtester = GeneralBacktester()
    >>> results = backtester.backtest_date_range("QUSDT", "2025-12-30", "2026-01-09", "squeeze")
    >>> backtester.print_summary(results)
    
    命令行使用：
    python tools/general_backtest.py QUSDT 2025-12-30 2026-01-09 squeeze
    """
    
    def __init__(self):
        self.rest_client = RESTClient()
        self.strategy_map = {
            "squeeze": SqueezeBreakoutLongStrategy,
            "top_distribution": TopDistributionShortStrategy,
            "trend_following": TrendFollowingLongStrategy
        }
        
    def _interval_ms(self, interval: str) -> int:
        """获取K线周期的毫秒数"""
        if interval == "15m":
            return 15 * 60_000
        if interval == "1h":
            return 60 * 60_000
        if interval == "4h":
            return 4 * 60 * 60_000
        if interval == "1m":
            return 60_000
        if interval == "5m":
            return 5 * 60_000
        raise ValueError(f"不支持的周期: {interval}")
    
    def _date_range_utc_ms(self, date_str: str) -> Tuple[int, int]:
        """将日期字符串转换为UTC时间戳范围（日期字符串为北京时间）"""
        # 解析为北京时间（UTC+8）
        d = datetime.strptime(date_str, "%Y-%m-%d")
        # 转换为UTC时间（减去8小时）
        d_utc = d - timedelta(hours=8)
        d_utc = d_utc.replace(tzinfo=timezone.utc)
        day_start = int(d_utc.timestamp() * 1000)
        day_end = int((d_utc + timedelta(days=1)).timestamp() * 1000)
        return day_start, day_end
    
    def _fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500) -> pd.DataFrame:
        """获取K线数据"""
        out = []
        cur = start_ms
        
        while cur < end_ms:
            df = self.rest_client.get_klines_raw(symbol, interval, limit=limit, start_time=cur, end_time=end_ms - 1)
            if df.empty:
                break
            
            out.append(df)
            last_close = int(df["close_time"].iloc[-1])
            nxt = last_close + 1
            
            if nxt <= cur:
                break
            
            cur = nxt
            time.sleep(0.05)  # 避免请求过快
        
        if not out:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
        
        df_all = pd.concat(out, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return df_all
    
    def backtest_single_day(self, symbol: str, date_str: str, strategy_name: str = "squeeze") -> Tuple[List[Dict], List[Dict]]:
        """回测单日数据
        
        Args:
            symbol: 代币名称，如 QUSDT
            date_str: 日期字符串，格式 YYYY-MM-DD
            strategy_name: 策略名称，squeeze 或 pullback
            
        Returns:
            信号列表和结果分析列表
        """
        day_start, day_end = self._date_range_utc_ms(date_str)
        
        print(f"拉取数据: {symbol} {date_str} (UTC)...")
        
        data_map = {}
        signals = []
        
        if strategy_name == "squeeze":
            # Squeeze策略需要的K线周期
            m15_lb = 500
            m5_lb = 500
            
            m15 = self._fetch_klines(symbol, "15m", day_start - m15_lb * self._interval_ms("15m"), day_end)
            m5 = self._fetch_klines(symbol, "5m", day_start - m5_lb * self._interval_ms("5m"), day_end)
            
            if len(m15) < 120 or len(m5) < 80:
                print(f"数据不足: m15={len(m15)} m5={len(m5)}")
                return signals, []
                
            data_map["15m"] = m15
            data_map["5m"] = m5
            
            driver_interval = "5m"
            driver_df = m5
            strategy = SqueezeBreakoutLongStrategy()
            
        elif strategy_name == "top_distribution":
            # Top Distribution策略需要的K线周期
            m15_lb = 500
            
            m15 = self._fetch_klines(symbol, "15m", day_start - m15_lb * self._interval_ms("15m"), day_end)
            
            if len(m15) < 128:
                print(f"数据不足: m15={len(m15)}")
                return signals, []
                
            data_map["15m"] = m15
            
            driver_interval = "15m"
            driver_df = m15
            strategy = TopDistributionShortStrategy()
            
        elif strategy_name == "trend_following":
            # Trend Following策略需要的K线周期
            m15_lb = 500
            
            m15 = self._fetch_klines(symbol, "15m", day_start - m15_lb * self._interval_ms("15m"), day_end)
            
            if len(m15) < 50:
                print(f"数据不足: m15={len(m15)}")
                return signals, []
                
            data_map["15m"] = m15
            
            driver_interval = "15m"
            driver_df = m15
            strategy = TrendFollowingLongStrategy()
            
        else:
            print(f"未知策略: {strategy_name}")
            return signals, []
        
        # 筛选当日的驱动K线
        driver_day = driver_df[(driver_df["open_time"] >= day_start) & (driver_df["open_time"] < day_end)].reset_index(drop=True)
        if driver_day.empty:
            print(f"当日无 {driver_interval} K 线数据。")
            return signals, []
        
        # 模拟回测
        last_key: tuple[int, float] | None = None
        
        for i in range(len(driver_day)):
            # 使用当前K线及其之前的所有K线进行分析
            current_df = driver_df.iloc[:len(driver_df) - len(driver_day) + i + 1]
            current_ts = current_df["close_time"].iloc[-1]
            
            # 为每个周期创建包含当前时间戳之前所有数据的新数据映射
            current_data_map = {}
            for k, v in data_map.items():
                # 根据时间戳筛选数据，而不是根据索引
                current_data = v[v["close_time"] <= current_ts]
                current_data_map[k] = current_data
            
            # 创建模拟数据源
            mock_ds = MockDataSource(current_data_map)
            
            # 分析信号
            sig = strategy.analyze(symbol, mock_ds, now_ms=current_ts)
            if sig:
                # 添加策略名称和时间戳
                sig["strategy"] = strategy_name
                sig["time"] = current_ts
                signals.append(sig)
                
        # 分析信号结果
        analyzer = OutcomeAnalyzer(driver_df, strategy_name)
        outcomes = analyzer.analyze_outcomes(signals)
        
        return signals, outcomes
    
    def backtest_date_range(self, symbol: str, start_date: str, end_date: str, strategy_name: str = "squeeze") -> Dict:
        """回测日期范围
        
        Args:
            symbol: 代币名称，如 QUSDT
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD
            strategy_name: 策略名称，squeeze 或 pullback
            
        Returns:
            回测结果，包含信号、结果和统计信息
        """
        # 转换日期
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        current = start
        all_signals = []
        all_outcomes = []
        
        print(f"开始测试 {symbol} {start_date} 到 {end_date} 的 {strategy_name} 策略信号...")
        print("=" * 70)
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            print(f"\n测试日期: {date_str}")
            print("-" * 30)
            
            # 回测单日
            signals, outcomes = self.backtest_single_day(symbol, date_str, strategy_name)
            all_signals.extend(signals)
            all_outcomes.extend(outcomes)
            
            current += timedelta(days=1)
            time.sleep(0.1)  # 避免请求过快
        
        # 统计结果
        total_signals = len(all_signals)
        total_target = sum(1 for o in all_outcomes if o["result"] == "target")
        total_stop = sum(1 for o in all_outcomes if o["result"] == "stop")
        total_no_entry = sum(1 for o in all_outcomes if o["result"] == "no_entry")
        
        # 计算胜率
        win_rate = total_target / (total_target + total_stop) if (total_target + total_stop) > 0 else 0
        
        results = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "strategy": strategy_name,
            "total_signals": total_signals,
            "total_target": total_target,
            "total_stop": total_stop,
            "total_no_entry": total_no_entry,
            "win_rate": win_rate,
            "signals": all_signals,
            "outcomes": all_outcomes
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """打印回测结果摘要
        
        Args:
            results: 回测结果，包含信号、结果和统计信息
        """
        print(f"\n{'=' * 70}")
        print(f"回测结果摘要")
        print(f"{'=' * 70}")
        print(f"代币: {results['symbol']}")
        print(f"日期范围: {results['start_date']} 到 {results['end_date']}")
        print(f"策略: {results['strategy']}")
        print(f"总信号数: {results['total_signals']}")
        print(f"成功信号: {results['total_target']}")
        print(f"止损信号: {results['total_stop']}")
        print(f"未入场: {results['total_no_entry']}")
        print(f"胜率: {results['win_rate']:.2%}")
        
        if results['outcomes']:
            print(f"\n信号详情:")
            print(f"{'=' * 70}")
            for i, (signal, outcome) in enumerate(zip(results['signals'], results['outcomes'])):
                signal_time = self._fmt_ts(int(signal['time']))
                entry_time = self._fmt_ts(int(outcome['entry_time'])) if outcome['entry_time'] else "-"
                exit_time = self._fmt_ts(int(outcome['exit_time'])) if outcome['exit_time'] else "-"
                
                print(f"信号 {i+1}:")
                print(f"   时间: {signal_time}")
                print(f"   类型: {signal.get('strategy', '')}")
                print(f"   入场: {entry_time}")
                print(f"   出场: {exit_time}")
                print(f"   结果: {outcome['result']}")
                
                # 打印策略特定参数
                if 'tp1' in signal and 'tp2' in signal:
                    print(f"   TP1: {signal['tp1']:.6f}, TP2: {signal['tp2']:.6f}")
                elif 'target' in signal:
                    print(f"   目标: {signal['target']:.6f}")
                
                print(f"   RR: {signal.get('rr', '')}, 分数: {signal.get('score', '')}")
                print(f"-" * 50)
    
    def _fmt_ts(self, ms: int) -> str:
        """格式化时间戳为北京时间
        
        对于15分钟K线，显示K线的开始时间而不是结束时间
        20:15-20:29的K线会显示为20:15而不是20:29
        """
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        
        # 对于15分钟K线，计算开始时间
        # 检查是否是15分钟K线的结束时间（接近整点或15/30/45分钟）
        if dt.minute in [14, 29, 44, 59] and dt.second >= 59 and dt.microsecond >= 999000:
            # 计算开始时间：将分钟数调整为15的倍数
            # 20:29:59.999 → 分钟数29 → 29 - (29 % 15) = 15 → 20:15
            start_minute = dt.minute - (dt.minute % 15)
            dt = dt.replace(minute=start_minute, second=0, microsecond=0)
        
        # 转换为北京时间
        dt_cn = dt + timedelta(hours=8)
        return dt_cn.strftime("%Y-%m-%d %H:%M")


# 结果分析器
class OutcomeAnalyzer:
    """
    信号结果分析器，分析信号的结果
    """
    
    def __init__(self, driver_df: pd.DataFrame, strategy_name: str):
        self.driver_df = driver_df
        self.strategy_name = strategy_name
    
    def analyze_outcomes(self, signals: List[Dict]) -> List[Dict]:
        """分析信号结果
        
        Args:
            signals: 信号列表
            
        Returns:
            结果分析列表
        """
        outcomes = []
        
        for s in signals:
            entry = float(s["entry"])
            stop = float(s["stop"])
            t0 = int(s["time"])
            
            # 检查策略类型
            if "tp1" in s and "tp2" in s:
                target = float(s["tp1"])  # 使用tp1作为目标
            elif "target" in s:
                target = float(s["target"])
            else:
                print(f"警告: 信号没有目标价格 {s}")
                continue
            
            # 确定方向
            is_long = target > entry
            
            # 筛选信号时间后的K线
            df_after = self.driver_df[self.driver_df["close_time"] >= t0].reset_index(drop=True)
            entered = False
            entry_time = None
            result = "no_entry"
            exit_time = None
            
            # 检查入场模式
            mode = s.get("mode", "")
            is_market_entry = mode in ["breakout_chase", "retest_entry"] or self.strategy_name == "squeeze"
            
            for row in df_after.itertuples():
                hi = float(row.high)
                lo = float(row.low)
                ct = int(row.close_time)
                
                if not entered:
                    if is_market_entry:
                        # 市价入场，立即入场
                        entered = True
                        entry_time = ct
                    else:
                        # 限价/止损入场逻辑
                        if is_long:
                            if lo <= entry:
                                entered = True
                                entry_time = ct
                        else:
                            if hi >= entry:
                                entered = True
                                entry_time = ct
                
                if entered:
                    # 检查止盈止损
                    if is_long:
                        if hi >= target:
                            result = "target"
                            exit_time = ct
                            break
                        if lo <= stop:
                            result = "stop"
                            exit_time = ct
                            break
                    else:
                        if lo <= target:
                            result = "target"
                            exit_time = ct
                            break
                        if hi >= stop:
                            result = "stop"
                            exit_time = ct
                            break
            
            outcomes.append({
                "time": t0,
                "result": result,
                "entry_time": entry_time,
                "exit_time": exit_time
            })
        
        return outcomes
