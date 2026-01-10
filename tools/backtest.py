import time
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.datasource import RESTClient, BinanceDataSource
from core.strategy import PullbackShortStrategy, SqueezeBreakoutLongStrategy


class MockDataSource(BinanceDataSource):
    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self.data_map = data_map

    def get_klines_df(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        df = self.data_map.get(interval)
        if df is None or df.empty:
            return None
        return df.iloc[max(0, len(df) - limit) :]


def fmt_signal(sig: Dict[str, object]) -> str:
    strat = sig.get("strategy", "")
    prefix = f"[{strat}] " if strat else ""
    symbol = sig["symbol"]
    level = round(sig["level"], 6)
    entry = round(sig["entry"], 6)
    stop = round(sig["stop"], 6)
    rr = sig["rr"]
    score = sig["score"]
    
    # Check if it's a squeeze strategy signal with tp1/tp2 or pullback with target
    if "tp1" in sig and "tp2" in sig:
        tp1 = round(sig["tp1"], 6)
        tp2 = round(sig["tp2"], 6)
        return f'{prefix}{symbol} level={level} entry={entry} stop={stop} tp1={tp1} tp2={tp2} rr={rr} score={score}'
    elif "target" in sig:
        target = round(sig["target"], 6)
        return f'{prefix}{symbol} level={level} entry={entry} stop={stop} target={target} rr={rr} score={score}'
    else:
        return f'{prefix}{symbol} level={level} entry={entry} stop={stop} rr={rr} score={score}'


def _date_range_utc_ms(date_str: str) -> tuple[int, int]:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = int(d.timestamp() * 1000)
    end = int((d + timedelta(days=1)).timestamp() * 1000)
    return start, end


def _fetch_klines(rest: RESTClient, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500) -> pd.DataFrame:
    out = []
    cur = start_ms
    while cur < end_ms:
        df = rest.get_klines_raw(symbol, interval, limit=limit, start_time=cur, end_time=end_ms - 1)
        if df.empty:
            break
        out.append(df)
        last_close = int(df["close_time"].iloc[-1])
        nxt = last_close + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)
    if not out:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    df_all = pd.concat(out, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df_all


def _interval_ms(interval: str) -> int:
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
    raise ValueError(f"unsupported interval: {interval}")


def backtest_day(symbol: str, date_str: str, strategy_name: str = "pullback") -> int:
    rest = RESTClient()
    day_start, day_end = _date_range_utc_ms(date_str)

    print(f"拉取数据: {symbol} {date_str} (UTC)...")
    
    data_map = {}
    
    if strategy_name == "pullback":
        h4_lb = 140
        h1_lb = 240
        m15_lb = 320
        h4 = _fetch_klines(rest, symbol, "4h", day_start - h4_lb * _interval_ms("4h"), day_end)
        h1 = _fetch_klines(rest, symbol, "1h", day_start - h1_lb * _interval_ms("1h"), day_end)
        m15 = _fetch_klines(rest, symbol, "15m", day_start - m15_lb * _interval_ms("15m"), day_end)
        
        if len(h4) < 60 or len(h1) < 60 or len(m15) < 60:
            print(f"数据不足: h4={len(h4)} h1={len(h1)} m15={len(m15)}", flush=True)
            return 0
            
        data_map["4h"] = h4
        data_map["1h"] = h1
        data_map["15m"] = m15
        
        driver_interval = "15m"
        driver_df = m15
        strategy = PullbackShortStrategy()
        
    elif strategy_name == "squeeze":
        m15_lb = 500
        m5_lb = 500
        m15 = _fetch_klines(rest, symbol, "15m", day_start - m15_lb * _interval_ms("15m"), day_end)
        m5 = _fetch_klines(rest, symbol, "5m", day_start - m5_lb * _interval_ms("5m"), day_end)
        
        if len(m15) < 120 or len(m5) < 80:
            print(f"数据不足: m15={len(m15)} m5={len(m5)}")
            return 0
            
        data_map["15m"] = m15
        data_map["5m"] = m5
        
        driver_interval = "5m"
        driver_df = m5
        strategy = SqueezeBreakoutLongStrategy()
        
    else:
        print(f"Unknown strategy: {strategy_name}")
        return 0

    driver_day = driver_df[(driver_df["open_time"] >= day_start) & (driver_df["open_time"] < day_end)].reset_index(drop=True)
    if driver_day.empty:
        print(f"当日无 {driver_interval} K 线数据。")
        return 0

    signals: list[Dict[str, object]] = []
    last_key: tuple[int, float] | None = None
    
    # Pre-compute close times for faster searching
    close_times = {k: v["close_time"].to_numpy() for k, v in data_map.items()}

    for i in range(len(driver_day)):
        cur_close_time = int(driver_day["close_time"].iloc[i])
        
        # Build snapshot for current time
        snapshot = {}
        for k, df in data_map.items():
            ct_arr = close_times[k]
            # Find index where close_time <= cur_close_time
            # searchsorted returns insertion point. 
            # If we want close_time <= cur, we find right insertion and slice.
            end_idx = int(ct_arr.searchsorted(cur_close_time, side="right"))
            # Squeeze strategy needs ~500 bars history for 15m and ~400 for 5m
            # Pullback needs ~200
            limit = 600
            snapshot[k] = df.iloc[max(0, end_idx - limit) : end_idx].reset_index(drop=True)
            
        mock_ds = MockDataSource(snapshot)
        
        # Pullback strategy has explicit analyze_frames, but BaseStrategy defines analyze.
        # We can use analyze for both if we pass mock_ds.
        # However, PullbackShortStrategy.analyze calls analyze_frames using ds.get_klines_df.
        # MockDataSource handles get_klines_df, so it should work!
        
        sig = strategy.analyze(symbol, mock_ds, now_ms=cur_close_time)
        
        if not sig:
            continue
            continue
        key = (int(sig["time"]), float(sig["level"]))
        if last_key == key:
            continue
        last_key = key
        if not (day_start <= int(sig["time"]) < day_end):
            continue
        signals.append(sig)

    def _fmt_ts(ms: int) -> str:
        # 将时间戳转换为UTC时间，然后添加8小时转换为北京时间
        utc_dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        beijing_dt = utc_dt + timedelta(hours=8)
        return beijing_dt.strftime("%Y-%m-%d %H:%M")

    print(f"发现信号数: {len(signals)}", flush=True)
    for s in signals[:50]:
        print(f'- time={_fmt_ts(int(s["time"]))} {fmt_signal(s)}')

    if not signals:
        return 0

    # m15_after = m15[m15["open_time"] >= day_start].reset_index(drop=True)
    # For Squeeze strategy, we might need 5m or 15m for outcome check.
    # But usually 15m is fine for high level check, or we use the driver interval.
    # Let's use the driver_df for outcome check to be consistent.
    driver_after = driver_df[driver_df["open_time"] >= day_start].reset_index(drop=True)
    
    outcomes = []
    for s in signals:
        entry = float(s["entry"])
        stop = float(s["stop"])
        t0 = int(s["time"])
        
        # Check if it's a squeeze strategy signal with tp1/tp2 or pullback with target
        if "tp1" in s and "tp2" in s:
            target = float(s["tp1"])  # Use tp1 as target for outcome check
        elif "target" in s:
            target = float(s["target"])
        else:
            print(f"Warning: No target found for signal {s}")
            continue
        
        # Determine direction
        is_long = target > entry
        
        df_after = driver_after[driver_after["close_time"] >= t0].reset_index(drop=True)
        entered = False
        entry_time = None
        result = "no_entry"
        exit_time = None
        
        # For SqueezeBreakout (Market Entry), we assume immediate entry at next open or same close?
        # The signal is at close_time. So we enter at the NEXT candle's open or during next candle.
        # Simplification: Assume entry if price touches entry level (Limit/Stop) or immediate if Market.
        # SqueezeBreakoutLongStrategy uses "entry = c", so it's a market entry at signal time.
        # PullbackShort uses limit entry.
        
        mode = s.get("mode", "")
        is_market_entry = mode in ["breakout_chase", "retest_entry"]
        
        for i, row in df_after.iterrows():
            hi = float(row["high"])
            lo = float(row["low"])
            ct = int(row["close_time"])
            
            if not entered:
                if is_market_entry:
                    entered = True
                    entry_time = ct
                else:
                    # Limit/Stop logic
                    if is_long:
                        # Buy Limit: hi >= entry (wait, buy limit means price drops to entry. buy stop means price rises to entry)
                        # Usually we assume Limit Entry for pullbacks?
                        # If pullback long, we wait for dip. lo <= entry.
                        # If breakout long, we wait for rise. hi >= entry.
                        # PullbackShort is "Short". We sell high. hi >= entry.
                        if not is_long: # Short
                            if hi >= entry:
                                entered = True
                                entry_time = ct
                        else: # Long
                            if lo <= entry: # Pullback Buy
                                entered = True
                                entry_time = ct
            
            if entered:
                if is_long:
                    if lo <= stop:
                        result = "stop"
                        exit_time = ct
                        break
                    if hi >= target:
                        result = "target"
                        exit_time = ct
                        break
                else: # Short
                    if hi >= stop:
                        result = "stop"
                        exit_time = ct
                        break
                    if lo <= target:
                        result = "target"
                        exit_time = ct
                        break
                        
        outcomes.append({"time": t0, "result": result, "entry_time": entry_time, "exit_time": exit_time})

    cnt_target = sum(1 for o in outcomes if o["result"] == "target")
    cnt_stop = sum(1 for o in outcomes if o["result"] == "stop")
    cnt_no = sum(1 for o in outcomes if o["result"] == "no_entry")
    print(f"结果统计: target={cnt_target} stop={cnt_stop} no_entry={cnt_no}")
    for o in outcomes[:50]:
        t = _fmt_ts(int(o["time"]))
        et = _fmt_ts(int(o["entry_time"])) if o["entry_time"] else "-"
        xt = _fmt_ts(int(o["exit_time"])) if o["exit_time"] else "-"
        print(f'- signal={t} entry={et} exit={xt} result={o["result"]}')
    return len(signals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--strategy", type=str, default="pullback", help="pullback | squeeze")
    args = parser.parse_args()

    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backtest_day(args.symbol.upper(), date_str, args.strategy)


if __name__ == "__main__":
    main()
