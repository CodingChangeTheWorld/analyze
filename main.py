import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict
from datasource import BinanceDataSource
from datasource import RESTClient
from strategies import PullbackShortStrategy
from scanner import MarketScanner
from config import Config
import pandas as pd

def fmt_signal(sig: Dict[str, object]) -> str:
    return f'{sig["symbol"]} level={round(sig["level"],6)} entry={round(sig["entry"],6)} stop={round(sig["stop"],6)} target={round(sig["target"],6)} rr={sig["rr"]} score={sig["score"]}'

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
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
    df_all = pd.concat(out, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df_all

def backtest_day(symbol: str, date_str: str) -> int:
    rest = RESTClient()
    day_start, day_end = _date_range_utc_ms(date_str)

    print(f"拉取数据: {symbol} {date_str} (UTC)...")
    h4 = rest.get_klines_raw(symbol, "4h", limit=400)
    h1 = rest.get_klines_raw(symbol, "1h", limit=600)
    m15 = rest.get_klines_raw(symbol, "15m", limit=900)

    if len(m15) == 0 or not (int(m15["close_time"].min()) <= day_start <= int(m15["close_time"].max())):
        print(f"当前拉取窗口不覆盖 {date_str}，请改用更近的日期或提高拉取窗口。")
        return 0

    if len(h4) < 60 or len(h1) < 60 or len(m15) < 60:
        print(f"数据不足: h4={len(h4)} h1={len(h1)} m15={len(m15)} (策略要求每档>=60)")
        return 0

    m15_day = m15[(m15["open_time"] >= day_start) & (m15["open_time"] < day_end)].reset_index(drop=True)
    if m15_day.empty:
        print("当日无 15m K 线数据。")
        return 0

    strategy = PullbackShortStrategy()
    signals: list[Dict[str, object]] = []
    last_key: tuple[int, float] | None = None

    m15_ct = m15["close_time"].to_numpy()
    h1_ct = h1["close_time"].to_numpy()
    h4_ct = h4["close_time"].to_numpy()

    for i in range(len(m15_day)):
        cur_close_time = int(m15_day["close_time"].iloc[i])
        m15_end = int(m15_ct.searchsorted(cur_close_time, side="right"))
        h1_end = int(h1_ct.searchsorted(cur_close_time, side="right"))
        h4_end = int(h4_ct.searchsorted(cur_close_time, side="right"))

        m15_slice = m15.iloc[max(0, m15_end - 200):m15_end].reset_index(drop=True)
        h1_slice = h1.iloc[max(0, h1_end - 400):h1_end].reset_index(drop=True)
        h4_slice = h4.iloc[max(0, h4_end - 400):h4_end].reset_index(drop=True)
        sig = strategy.analyze_frames(symbol, h4=h4_slice, h1=h1_slice, m15=m15_slice)
        if not sig:
            continue
        key = (int(sig["time"]), float(sig["level"]))
        if last_key == key:
            continue
        last_key = key
        if not (day_start <= int(sig["time"]) < day_end):
            continue
        signals.append(sig)

    def _fmt_ts(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    print(f"发现信号数: {len(signals)}")
    for s in signals[:50]:
        print(f'- time={_fmt_ts(int(s["time"]))} {fmt_signal(s)}')

    if not signals:
        return 0

    m15_after = m15[m15["open_time"] >= day_start].reset_index(drop=True)
    outcomes = []
    for s in signals:
        entry = float(s["entry"])
        stop = float(s["stop"])
        target = float(s["target"])
        t0 = int(s["time"])
        df_after = m15_after[m15_after["close_time"] >= t0].reset_index(drop=True)
        entered = False
        entry_time = None
        result = "no_entry"
        exit_time = None
        for _, row in df_after.iterrows():
            hi = float(row["high"])
            lo = float(row["low"])
            ct = int(row["close_time"])
            if not entered:
                if lo <= entry:
                    entered = True
                    entry_time = ct
                    if hi >= stop:
                        result = "stop"
                        exit_time = ct
                        break
                    if lo <= target:
                        result = "target"
                        exit_time = ct
                        break
                continue
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
    parser.add_argument("--backtest", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    args = parser.parse_args()

    if args.backtest:
        date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        backtest_day(args.backtest.upper(), date_str)
        return

    print("Initializing BinanceDataSource...")
    ds = BinanceDataSource(top_n=Config.TOP_N, min_quote_volume=Config.MIN_QUOTE_VOLUME)
    
    print("Initializing PullbackShortStrategy...")
    strategy = PullbackShortStrategy()
    
    print("Initializing MarketScanner...")
    scanner = MarketScanner(ds, strategy)
    
    try:
        print(f"Starting background monitoring (Top {Config.TOP_N}, Min Vol {Config.MIN_QUOTE_VOLUME})...")
        scanner.start()
        
        while True:
            print(f"\n[{time.strftime('%H:%M:%S')}] Scanning...")
            start_ts = time.time()
            res = scanner.scan()
            duration = time.time() - start_ts
            
            if res:
                print(f"Found {len(res)} candidates (took {duration:.2f}s):")
                for s in res[:20]:
                    print(fmt_signal(s))
            else:
                print(f"No signals found (took {duration:.2f}s).")
            
            print("Sleeping 60s...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        scanner.stop()

if __name__ == "__main__":
    main()
