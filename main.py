import time
import json
from datetime import datetime
from typing import Dict, List
from core.datasource import BinanceDataSource
from core.strategy import PullbackShortStrategy, SqueezeBreakoutLongStrategy
from core.scanner import MarketScanner
from config import Config

def fmt_signal(sig: Dict[str, object]) -> str:
    strat = sig.get("strategy", "")
    prefix = f'[{strat}] ' if strat else ""
    sym = sig.get("symbol")
    rr = sig.get("rr")
    score = sig.get("score")
    mode = sig.get("mode")
    base = f"{prefix}{sym} rr={rr} score={score}"
    if mode:
        base = f"{base} mode={mode}"
    if "target" in sig:
        return (
            f"{base} level={round(float(sig['level']), 6)} entry={round(float(sig['entry']), 6)} "
            f"stop={round(float(sig['stop']), 6)} target={round(float(sig['target']), 6)}"
        )
    if "tp1" in sig and "tp2" in sig:
        extra = ""
        if "trail_stop_15m" in sig:
            extra = f" trail_stop_15m={round(float(sig['trail_stop_15m']), 6)}"
        return (
            f"{base} level={round(float(sig['level']), 6)} entry={round(float(sig['entry']), 6)} "
            f"stop={round(float(sig['stop']), 6)} tp1={round(float(sig['tp1']), 6)} tp2={round(float(sig['tp2']), 6)}{extra}"
        )
    return base

def save_signals_to_file(signals: List[Dict[str, object]], file_path: str = None):
    """将信号结果保存到文件
    
    Args:
        signals: 信号列表
        file_path: 输出文件路径，默认保存到signals目录下
    """
    import os
    
    if not signals:
        return
    
    # 默认文件路径
    if not file_path:
        # 创建signals目录（如果不存在）
        os.makedirs("signals", exist_ok=True)
        # 使用当前日期和时间命名文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"signals/signals_{timestamp}.json"
    
    # 添加时间戳到每个信号
    for signal in signals:
        if "time" not in signal:
            signal["time"] = int(time.time() * 1000)
        signal["time_readable"] = datetime.fromtimestamp(signal["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存为JSON文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)
    
    print(f"\n信号结果已保存到: {file_path}")

def main():
    print("Initializing BinanceDataSource...")
    ds = BinanceDataSource(top_n=Config.TOP_N, min_quote_volume=Config.MIN_QUOTE_VOLUME)
    
    print("Initializing PullbackShortStrategy...")
    strategies = [SqueezeBreakoutLongStrategy()]
    
    print("Initializing MarketScanner...")
    scanner = MarketScanner(ds, strategies)
    
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
                # 保存信号到文件
                save_signals_to_file(res)
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
