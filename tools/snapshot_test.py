import time
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.datasource import RESTClient, KlineStore, WSClient
from config import Config


def snapshot_test(symbol: str, seconds: int, rows: int, flush_minutes: int, do_backfill: bool) -> None:
    Config.ensure_dirs()
    if flush_minutes is not None:
        Config.SNAPSHOT_FLUSH_MINUTES = int(flush_minutes)
    store = KlineStore(snapshot_dir=Config.DATA_DIR)
    rest = RESTClient()
    if do_backfill:
        try:
            store.backfill_1m(rest, symbol)
        except Exception as e:
            print(f"Backfill failed: {e}")
    ws = WSClient(store=store)

    fmt = (getattr(Config, "SNAPSHOT_FORMAT", "csv") or "csv").lower()
    parquet_ok = bool(getattr(store, "parquet_supported", lambda: False)())
    print(f"Snapshot format={fmt} parquet_supported={parquet_ok}")
    print(f"Starting WS snapshot test: symbol={symbol} seconds={seconds} flush_min={Config.SNAPSHOT_FLUSH_MINUTES}")
    ws.start([symbol])
    try:
        time.sleep(max(int(seconds), 1))
    finally:
        ws.stop()

    paths = store.snapshot_paths(symbol)
    parquet_path = paths.get("parquet")
    csv_path = paths.get("csv")
    saved_path = parquet_path if parquet_path and os.path.exists(parquet_path) else (csv_path if csv_path and os.path.exists(csv_path) else "")
    if saved_path:
        print(f"Snapshot file: {saved_path}")
    else:
        print("Snapshot file not found.")

    df = store.load_snapshot_1m(symbol)
    if df.empty:
        print("Snapshot is empty.")
        return
    out = df.copy()
    out["delta"] = (2.0 * pd.to_numeric(out.get("tb_base", 0.0), errors="coerce").fillna(0.0)) - pd.to_numeric(out.get("volume", 0.0), errors="coerce").fillna(0.0)
    out["cvd"] = out["delta"].cumsum()
    out["open_ts"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["close_ts"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)
    cols = ["open_ts", "close_ts", "open", "high", "low", "close", "volume", "tb_base", "tb_quote", "delta", "cvd"]
    cols = [c for c in cols if c in out.columns]
    print(f"Snapshot rows: {len(out)} cols={list(out.columns)}")
    n = max(int(rows), 1)
    print(out[cols].tail(n).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--seconds", type=int, default=130)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--flush-minutes", type=int, default=1)
    parser.add_argument("--no-backfill", action="store_true")
    args = parser.parse_args()

    snapshot_test(
        symbol=args.symbol.upper(),
        seconds=args.seconds,
        rows=args.rows,
        flush_minutes=args.flush_minutes,
        do_backfill=not args.no_backfill,
    )


if __name__ == "__main__":
    main()
