import os
import json
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd
from config import Config

class RESTClient:
    def http_get_json(self, path: str, params: Dict[str, str]) -> object:
        qs = urlencode(params)
        url = f"{Config.BASE_URL}{path}?{qs}" if qs else f"{Config.BASE_URL}{path}"
        with urlopen(url, timeout=Config.HTTP_TIMEOUT) as r:
            return json.loads(r.read().decode())
    def get_exchange_symbols(self) -> List[str]:
        info = self.http_get_json("/fapi/v1/exchangeInfo", {})
        syms = []
        for s in info.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT":
                syms.append(s.get("symbol"))
        return syms
    def get_24h_tickers_df(self) -> pd.DataFrame:
        rows = self.http_get_json("/fapi/v1/ticker/24hr", {})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["quoteVolume"] = pd.to_numeric(df.get("quoteVolume", 0), errors="coerce").fillna(0.0)
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
        df["lastPrice"] = pd.to_numeric(df.get("lastPrice", 0), errors="coerce").fillna(0.0)
        return df[["symbol", "quoteVolume", "volume", "lastPrice"]]
    def get_klines_raw(self, symbol: str, interval: str, limit: int = 500, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        params: Dict[str, str] = {"symbol": symbol, "interval": interval, "limit": str(limit)}
        if start_time is not None:
            params["startTime"] = str(int(start_time))
        if end_time is not None:
            params["endTime"] = str(int(end_time))
        rows = self.http_get_json("/fapi/v1/klines", params)
        cols = ["open_time","open","high","low","close","volume","close_time","qv","trades","tb_base","tb_quote","ignore"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
        return df[["open_time","open","high","low","close","volume","close_time"]]

class KlineStore:
    def __init__(self, snapshot_dir: str = Config.DATA_DIR):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.cache_1m: Dict[str, pd.DataFrame] = {}
    def snapshot_path(self, symbol: str) -> str:
        return os.path.join(self.snapshot_dir, f"{symbol}_1m.csv")
    def load_snapshot_1m(self, symbol: str) -> pd.DataFrame:
        path = self.snapshot_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
        df = pd.read_csv(path)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
        return df
    def save_snapshot_1m(self, symbol: str, df: pd.DataFrame):
        path = self.snapshot_path(symbol)
        limit = Config.SNAPSHOT_RETENTION_MINUTES
        if len(df) > limit:
            df = df.iloc[-limit:]
        df.to_csv(path, index=False)
    def backfill_1m(self, rest: RESTClient, symbol: str, max_rounds: int = Config.BACKFILL_MAX_ROUNDS):
        df = self.cache_1m.get(symbol)
        if df is None or df.empty:
            df = self.load_snapshot_1m(symbol)
        
        # Check data integrity (start time)
        now_ms = int(time.time() * 1000)
        retention_ms = Config.SNAPSHOT_RETENTION_MINUTES * 60 * 1000
        target_start = now_ms - retention_ms
        
        need_reset = False
        if df.empty:
            need_reset = True
        else:
            first_open = int(df["open_time"].iloc[0])
            # If data is too fresh (missing history) and we expect more history
            # Allow 1 hour tolerance (3600_000 ms) to avoid unnecessary reset for small gaps
            tolerance_ms = 3600_000
            if first_open > target_start + tolerance_ms:
                need_reset = True
        
        if need_reset:
            df = pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
            start_time = target_start
        else:
            last_close = int(df["close_time"].iloc[-1])
            start_time = last_close + 1
            
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            # Check if we are already up to date
            now_ms = int(time.time() * 1000)
            if start_time >= now_ms - 60_000:
                break

            new_df = rest.get_klines_raw(symbol, "1m", Config.BACKFILL_BATCH_SIZE, start_time=start_time, end_time=None)
            if new_df.empty:
                break
                
            # Enforce types for new_df
            for col in ["open","high","low","close","volume"]:
                new_df[col] = new_df[col].astype(float)
            new_df["open_time"] = new_df["open_time"].astype("int64")
            new_df["close_time"] = new_df["close_time"].astype("int64")

            # Enforce types for df if needed (e.g. if it was initialized empty)
            if df.empty:
                 for col in ["open","high","low","close","volume"]:
                    df[col] = df[col].astype(float)
                 df["open_time"] = df["open_time"].astype("int64")
                 df["close_time"] = df["close_time"].astype("int64")

            df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
            
            last_close = int(df["close_time"].iloc[-1])
            start_time = last_close + 1
            
            time.sleep(Config.BACKFILL_SLEEP_SEC)
        
        limit = Config.SNAPSHOT_RETENTION_MINUTES
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
            
        self.cache_1m[symbol] = df
        self.save_snapshot_1m(symbol, df)
    def resample_ohlc(self, df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
        if df_1m.empty:
            return df_1m
        ts = pd.to_datetime(df_1m["open_time"], unit="ms")
        df_1m = df_1m.copy()
        df_1m["ts"] = ts
        df_1m = df_1m.set_index("ts")
        rule = {"15m":"15min","1h":"1h","4h":"4h"}.get(interval, None)
        if rule is None:
            return df_1m.reset_index(drop=True)[["open_time","open","high","low","close","volume","close_time"]]
        agg = {
            "open_time": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last"
        }
        res = df_1m.resample(rule).agg(agg).dropna()
        res = res.reset_index(drop=True)
        return res[["open_time","open","high","low","close","volume","close_time"]]
    def upsert_ws_kline_1m(self, symbol: str, k: Dict[str, object]):
        t = int(k.get("t"))
        o = float(k.get("o"))
        h = float(k.get("h"))
        l = float(k.get("l"))
        c = float(k.get("c"))
        v = float(k.get("v"))
        T = int(k.get("T"))
        df = self.cache_1m.get(symbol)
        if df is None or df.empty:
            df = pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
        mask = df["open_time"] == t
        if mask.any():
            idx = df.index[mask][0]
            df.at[idx, "open"] = o
            df.at[idx, "high"] = h
            df.at[idx, "low"] = l
            df.at[idx, "close"] = c
            df.at[idx, "volume"] = v
            df.at[idx, "close_time"] = T
        else:
            row = {"open_time": t, "open": o, "high": h, "low": l, "close": c, "volume": v, "close_time": T}
            new_row_df = pd.DataFrame([row])
            # Enforce types to match load_snapshot_1m
            for col in ["open","high","low","close","volume"]:
                new_row_df[col] = new_row_df[col].astype(float)
            new_row_df["open_time"] = new_row_df["open_time"].astype("int64")
            new_row_df["close_time"] = new_row_df["close_time"].astype("int64")
            
            if not new_row_df.empty:
                # Ensure df has same types before concat if it's empty/new
                if df.empty:
                     for col in ["open","high","low","close","volume"]:
                        df[col] = df[col].astype(float)
                     df["open_time"] = df["open_time"].astype("int64")
                     df["close_time"] = df["close_time"].astype("int64")

                df = pd.concat([df, new_row_df], ignore_index=True)
                df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        self.cache_1m[symbol] = df
    def get_klines_df(self, rest: RESTClient, symbol: str, interval: str, limit: int = 500, prefer_store: bool = True) -> pd.DataFrame:
        if prefer_store and interval in ("15m","1h","4h"):
            self.backfill_1m(rest, symbol)
            base = self.cache_1m.get(symbol, pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"]))
            res = self.resample_ohlc(base, interval)
            if limit and len(res) > limit:
                res = res.iloc[-limit:].reset_index(drop=True)
            return res
        df = rest.get_klines_raw(symbol, interval, limit)
        return df

class WSClient:
    def __init__(self, store: KlineStore):
        self.store = store
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.symbols: List[str] = []
    async def run(self, symbols: List[str]):
        import websockets
        streams = "/".join([f"{s.lower()}@kline_1m" for s in symbols])
        url = f"{Config.WS_BASE_URL}?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=Config.WS_PING_INTERVAL, ping_timeout=Config.WS_PING_TIMEOUT) as ws:
                    print(f"WS Connected to {len(symbols)} streams.")
                    while self.running:
                        try:
                            msg = await ws.recv()
                            try:
                                data = json.loads(msg)
                            except:
                                continue
                            d = data.get("data") or data
                            k = d.get("k")
                            if not k:
                                continue
                            sym = k.get("s")
                            self.store.upsert_ws_kline_1m(sym, k)
                            if k.get("x"):
                                self.store.save_snapshot_1m(sym, self.store.cache_1m.get(sym, pd.DataFrame()))
                        except websockets.exceptions.ConnectionClosed:
                            print("WS ConnectionClosed, reconnecting...")
                            break
                        except Exception as e:
                            print(f"WS Error: {e}, reconnecting...")
                            break
            except Exception as e:
                print(f"WS Connect Error: {e}, retry in 5s...")
                import asyncio
                await asyncio.sleep(5)
    def start(self, symbols: List[str]):
        if self.running:
            return
        self.symbols = symbols
        def target():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run(symbols))
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.loop:
            try:
                self.loop.stop()
            except:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

class BinanceDataSource:
    def __init__(self, snapshot_dir: str = Config.DATA_DIR, top_n: int = Config.TOP_N, min_quote_volume: float = Config.MIN_QUOTE_VOLUME):
        self.rest = RESTClient()
        self.store = KlineStore(snapshot_dir=snapshot_dir)
        self.ws = WSClient(store=self.store)
        self.top_n = top_n
        self.min_quote_volume = min_quote_volume
        self.top_symbols: List[str] = []
    def refresh_top_symbols(self):
        syms = self.rest.get_exchange_symbols()
        tickers = self.rest.get_24h_tickers_df()
        tickers = tickers[tickers["symbol"].isin(syms)]
        tickers = tickers[tickers["quoteVolume"] >= self.min_quote_volume]
        tickers = tickers.sort_values("quoteVolume", ascending=False).head(self.top_n)
        self.top_symbols = tickers["symbol"].tolist()
    def start_ws_for_top(self):
        if not self.top_symbols:
            self.refresh_top_symbols()
        
        print(f"Backfilling data for {len(self.top_symbols)} symbols (Concurrency: 20)...")
        # Use a larger pool for IO-bound tasks, but be mindful of rate limits
        # Binance weight limit is high enough for this burst
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(self.store.backfill_1m, self.rest, s): s for s in self.top_symbols}
            
            completed_count = 0
            total_count = len(self.top_symbols)
            
            for future in as_completed(future_to_symbol):
                s = future_to_symbol[future]
                try:
                    future.result()
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"Backfill progress: {completed_count}/{total_count}")
                except Exception as e:
                    print(f"Backfill error for {s}: {e}")
        
        print("Backfill completed. Starting WebSocket...")
        self.ws.start(self.top_symbols)
    def stop_ws(self):
        self.ws.stop()
    def get_exchange_symbols(self) -> List[str]:
        return self.rest.get_exchange_symbols()
    def get_24h_tickers_df(self) -> pd.DataFrame:
        return self.rest.get_24h_tickers_df()
    def get_klines_df(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        prefer_store = symbol in self.top_symbols and interval in ("15m","1h","4h")
        return self.store.get_klines_df(self.rest, symbol, interval, limit, prefer_store=prefer_store)
