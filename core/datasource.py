import os
import json
import time
import threading
import asyncio
import random
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import pandas as pd
from config import Config


class RateLimiter:
    def __init__(self, rps: int):
        self.rps = max(int(rps), 1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self):
        interval = 1.0 / float(self.rps)
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._next_allowed - now
                if wait <= 0:
                    self._next_allowed = now + interval
                    return
            time.sleep(min(wait, 0.2))


class RESTClient:
    def __init__(self):
        self._limiter = RateLimiter(getattr(Config, "REST_RPS", 8))

    def http_get_json(self, path: str, params: Dict[str, str]) -> object:
        qs = urlencode(params)
        url = f"{Config.BASE_URL}{path}?{qs}" if qs else f"{Config.BASE_URL}{path}"
        max_retries = int(getattr(Config, "REST_MAX_RETRIES", 5) or 5)
        backoff_base = float(getattr(Config, "REST_BACKOFF_BASE_SEC", 0.5) or 0.5)
        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries):
            self._limiter.acquire()
            try:
                with urlopen(url, timeout=Config.HTTP_TIMEOUT) as r:
                    return json.loads(r.read().decode())
            except HTTPError as e:
                last_exc = e
                code = getattr(e, "code", None)
                if code in (418, 429) or (code is not None and 500 <= int(code) < 600):
                    sleep_s = backoff_base * (2 ** attempt) + random.random() * 0.1
                    time.sleep(min(sleep_s, 10.0))
                    continue
                raise
            except (URLError, TimeoutError) as e:
                last_exc = e
                sleep_s = backoff_base * (2 ** attempt) + random.random() * 0.1
                time.sleep(min(sleep_s, 10.0))
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("http_get_json failed")

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

    def get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        params: Dict[str, str] = {"symbol": symbol, "interval": interval, "limit": str(limit)}
        if start_time is not None:
            params["startTime"] = str(int(start_time))
        if end_time is not None:
            params["endTime"] = str(int(end_time))
        rows = self.http_get_json("/fapi/v1/klines", params)
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades", "tb_base", "tb_quote", "ignore"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"])
        for c in ["open", "high", "low", "close", "volume", "tb_base", "tb_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
        return df[["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"]]

    def get_open_interest_current(self, symbol: str) -> float:
        row = self.http_get_json("/fapi/v1/openInterest", {"symbol": symbol})
        try:
            return float(row.get("openInterest", 0.0))
        except Exception:
            return 0.0

    def get_open_interest_hist(
        self,
        symbol: str,
        period: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        params: Dict[str, str] = {"symbol": symbol, "period": period, "limit": str(limit)}
        if start_time is not None:
            params["startTime"] = str(int(start_time))
        if end_time is not None:
            params["endTime"] = str(int(end_time))
        rows = self.http_get_json("/futures/data/openInterestHist", params)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "oi"])
        ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
        oi_col = "sumOpenInterest" if "sumOpenInterest" in df.columns else ("openInterest" if "openInterest" in df.columns else None)
        if ts_col is None or oi_col is None:
            return pd.DataFrame(columns=["timestamp", "oi"])
        out = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(df[ts_col], errors="coerce").fillna(0).astype("int64"),
                "oi": pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0).astype(float),
            }
        )
        out = out.sort_values("timestamp").reset_index(drop=True)
        return out


class KlineStore:
    def __init__(self, snapshot_dir: str = Config.DATA_DIR):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.cache_1m: Dict[str, pd.DataFrame] = {}
        self._last_snapshot_close_time_ms: Dict[str, int] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()
        self._resampled_cache: Dict[Tuple[str, str], Tuple[int, pd.DataFrame]] = {}
        self._oi_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
        self._parquet_supported: Optional[bool] = None
        self._parquet_warned = False

    def _lock_for(self, symbol: str) -> threading.RLock:
        with self._locks_lock:
            lock = self._locks.get(symbol)
            if lock is None:
                lock = threading.RLock()
                self._locks[symbol] = lock
            return lock

    def _invalidate_resample_cache(self, symbol: str):
        self._resampled_cache.pop((symbol, "5m"), None)
        self._resampled_cache.pop((symbol, "15m"), None)
        self._resampled_cache.pop((symbol, "1h"), None)
        self._resampled_cache.pop((symbol, "4h"), None)
        self._oi_cache.pop((symbol, "5m"), None)
        self._oi_cache.pop((symbol, "15m"), None)
        self._oi_cache.pop((symbol, "1h"), None)
        self._oi_cache.pop((symbol, "4h"), None)

    def snapshot_paths(self, symbol: str) -> Dict[str, str]:
        return {
            "parquet": os.path.join(self.snapshot_dir, f"{symbol}_1m.parquet"),
            "csv": os.path.join(self.snapshot_dir, f"{symbol}_1m.csv"),
        }

    def parquet_supported(self) -> bool:
        if self._parquet_supported is not None:
            return bool(self._parquet_supported)
        has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        has_fastparquet = importlib.util.find_spec("fastparquet") is not None
        self._parquet_supported = bool(has_pyarrow or has_fastparquet)
        return bool(self._parquet_supported)

    def load_snapshot_1m(self, symbol: str) -> pd.DataFrame:
        paths = self.snapshot_paths(symbol)
        fmt = (getattr(Config, "SNAPSHOT_FORMAT", "csv") or "csv").lower()
        parquet_path = paths["parquet"]
        csv_path = paths["csv"]

        df: pd.DataFrame
        if fmt == "parquet":
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            elif os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"])
        else:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            elif os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            else:
                return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"])

        if "tb_base" not in df.columns:
            df["tb_base"] = 0.0
        if "tb_quote" not in df.columns:
            df["tb_quote"] = 0.0

        for c in ["open", "high", "low", "close", "volume", "tb_base", "tb_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"]
        return df[cols]

    def save_snapshot_1m(self, symbol: str, df: pd.DataFrame):
        paths = self.snapshot_paths(symbol)
        limit = Config.SNAPSHOT_RETENTION_MINUTES
        if len(df) > limit:
            df = df.iloc[-limit:]
        fmt = (getattr(Config, "SNAPSHOT_FORMAT", "csv") or "csv").lower()
        if fmt == "parquet":
            if not self.parquet_supported():
                if not self._parquet_warned:
                    print("Parquet engine not found (pyarrow/fastparquet). Fallback to CSV.")
                    self._parquet_warned = True
                df.to_csv(paths["csv"], index=False)
                return
            try:
                df.to_parquet(paths["parquet"], index=False)
                return
            except Exception:
                df.to_csv(paths["csv"], index=False)
                return
        df.to_csv(paths["csv"], index=False)

    def maybe_save_snapshot_1m(self, symbol: str, close_time_ms: int):
        interval_min = int(getattr(Config, "SNAPSHOT_FLUSH_MINUTES", 10) or 10)
        if interval_min <= 0:
            return
        with self._lock_for(symbol):
            last = self._last_snapshot_close_time_ms.get(symbol)
            if last is not None and close_time_ms - last < interval_min * 60_000:
                return
            df = self.cache_1m.get(symbol)
            if df is None or df.empty:
                return
            self.save_snapshot_1m(symbol, df)
            self._last_snapshot_close_time_ms[symbol] = int(close_time_ms)

    def backfill_1m(self, rest: RESTClient, symbol: str, max_rounds: int = Config.BACKFILL_MAX_ROUNDS):
        with self._lock_for(symbol):
            df = self.cache_1m.get(symbol)
            if df is None or df.empty:
                df = self.load_snapshot_1m(symbol)

        now_ms = int(time.time() * 1000)
        retention_ms = Config.SNAPSHOT_RETENTION_MINUTES * 60 * 1000
        target_start = now_ms - retention_ms

        need_reset = False
        if df.empty:
            need_reset = True
        else:
            first_open = int(df["open_time"].iloc[0])
            tolerance_ms = 3600_000
            if first_open > target_start + tolerance_ms:
                need_reset = True

        if need_reset:
            df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
            start_time = target_start
        else:
            last_close = int(df["close_time"].iloc[-1])
            start_time = last_close + 1

        rounds = 0
        # 累积数据批次以减少concat和去重操作的开销
        accumulated_data = []
        batch_count = 0
        BATCH_ACCUMULATE_LIMIT = 3  # 每累积3批数据处理一次

        while rounds < max_rounds:
            rounds += 1
            now_ms = int(time.time() * 1000)
            if start_time >= now_ms - 60_000:
                break

            batch_size = Config.BACKFILL_BATCH_SIZE
            new_df = rest.get_klines_raw(symbol, "1m", batch_size, start_time=start_time, end_time=None)
            if new_df.empty:
                break

            # 只转换需要的列，减少转换开销
            new_df = new_df.astype({
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
                "open_time": "int64",
                "close_time": "int64"
            })

            accumulated_data.append(new_df)
            batch_count += 1

            # 更新下一次请求的开始时间
            start_time = int(new_df["close_time"].iloc[-1]) + 1

            # 每累积一定数量的批次或者达到最大轮数时，合并数据
            if batch_count >= BATCH_ACCUMULATE_LIMIT or rounds >= max_rounds:
                if accumulated_data:
                    # 一次性合并所有累积的数据
                    merged_df = pd.concat(accumulated_data, ignore_index=True)
                    
                    if df.empty:
                        # 初始化主数据框的列类型
                        df = merged_df.copy()
                    else:
                        # 合并主数据框和累积数据
                        df = pd.concat([df, merged_df], ignore_index=True)
                        # 只做一次去重和排序操作
                        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
                    
                    # 重置累积数据
                    accumulated_data = []
                    batch_count = 0

            time.sleep(Config.BACKFILL_SLEEP_SEC)

        limit = Config.SNAPSHOT_RETENTION_MINUTES
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)

        with self._lock_for(symbol):
            self.cache_1m[symbol] = df
            self._invalidate_resample_cache(symbol)
            self.save_snapshot_1m(symbol, df)

    def resample_ohlc(self, df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
        if df_1m.empty:
            return df_1m
        df_1m = df_1m.copy()
        if "tb_base" not in df_1m.columns:
            df_1m["tb_base"] = 0.0
        if "tb_quote" not in df_1m.columns:
            df_1m["tb_quote"] = 0.0
        ts = pd.to_datetime(df_1m["open_time"], unit="ms")
        df_1m["ts"] = ts
        df_1m = df_1m.set_index("ts")
        rule = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}.get(interval, None)
        if rule is None:
            return df_1m.reset_index(drop=True)[["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"]]
        agg = {
            "open_time": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
            "tb_base": "sum",
            "tb_quote": "sum",
        }
        res = df_1m.resample(rule).agg(agg).dropna()
        res = res.reset_index(drop=True)
        return res[["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"]]

    def upsert_ws_kline_1m(self, symbol: str, k: Dict[str, object]):
        t = int(k.get("t"))
        o = float(k.get("o"))
        h = float(k.get("h"))
        l = float(k.get("l"))
        c = float(k.get("c"))
        v = float(k.get("v"))
        T = int(k.get("T"))
        tb_base = float(k.get("V") or 0.0)
        tb_quote = float(k.get("Q") or 0.0)
        with self._lock_for(symbol):
            df = self.cache_1m.get(symbol)
            if df is None or df.empty:
                df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"])
            mask = df["open_time"] == t
            if mask.any():
                idx = df.index[mask][0]
                df.at[idx, "open"] = o
                df.at[idx, "high"] = h
                df.at[idx, "low"] = l
                df.at[idx, "close"] = c
                df.at[idx, "volume"] = v
                df.at[idx, "close_time"] = T
                df.at[idx, "tb_base"] = tb_base
                df.at[idx, "tb_quote"] = tb_quote
            else:
                row = {
                    "open_time": t,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "close_time": T,
                    "tb_base": tb_base,
                    "tb_quote": tb_quote,
                }
                new_row_df = pd.DataFrame([row])
                for col in ["open", "high", "low", "close", "volume", "tb_base", "tb_quote"]:
                    new_row_df[col] = new_row_df[col].astype(float)
                new_row_df["open_time"] = new_row_df["open_time"].astype("int64")
                new_row_df["close_time"] = new_row_df["close_time"].astype("int64")

                if not new_row_df.empty:
                    if df.empty:
                        for col in ["open", "high", "low", "close", "volume", "tb_base", "tb_quote"]:
                            df[col] = df[col].astype(float)
                        df["open_time"] = df["open_time"].astype("int64")
                        df["close_time"] = df["close_time"].astype("int64")

                    df = pd.concat([df, new_row_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
            self.cache_1m[symbol] = df
            self._invalidate_resample_cache(symbol)

    def get_klines_df(self, rest: RESTClient, symbol: str, interval: str, limit: int = 500, prefer_store: bool = True) -> pd.DataFrame:
        if prefer_store and interval in ("5m", "15m", "1h", "4h"):
            self.backfill_1m(rest, symbol)
            with self._lock_for(symbol):
                base = self.cache_1m.get(symbol, pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "tb_base", "tb_quote"]))
                base_last_close = int(base["close_time"].iloc[-1]) if not base.empty else 0
                cached = self._resampled_cache.get((symbol, interval))
                if cached and cached[0] == base_last_close:
                    res = cached[1]
                else:
                    res = self.resample_ohlc(base, interval)
                    self._resampled_cache[(symbol, interval)] = (base_last_close, res)
            if limit and len(res) > limit:
                res = res.iloc[-limit:].reset_index(drop=True)
            out = self._enrich_cvd(res)
            return self._enrich_oi(rest, symbol, interval, out)
        df = rest.get_klines_raw(symbol, interval, limit)
        out = self._enrich_cvd(df)
        return self._enrich_oi(rest, symbol, interval, out)

    def _enrich_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "tb_base" not in df.columns:
            df = df.copy()
            df["tb_base"] = 0.0
        if "volume" not in df.columns:
            return df
        out = df.copy()
        out["delta"] = (2.0 * pd.to_numeric(out["tb_base"], errors="coerce").fillna(0.0)) - pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
        out["cvd"] = out["delta"].cumsum()
        return out

    def _enrich_oi(self, rest: RESTClient, symbol: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not bool(getattr(Config, "OI_ENRICH_ENABLED", False)):
            return df
        if interval not in ("15m", "1h", "4h"):
            return df
        cache_key = (symbol, interval)
        now_mono = time.monotonic()
        with self._lock_for(symbol):
            cached = self._oi_cache.get(cache_key)
            if cached and now_mono - cached[0] <= float(getattr(Config, "OI_CACHE_SEC", 30) or 30):
                oi_df = cached[1]
            else:
                try:
                    req_limit = min(int(len(df)), 500)
                    if req_limit < 10:
                        req_limit = 10
                    oi_df = rest.get_open_interest_hist(symbol, interval, limit=req_limit)
                except Exception:
                    oi_df = pd.DataFrame(columns=["timestamp", "oi"])
                self._oi_cache[cache_key] = (now_mono, oi_df)
        if oi_df.empty:
            try:
                oi_now = rest.get_open_interest_current(symbol)
                out = df.copy()
                out["oi"] = float(oi_now)
                return out
            except Exception:
                return df
        left = df[["close_time"]].copy()
        left["close_time"] = pd.to_numeric(left["close_time"], errors="coerce").fillna(0).astype("int64")
        right = oi_df.rename(columns={"timestamp": "close_time"}).copy()
        right["close_time"] = pd.to_numeric(right["close_time"], errors="coerce").fillna(0).astype("int64")
        right = right.sort_values("close_time").reset_index(drop=True)
        left = left.sort_values("close_time").reset_index(drop=True)
        merged = pd.merge_asof(left, right, on="close_time", direction="backward")
        out = df.copy()
        out["oi"] = merged["oi"].to_numpy()
        return out


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
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
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
                                self.store.maybe_save_snapshot_1m(sym, int(k.get("T")))
                        except websockets.exceptions.ConnectionClosed:
                            print("WS ConnectionClosed, reconnecting...")
                            break
                        except Exception as e:
                            print(f"WS Error: {e}, reconnecting...")
                            break
            except Exception as e:
                if not self.running:
                    break
                print(f"WS Connect Error: {e}, retry in 5s...")
                await asyncio.sleep(5)

    def start(self, symbols: List[str]):
        if self.running:
            return
        self.running = True
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
                self.loop.call_soon_threadsafe(lambda: None)
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

        concurrency = int(getattr(Config, "BACKFILL_CONCURRENCY", 12) or 12)
        print(f"Backfilling data for {len(self.top_symbols)} symbols (Concurrency: {concurrency})...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
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
        prefer_store = symbol in self.top_symbols and interval in ("5m", "15m", "1h", "4h")
        return self.store.get_klines_df(self.rest, symbol, interval, limit, prefer_store=prefer_store)
