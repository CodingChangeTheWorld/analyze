import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config
from core.datasource import BinanceDataSource
from core.strategy import BaseStrategy


class MarketScanner:
    def __init__(self, datasource: BinanceDataSource, strategies: List[BaseStrategy]):
        self.ds = datasource
        self.strategies = strategies

    def _analyze_symbol(self, symbol: str) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for strat in self.strategies:
            sig = strat.analyze(symbol, self.ds)
            if sig:
                if "strategy" not in sig:
                    sig = dict(sig)
                    sig["strategy"] = strat.__class__.__name__
                out.append(sig)
        return out

    def start(self):
        print("Refreshing top symbols...")
        self.ds.refresh_top_symbols()
        print(f"Starting WebSocket for {len(self.ds.top_symbols)} symbols...")
        self.ds.start_ws_for_top()
        print(f"Waiting {Config.WS_WARMUP_SEC}s for WS warmup...")
        time.sleep(Config.WS_WARMUP_SEC)

    def stop(self):
        self.ds.stop_ws()

    def scan(self) -> List[Dict[str, object]]:
        tickers_df = self.ds.get_24h_tickers_df()
        syms = set(self.ds.top_symbols)
        tickers_df = tickers_df[tickers_df["symbol"].isin(syms)]
        tickers_df = tickers_df.sort_values("quoteVolume", ascending=False)

        symbols = tickers_df["symbol"].tolist()
        out = []

        with ThreadPoolExecutor(max_workers=Config.SCAN_MAX_WORKERS) as executor:
            future_to_symbol: Dict[object, str] = {}
            for s in symbols:
                future_to_symbol[executor.submit(self._analyze_symbol, s)] = s
            for future in as_completed(future_to_symbol):
                try:
                    sigs = future.result()
                    if sigs:
                        out.extend(sigs)
                except Exception:
                    continue

        out.sort(key=lambda x: (x.get("score", 0), x.get("rr", 0)), reverse=True)
        return out

