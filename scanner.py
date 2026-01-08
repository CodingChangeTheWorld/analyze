import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config
from datasource import BinanceDataSource
from strategies import BaseStrategy

class MarketScanner:
    def __init__(self, datasource: BinanceDataSource, strategy: BaseStrategy):
        self.ds = datasource
        self.strategy = strategy

    def start(self):
        """Start data collection and maintain real-time data."""
        print("Refreshing top symbols...")
        self.ds.refresh_top_symbols()
        print(f"Starting WebSocket for {len(self.ds.top_symbols)} symbols...")
        self.ds.start_ws_for_top()
        print(f"Waiting {Config.WS_WARMUP_SEC}s for WS warmup...")
        time.sleep(Config.WS_WARMUP_SEC)

    def stop(self):
        """Stop data collection."""
        self.ds.stop_ws()

    def scan(self) -> List[Dict[str, object]]:
        """Perform a single pass of analysis on currently maintained data."""
        # Use current top symbols from datasource
        tickers_df = self.ds.get_24h_tickers_df()
        
        # Filter again just to be safe and get fresh quote volumes
        syms = set(self.ds.top_symbols)
        tickers_df = tickers_df[tickers_df["symbol"].isin(syms)]
        tickers_df = tickers_df.sort_values("quoteVolume", ascending=False)
        
        symbols = tickers_df["symbol"].tolist()
        out = []
        
        # Parallel Analysis
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(self.strategy.analyze, s, self.ds): s for s in symbols}
            for future in as_completed(future_to_symbol):
                s = future_to_symbol[future]
                try:
                    sig = future.result()
                    if sig:
                        out.append(sig)
                except Exception as e:
                    # print(f"Error processing {s}: {e}")
                    continue
        
        out.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
        return out
