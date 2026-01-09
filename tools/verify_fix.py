
import pandas as pd
import time
from typing import Dict, Optional, List
from core.datasource import BinanceDataSource
from core.strategy.squeeze_breakout_long import SqueezeBreakoutLongStrategy

class MockDataSource(BinanceDataSource):
    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self.data_map = data_map

    def get_klines_df(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        df = self.data_map.get(interval)
        if df is None or df.empty:
            return None
        # Return all data; the strategy's analyze method uses 'now_ms' to slice/index correctly
        # But wait, analyze calls _last_closed_idx which looks at the end of the DF or relative to now_ms.
        # Squeeze strategy logic:
        # idx = self._last_closed_idx(m5, now_ms)
        # _last_closed_idx checks if last_ct <= now_ms - 2000.
        # So passing the full historical DF is fine as long as it covers the target time.
        return df

def run_verification():
    ds_real = BinanceDataSource()
    symbol = "ZKPUSDT"
    
    # Time: 2026-01-07 10:00 UTC to 11:00 UTC
    # Need enough history for 120 candles of 15m (~30h). Let's fetch 3 days.
    # Timestamps in ms
    start_ts = int(pd.Timestamp("2026-01-04 00:00:00", tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp("2026-01-07 12:00:00", tz="UTC").timestamp() * 1000)
    
    print(f"Fetching data for {symbol}...")
    # Fetch 15m and 5m data
    k15m = ds_real.rest.get_klines_raw(symbol, "15m", limit=1500, start_time=start_ts, end_time=end_ts)
    k5m = ds_real.rest.get_klines_raw(symbol, "5m", limit=1500, start_time=start_ts, end_time=end_ts)
    
    df15 = k15m
    df5 = k5m
    
    print(f"Data fetched: 15m={len(df15)}, 5m={len(df5)}", flush=True)

    for df in [df15, df5]:
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["close_time"] = df["close_time"].astype("int64")
    
    mock_ds = MockDataSource({
        "15m": df15,
        "5m": df5
    })
    
    strategy = SqueezeBreakoutLongStrategy()
    
    # Check minute by minute from 10:30 to 11:00 UTC
    check_start = int(pd.Timestamp("2026-01-07 10:30:00", tz="UTC").timestamp() * 1000)
    check_end = int(pd.Timestamp("2026-01-07 11:00:00", tz="UTC").timestamp() * 1000)
    
    print("\n--- Starting Verification Loop ---")
    current_time = check_start
    found_signal = False
    
    while current_time <= check_end:
        t_str = pd.Timestamp(current_time, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        
        # We simulate 'now' being current_time. 
        # Strategy expects data up to this point. 
        # Our MockDataSource returns ALL data.
        # Strategy's _last_closed_idx will find the candle closed before 'now_ms'.
        
        # NOTE: SqueezeBreakoutLongStrategy._squeeze_box_15m uses -2 index.
        # If we pass the WHOLE dataframe (which goes up to 12:00), -2 will be 11:45.
        # This is WRONG for historical simulation if we don't slice the DF in MockDataSource.
        
        # Correct approach: Update MockDataSource to slice based on current_time?
        # But analyze() accepts now_ms. The strategy logic:
        # m15 = ds.get_klines_df(...)
        # squeeze = self._squeeze_box_15m(m15) -> uses check_idx = -2
        # So m15 MUST end around 'now_ms'.
        
        # So I need to update MockDataSource to simulate 'limit' or just slice manually before passing.
        # But analyze() calls get_klines_df internally.
        # So I should instantiate a new MockDataSource or update it inside the loop?
        # Or better: make MockDataSource aware of 'now' or just slice in get_klines_df?
        # get_klines_df doesn't take 'now'.
        
        # Strategy logic limitation: It assumes get_klines_df returns recent data relative to 'now'.
        # Since I cannot change strategy's call to get_klines_df to pass 'now',
        # I must handle this.
        
        # Actually, in backtest.py I did:
        # df.iloc[max(0, len(df) - limit) :]
        # This just takes the LAST 'limit' candles.
        
        # If I want to test history, I need to slice the DF *before* passing it to the strategy
        # or use a smarter MockDataSource.
        # Let's use a smarter MockDataSource that we can update.
        
        pass 
        
        # Update mock data to only include data up to current_time
        df15_slice = df15[df15["close_time"] <= current_time]
        df5_slice = df5[df5["close_time"] <= current_time]
        
        mock_ds.data_map["15m"] = df15_slice
        mock_ds.data_map["5m"] = df5_slice
        
        if current_time == check_start:
             print("DEBUG DF15 TAIL:")
             print(df15_slice.tail())
             print("DEBUG DF5 TAIL:")
             print(df5_slice.tail())
        
        res = strategy.analyze(symbol, mock_ds, now_ms=current_time)
        
        if res:
            print(f"[{t_str}] SIGNAL FOUND!")
            print(res)
            found_signal = True
        # else:
            # print(f"[{t_str}] No signal")
            
        current_time += 60 * 1000 # Advance 1 minute

    if not found_signal:
        print("\nNo signal found in the specified window.")
    else:
        print("\nVerification Successful: Signal detected.")

if __name__ == "__main__":
    run_verification()
