import os

class Config:
    # API Endpoints
    BASE_URL = "https://fapi.binance.com"
    WS_BASE_URL = "wss://fstream.binance.com/stream"
    
    # Data Storage
    DATA_DIR = "data"
    SNAPSHOT_RETENTION_DAYS = 4
    SNAPSHOT_RETENTION_MINUTES = SNAPSHOT_RETENTION_DAYS * 24 * 60  # 5760
    
    # Scanner Settings
    TOP_N = 300
    MIN_QUOTE_VOLUME = 5_000_000  # 5 Million USDT
    
    # Backfill Settings
    BACKFILL_MAX_ROUNDS = 10
    BACKFILL_BATCH_SIZE = 1000
    BACKFILL_SLEEP_SEC = 0.1
    WS_WARMUP_SEC = 5
    
    # WebSocket Settings
    WS_PING_INTERVAL = 20
    WS_PING_TIMEOUT = 20
    
    # Network Settings
    HTTP_TIMEOUT = 10
    
    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
