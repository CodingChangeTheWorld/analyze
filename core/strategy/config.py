
class StrategyConfig:
    SQUEEZE_BREAKOUT_LONG = {
        "k1": 0.4,
        "k2": 1.8,
        "k3": 2.5,
        "close_pos_min": 0.65,
        "min_conditions": 3,
        "stand_m": 3,
        "stand_low_atr": 0.5,
        "ksl": 1.0,
        "tp1_r": 1.2,
        "tp2_r": 3.0,
        "trail_m": 3.0,
        "vol_ma_period": 48,
        "atr_period": 14,
        "box_lookback": 20, # used for highest_close_15m
    }

    PULLBACK_SHORT = {
        "price_tolerance_pct": 0.02,
        "wick_ratio_min": 0.4,
        "atr_small_factor": 0.1,
        "atr_large_factor": 0.5,
        "entry_offset_factor": 0.0005,
        "stop_offset_factor": 0.0015,
        "target_risk_reward": 2.0,
        "trend_ema_fast": 20,
        "trend_ema_slow": 50,
        "rsi_period": 14,
        "rsi_threshold": 50,
    }
