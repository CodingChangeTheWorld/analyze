
class StrategyConfig:
    SQUEEZE_BREAKOUT_LONG = {
        "k1": 0.5, # 价格突破atr倍数
        "k2": 2,  # K线波动atr倍数
        "k3": 10,  # 量能倍数
        "close_pos_min": 0.6,
        "min_conditions": 3,
        "atr_multi" : 1.5, # atr multiplier for breakout
        "ksl": 2.0,
        "tp1_r": 1.2,   
        "tp2_r": 3.0,
        "trail_m": 3.0,
        "vol_ma_period": 14,
        "atr_period": 14,
        "box_lookback": 20, # used for highest_close_15m
        "vol_shock_factor": 1.2, # vol_ratio >= k3 * factor
        "squeeze_box_window": 48, # window for rolling quantile
        "squeeze_quantile": 0.3, # quantile for squeeze detection
        "box_n": 32, # box lookback period
        "box_width_min": 0.02, # minimum box width
        "box_width_max": 0.15, # maximum box width
        "kline_15m_limit": 500, # 15m kline limit
        "kline_5m_limit": 400, # 5m kline limit
        "min_15m_candles": 120, # minimum 15m candles required
        "min_5m_candles": 80, # minimum 5m candles required
    }


    TOP_DISTRIBUTION_SHORT = {
        # 数据参数
        "kline_15m_limit": 500,  # 15m K线数量限制
        "min_15m_candles": 128,   # 最小15m K线数量
        
        # 顶部区域识别参数
        "distribution_zone_n": 128,      # 计算顶部区域的K线数量（16~32小时）
        "zone_high_quantile": 0.95,      # 区域上限的分位数
        "zone_low_quantile": 0.05,       # 区域下限的分位数
        "zone_width_min": 0.02,          # 区域宽度最小值（2%）
        "zone_width_max": 0.1,          # 区域宽度最大值
        "touch_min": 3,                  # 触碰上沿的最小次数
        "touch_tolerance_pct": 0.003,    # 触碰上沿的价格容忍度
        "require_atr_decrease": False,    # 是否要求ATR下降
        "atr_decrease_ratio": 0.9,       # ATR下降比例阈值
        "atr_period": 14,                # ATR计算周期
        
        # UTAD（假突破）参数
        "utad_pierce_factor": 0.15,      # 刺破上沿的ATR倍数
        "utad_upper_wick_ratio": 0.50,   # 上影线比例最小值
        "utad_close_pos_max": 0.45,      # 收盘位置最大值
        
        # 入场确认参数
        "entry_break_utad_low_factor": 0.10,  # 跌破UTAD低点的ATR倍数
        "entry_retest_tolerance_pct": 0.005,  # 回抽确认的价格容忍度
        "entry_retest_wick_ratio": 0.4,        # 回抽确认的上影线比例
        "tp_r": 2.0,                           # 目标风险收益比
        "confirm_wait_bars": 6,                # UTAD确认的最大等待K线数（15分钟K线）
        "sl_pad_atr": 0.3,                     # 止损ATR padding倍数，常用0.2~0.4
    }

    TREND_FOLLOWING_LONG = {
        # 数据参数
        "kline_15m_limit": 500,  # 15m K线数量限制
        "min_15m_candles": 50,   # 最小15m K线数量
        
        # EMA参数
        "m15_ema_period": 10,     # 15min EMA周期
        
        # 策略参数
        "consecutive_bars": 3,    # 连续K线数量
        "signal_cooldown_ms": 3600000,  # 信号冷却时间（1小时）
        "tp1_r": 1.5,            # 第一目标风险收益比
        "tp2_r": 3.0,            # 第二目标风险收益比
        
        # 新增配置参数
        "body_ratio_min": 0.2,    # K线实体占比最小值
        "volume_sma_period": 24,   # 成交量SMA周期
    }
