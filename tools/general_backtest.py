import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.backtesting.general_backtester import GeneralBacktester

def main():
    """
    通用回测脚本演示
    使用方式: python tools/general_backtest.py <symbol> <start_date> <end_date> <strategy>
    示例: python tools/general_backtest.py QUSDT 2025-12-30 2026-01-09 squeeze
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="通用回测工具")
    parser.add_argument("symbol", type=str, help="代币名称，如 QUSDT")
    parser.add_argument("start_date", type=str, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("end_date", type=str, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("strategy", type=str, default="squeeze", choices=["squeeze", "pullback", "top_distribution", "trend_following"], help="策略名称")
    
    args = parser.parse_args()
    
    # 创建回测实例
    backtester = GeneralBacktester()
    
    # 执行回测
    results = backtester.backtest_date_range(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_name=args.strategy
    )
    
    # 打印结果
    backtester.print_summary(results)

if __name__ == "__main__":
    main()
