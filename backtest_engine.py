import pandas as pd
import numpy as np
import vectorbt as vbt

def get_atr(df, length):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def get_zlsma(series, length):
    lag = int((length - 1) / 2)
    data = series + (series - series.shift(lag))
    return data.ewm(span=length, adjust=False).mean()

def run_strategy_benchmark(csv_1h="BTC_USDT_1h.csv"):
    print("🚀 [1시간봉 격상] 종가 마감 vs 실시간 돌파 승률 테스트 가동...\n")
    df = pd.read_csv(csv_1h, index_col='timestamp', parse_dates=True)
    
    len_A = 130
    df['ZLSMA'] = get_zlsma(df['close'], len_A)
    
    atr_A, mult_A = 22, 3
    df['ATR'] = get_atr(df, atr_A)
    df['CE_Long'] = df['high'].rolling(atr_A).max() - (df['ATR'] * mult_A)
    df['CE_Short'] = df['low'].rolling(atr_A).min() + (df['ATR'] * mult_A)
    
    buy_close = (df['close'] > df['ZLSMA']) & (df['close'] > df['CE_Short']) & (df['close'].shift(1) <= df['CE_Short'].shift(1))
    sell_close = (df['close'] < df['CE_Long']) & (df['close'].shift(1) >= df['CE_Long'].shift(1))
    
    entry_prc_close = df['close'].where(buy_close).ffill()
    entry_atr_close = df['ATR'].where(buy_close).ffill()
    sl_close = df['close'] < (entry_prc_close - (entry_atr_close * 1.5))
    exit_close = sell_close | sl_close

    buy_high = (df['high'] > df['ZLSMA'].shift(1)) & (df['high'] > df['CE_Short'].shift(1)) & (df['close'].shift(1) <= df['CE_Short'].shift(1))
    sell_low = (df['low'] < df['CE_Long'].shift(1)) & (df['close'].shift(1) >= df['CE_Long'].shift(1))
    
    entry_prc_high = df['close'].where(buy_high).ffill() 
    entry_atr_high = df['ATR'].where(buy_high).ffill()
    sl_high = df['close'] < (entry_prc_high - (entry_atr_high * 1.5))
    exit_high = sell_low | sl_high

    target_profit = 0.004 # 0.4% 타겟 익절 (50배 기준 20% 수익)

    strategies = {
        "1. [정석] 1시간봉 종가 마감 확인 후 진입": {'entries': buy_close, 'exits': exit_close},
        "2. [공격] 1시간봉 꼬리 돌파 즉시 진입": {'entries': buy_high, 'exits': exit_high}
    }

    print("📊 10,000개의 1시간봉 캔들에서 매매 타점을 추출합니다...")
    
    for name, params in strategies.items():
        pf = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=params['entries'],
            exits=params['exits'],
            tp_stop=target_profit,
            init_cash=10000,
            fees=0.0004, 
            freq='1h' # 빈도를 1시간으로 변경
        )
        
        if pf.trades.count() > 0:
            returns_1x = pf.trades.returns.values
            returns_50x = returns_1x * 50.0 
            returns_50x = np.where(returns_50x <= -1.0, -1.0, returns_50x)
            
            total_trades = len(returns_50x)
            winning_trades = np.sum(returns_50x > 0)
            losing_trades = np.sum(returns_50x <= 0)
            win_rate = (winning_trades / total_trades) * 100
            
            profit_amounts = returns_50x[returns_50x > 0] * 10000
            loss_amounts = returns_50x[returns_50x <= 0] * 10000
            total_gross_profit = np.sum(profit_amounts)
            total_gross_loss = np.sum(loss_amounts)
            net_profit = total_gross_profit + total_gross_loss
            
            print("\n" + "=" * 55)
            print(f"▶ {name}")
            print("-" * 55)
            print(f"총 진입 횟수 : {total_trades} 번")
            print(f"승리(Win)    : {winning_trades} 번 / 패배(Loss) : {losing_trades} 번")
            print(f"최종 승률    : {win_rate:.2f} %")
            print(f"최종 순 손익 : ${net_profit:,.2f}")
        else:
            print(f"\n▶ {name} : 매매 내역이 없습니다.")

if __name__ == "__main__":
    run_strategy_benchmark()