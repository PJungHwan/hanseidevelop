import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta

def run_strategy_benchmark(csv_5m="BTC_USDT_5m.csv"):
    print("🚀 [상세 영수증 출력] 50배 스캘핑 매매 내역 정밀 분석 엔진 가동...\n")
    df = pd.read_csv(csv_5m, index_col='timestamp', parse_dates=True)
    
    # --- [전략 1] 무거운 지표 단일 연산 (ZLSMA 130, CE 22/3) ---
    len_A = 130
    lag_A = int((len_A - 1) / 2)
    df['ZLSMA_130'] = ta.ema(df['close'] + (df['close'] - df['close'].shift(lag_A)), length=len_A)
    
    atr_A, mult_A = 22, 3
    df['ATR_A'] = ta.atr(df['high'], df['low'], df['close'], length=atr_A)
    df['CE_Long_A'] = df['high'].rolling(atr_A).max() - (df['ATR_A'] * mult_A)
    df['CE_Short_A'] = df['low'].rolling(atr_A).min() + (df['ATR_A'] * mult_A)
    
    buy_A = (df['close'] > df['ZLSMA_130']) & (df['close'] > df['CE_Short_A']) & (df['close'].shift(1) <= df['CE_Short_A'].shift(1))
    sell_A = (df['close'] < df['CE_Long_A']) & (df['close'].shift(1) >= df['CE_Long_A'].shift(1))
    
    entry_prc_A = df['close'].where(buy_A).ffill()
    entry_atr_A = df['ATR_A'].where(buy_A).ffill()
    sl_atr_A = df['close'] < (entry_prc_A - (entry_atr_A * 1.5))
    exit_A = sell_A | sl_atr_A

    target_profit = 0.004 # 0.4% 타겟 익절 (50배 기준 20% 수익)

    pf = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=buy_A,
        exits=exit_A,
        tp_stop=target_profit,
        init_cash=10000,
        fees=0.0004, 
        freq='5m'
    )
    
    if pf.trades.count() > 0:
        returns_1x = pf.trades.returns.values
        returns_50x = returns_1x * 50.0 
        
        # [청산 방어] 단일 매매에서 -100% 이상 손실 시 -100%로 고정
        returns_50x = np.where(returns_50x <= -1.0, -1.0, returns_50x)
        
        # [상세 통계 연산]
        total_trades = len(returns_50x)
        winning_trades = np.sum(returns_50x > 0)
        losing_trades = np.sum(returns_50x <= 0)
        win_rate = (winning_trades / total_trades) * 100
        
        # 매번 10,000달러(고정 시드) 투입 시 발생하는 수익/손실액 계산
        profit_amounts = returns_50x[returns_50x > 0] * 10000
        loss_amounts = returns_50x[returns_50x <= 0] * 10000
        
        total_gross_profit = np.sum(profit_amounts)
        total_gross_loss = np.sum(loss_amounts)
        net_profit = total_gross_profit + total_gross_loss
        
        print(f"📊 [전략 1] 무거운 지표 (ZLSMA 130, CE 22/3) 상세 성과 영수증")
        print("=" * 55)
        print(f"▶ 총 진입 횟수 : {total_trades} 번")
        print(f"▶ 익절(Win)    : {winning_trades} 번 (+20% 도달)")
        print(f"▶ 손절(Loss)   : {losing_trades} 번 (ATR 1.5배 이탈 또는 CE 붕괴)")
        print(f"▶ 최종 승률    : {win_rate:.2f} %")
        print("-" * 55)
        print(f"💰 누적 획득 수익금 : + ${total_gross_profit:,.2f}")
        print(f"💸 누적 발생 손실금 :   ${total_gross_loss:,.2f}")
        print("=" * 55)
        print(f"💵 최종 순 손익금   :   ${net_profit:,.2f}")
        print("   (※ 매 진입 시 초기 시드 $10,000 고정 투입 가정)")
        print("=" * 55)
    else:
        print("매매 내역이 없습니다.")

if __name__ == "__main__":
    run_strategy_benchmark()