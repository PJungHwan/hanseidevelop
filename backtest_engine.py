import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_optimization_backtest(csv_1h="BTC_USDT_1h.csv", csv_4h="BTC_USDT_4h.csv"):
    """
    동적 레버리지와 충돌하지 않도록 반복문(Loop) 기반의 Grid Search를 수행하는 최적화 엔진입니다.
    """
    print("🚀 [Step 7] 컴퓨터 연산력을 활용한 트레일링 스탑 파라미터 최적화(Grid Search) 가동 중...\n")
    
    # 1. 멀티 타임프레임 연동 및 변동성 지표 추가
    df_1h = add_indicators(csv_1h)
    df_4h = add_indicators(csv_4h)
    
    df_4h['Trend_4h_Bullish'] = df_4h['close'] > df_4h['SMA_20']
    df_1h['Trend_4h_Bullish'] = df_4h['Trend_4h_Bullish'].reindex(df_1h.index, method='ffill')
    
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    df_1h.dropna(inplace=True) 
    
    # 2. 매수/매도 시그널
    entries = (
        (df_1h['close'] > df_1h['AVWAP_Weekly']) & 
        (df_1h['close'].shift(1) <= df_1h['AVWAP_Weekly'].shift(1)) & 
        (df_1h['Trend_4h_Bullish'] == True)
    )
    exits = df_1h['RSI_14'] >= 70
    
    # 3. 실시간 동적 레버리지 연산
    atr_pct = df_1h['ATR'] / df_1h['close']
    dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0)
    
    # 4. 테스트할 트레일링 스탑 후보군 배열 생성 (1% ~ 10%)
    sl_stops = np.arange(0.01, 0.11, 0.01)
    print(f"🔍 테스트할 트레일링 스탑 후보군: {np.round(sl_stops * 100)} (%)\n")
    
    # 5. [충돌 해결] 반복문(for loop)을 활용한 안전한 시뮬레이션
    results = []
    
    for sl in sl_stops:
        pf = vbt.Portfolio.from_signals(
            close=df_1h['close'],
            entries=entries,
            exits=exits,
            size=dynamic_leverage,
            size_type='percent',
            sl_stop=sl,         # 10개의 파라미터를 하나씩 순차적으로 투입
            sl_trail=True,      
            tp_stop=0.20,       
            init_cash=10000,
            fees=0.0005,
            freq='1h'
        )
        # 해당 파라미터의 수익률을 추출하여 리스트에 저장
        returns = pf.total_return() * 100
        results.append({'Trailing_Stop': f"{sl*100:.1f}%", 'Total_Return(%)': returns})
        
    # 6. 성과 랭킹표 데이터프레임 변환 및 정렬
    res_df = pd.DataFrame(results)
    best_res = res_df.sort_values(by='Total_Return(%)', ascending=False)
    
    print("🏆 [파라미터 최적화(Grid Search) 결과 랭킹]")
    print(best_res.to_string(index=False))
    
    # 1등 결과 추출
    best_param = best_res.iloc[0]['Trailing_Stop']
    best_return = best_res.iloc[0]['Total_Return(%)']
    print(f"\n💡 결론: 비트코인 1시간봉 기준 최적의 트레일링 스탑은 [{best_param}] 이며, 최고 수익률은 [{best_return:.2f}%] 입니다.")
    
    return best_res

if __name__ == "__main__":
    run_optimization_backtest()