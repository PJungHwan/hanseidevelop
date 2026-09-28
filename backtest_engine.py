import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_rsi_optimization(csv_1h="BTC_USDT_1h.csv", csv_4h="BTC_USDT_4h.csv"):
    """
    데이터 길이 불일치 오류를 수정하고,
    수익률 극대화를 위해 비트코인의 과매수(익절) 타점을 찾아내는 RSI Grid Search 엔진입니다.
    """
    print("🚀 [Step 9] RSI 익절 타점 최적화 가동 중...\n")
    
    # 1. 멀티 타임프레임 데이터 로드
    df_1h = add_indicators(csv_1h)
    df_4h = add_indicators(csv_4h)
    
    # 2. [고정된 최적값] 이전 단계에서 검증된 최상의 방패와 필터 장착
    optimal_sl = 0.05 
    optimal_sma = 30
    
    df_4h[f'SMA_{optimal_sma}'] = ta.sma(df_4h['close'], length=optimal_sma)
    df_4h['Trend_4h_Bullish'] = df_4h['close'] > df_4h[f'SMA_{optimal_sma}']
    
    # [버그 수정] temp_trend 변수 대신 df_1h의 정식 컬럼으로 추가하여 길이 동기화
    df_1h['Trend_4h_Bullish'] = df_4h['Trend_4h_Bullish'].reindex(df_1h.index, method='ffill')
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    
    # 여기서 결측치를 제거하면 Trend_4h_Bullish도 다른 데이터와 똑같이 968개로 맞춰집니다.
    df_1h.dropna(inplace=True) 
    
    # 3. 진입 조건 (df_1h 내부 컬럼으로 통일)
    entries = (
        (df_1h['close'] > df_1h['AVWAP_Weekly']) & 
        (df_1h['close'].shift(1) <= df_1h['AVWAP_Weekly'].shift(1)) & 
        (df_1h['Trend_4h_Bullish'] == True)
    )
    
    # 4. 동적 레버리지
    atr_pct = df_1h['ATR'] / df_1h['close']
    dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0)
    
    # [창끝 튜닝] 테스트할 RSI 익절 후보군 (60 ~ 90)
    rsi_targets = range(60, 91, 5)
    print(f"🔍 테스트할 RSI 익절 후보군: {list(rsi_targets)} (과매수 구간)\n")
    
    results = []
    
    # 후보군들을 하나씩 엔진에 넣어 교차 검증 시작
    for rsi_val in rsi_targets:
        exits = df_1h['RSI_14'] >= rsi_val
        
        pf = vbt.Portfolio.from_signals(
            close=df_1h['close'],
            entries=entries,
            exits=exits,
            size=dynamic_leverage,
            size_type='percent',
            sl_stop=optimal_sl,
            sl_trail=True,
            tp_stop=0.20,
            init_cash=10000,
            fees=0.0005,
            freq='1h'
        )
        
        returns = pf.total_return() * 100
        win_rate = pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0
        
        results.append({
            'RSI_Exit': int(rsi_val),
            'Total_Return(%)': returns,
            'Win_Rate(%)': win_rate,
            'Total_Trades': pf.trades.count()
        })
        
    # 5. 성과 랭킹표 출력
    res_df = pd.DataFrame(results)
    best_res = res_df.sort_values(by='Total_Return(%)', ascending=False)
    
    print("🏆 [RSI 익절 타점 최적화 결과 랭킹]")
    print(best_res.to_string(index=False))
    
    best_rsi = best_res.iloc[0]['RSI_Exit']
    best_ret = best_res.iloc[0]['Total_Return(%)']
    print(f"\n💡 결론: 비트코인 1시간봉 기준 최적의 RSI 익절선은 [{int(best_rsi)}] 이며, 최고 수익률은 [{best_ret:.2f}%] 입니다.")
    
    return best_res

if __name__ == "__main__":
    run_rsi_optimization()