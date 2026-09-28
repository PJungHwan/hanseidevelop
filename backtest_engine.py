import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_sma_optimization(csv_1h="BTC_USDT_1h.csv", csv_4h="BTC_USDT_4h.csv"):
    """
    트레이딩뷰 커스텀 지표처럼 예리한 타점을 잡기 위해 
    4시간봉의 최적 추세선(SMA) 길이를 찾아내는 Grid Search 엔진입니다.
    """
    print("🚀 [Step 8] 거시 추세 필터(4시간봉 이평선) 최적화 가동 중...\n")
    
    # 1. 멀티 타임프레임 데이터 로드
    df_1h = add_indicators(csv_1h)
    df_4h = add_indicators(csv_4h)
    
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    df_1h.dropna(inplace=True) 
    
    # [방패 고정] 7단계에서 증명된 최적의 트레일링 스탑 적용
    optimal_sl = 0.05 
    
    # [창끝 튜닝] 테스트할 4시간봉 이평선(SMA) 후보군 (10 ~ 60)
    sma_periods = range(10, 61, 10)
    print(f"🔍 테스트할 4H 이평선(SMA) 후보군: {list(sma_periods)} (기간)\n")
    
    results = []
    
    # 후보군들을 하나씩 엔진에 넣어 교차 검증 시작
    for sma in sma_periods:
        # 1. 4시간봉 SMA 동적 계산 (파라미터 변경)
        df_4h[f'SMA_{sma}'] = ta.sma(df_4h['close'], length=sma)
        df_4h['Trend_4h_Bullish'] = df_4h['close'] > df_4h[f'SMA_{sma}']
        
        # 2. 미래 참조 방지하며 1시간봉에 매핑
        temp_trend = df_4h['Trend_4h_Bullish'].reindex(df_1h.index, method='ffill')
        
        # 3. 진입 조건 (1시간봉 AVWAP 돌파 AND 최적화된 4시간봉 추세 필터)
        entries = (
            (df_1h['close'] > df_1h['AVWAP_Weekly']) & 
            (df_1h['close'].shift(1) <= df_1h['AVWAP_Weekly'].shift(1)) & 
            (temp_trend == True)
        )
        
        # 현재는 RSI 70 과매수 매도 (Step 9에서 익절 타점도 최적화 예정)
        exits = df_1h['RSI_14'] >= 70
        
        # 4. 동적 레버리지
        atr_pct = df_1h['ATR'] / df_1h['close']
        dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0)
        
        # 5. 백테스트 구동
        pf = vbt.Portfolio.from_signals(
            close=df_1h['close'],
            entries=entries,
            exits=exits,
            size=dynamic_leverage,
            size_type='percent',
            sl_stop=optimal_sl,    # 5% 추적 손절
            sl_trail=True,
            tp_stop=0.20,          # 20% 강제 익절
            init_cash=10000,
            fees=0.0005,
            freq='1h'
        )
        
        # 성과 추출
        returns = pf.total_return() * 100
        win_rate = pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0
        
        results.append({
            '4H_SMA_Length': int(sma),
            'Total_Return(%)': returns,
            'Win_Rate(%)': win_rate,
            'Total_Trades': pf.trades.count()
        })
        
    # 6. 성과 랭킹표 출력
    res_df = pd.DataFrame(results)
    best_res = res_df.sort_values(by='Total_Return(%)', ascending=False)
    
    print("🏆 [4시간봉 추세 필터(SMA) 최적화 결과 랭킹]")
    print(best_res.to_string(index=False))
    
    best_sma = best_res.iloc[0]['4H_SMA_Length']
    best_ret = best_res.iloc[0]['Total_Return(%)']
    print(f"\n💡 결론: 비트코인 4시간봉 기준 최적의 추세 이평선은 [SMA {int(best_sma)}] 이며, 최고 수익률은 [{best_ret:.2f}%] 입니다.")
    
    return best_res

if __name__ == "__main__":
    run_sma_optimization()