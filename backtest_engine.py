import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_advanced_backtest(csv_1h="BTC_USDT_1h.csv", csv_4h="BTC_USDT_4h.csv"):
    """
    1시간봉 시그널과 4시간봉 추세를 융합한 멀티 타임프레임(MTF) 백테스트 엔진입니다.
    """
    print(f"🚀 멀티 타임프레임(1h + 4h) 융합 엔진 구동 중...\n")
    
    # 1. 두 개의 타임프레임 데이터 각각 불러오기
    df_1h = add_indicators(csv_1h)
    df_4h = add_indicators(csv_4h)
    
    # 2. [핵심] 4시간봉 거시적 추세 판별 (4시간봉 종가가 20 이평선 위에 있으면 상승장)
    df_4h['Trend_4h_Bullish'] = df_4h['close'] > df_4h['SMA_20']
    
    # 3. 미래 참조(Lookahead Bias) 오류를 방지하며 4시간봉 결과를 1시간봉 시간에 맞춰 병합
    # ffill(Forward Fill): 새로운 4시간봉 캔들이 완성되기 전까지는 이전 상태를 유지
    df_1h['Trend_4h_Bullish'] = df_4h['Trend_4h_Bullish'].reindex(df_1h.index, method='ffill')
    
    # 4. 변동성 지표(ATR) 추가
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    df_1h.dropna(inplace=True) 
    
    # 5. [진입 조건 심화] 1시간봉 AVWAP 상향 돌파 AND 4시간봉 전체 추세가 상승장일 때만 매수
    entries = (
        (df_1h['close'] > df_1h['AVWAP_Weekly']) & 
        (df_1h['close'].shift(1) <= df_1h['AVWAP_Weekly'].shift(1)) & 
        (df_1h['Trend_4h_Bullish'] == True)
    )
    
    # [매도 조건] RSI 70 이상 과매수 구간 진입 시
    exits = df_1h['RSI_14'] >= 70
    
    # 6. 동적 레버리지 계산 (이전 단계 유지)
    atr_pct = df_1h['ATR'] / df_1h['close']
    dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0)
    
    # 7. Vectorbt 포트폴리오 가동
    pf = vbt.Portfolio.from_signals(
        close=df_1h['close'],
        entries=entries,
        exits=exits,
        size=dynamic_leverage,
        size_type='percent',
        sl_stop=0.05,  # 5% 고정 손절 (다음 16단계에서 트레일링 스탑으로 교체 예정)
        tp_stop=0.15,
        init_cash=10000,
        fees=0.0005,
        freq='1h'
    )
    
    print("\n📊 [15단계 MTF 최적화 성과 리포트]")
    print(pf.stats())
    
    return pf

if __name__ == "__main__":
    run_advanced_backtest()