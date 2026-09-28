import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_advanced_backtest(csv_1h="BTC_USDT_1h.csv", csv_4h="BTC_USDT_4h.csv"):
    """
    MTF 추세 필터와 스마트 트레일링 스탑(수익 보존)이 결합된 최적화 엔진입니다.
    """
    print(f"🚀 [Step 6] 트레일링 스탑(Trailing Stop) 리스크 관리 엔진 구동 중...\n")
    
    # 1. 멀티 타임프레임 연동 (이전 단계와 동일)
    df_1h = add_indicators(csv_1h)
    df_4h = add_indicators(csv_4h)
    
    df_4h['Trend_4h_Bullish'] = df_4h['close'] > df_4h['SMA_20']
    df_1h['Trend_4h_Bullish'] = df_4h['Trend_4h_Bullish'].reindex(df_1h.index, method='ffill')
    
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    df_1h.dropna(inplace=True) 
    
    # 2. 진입/청산 시그널
    entries = (
        (df_1h['close'] > df_1h['AVWAP_Weekly']) & 
        (df_1h['close'].shift(1) <= df_1h['AVWAP_Weekly'].shift(1)) & 
        (df_1h['Trend_4h_Bullish'] == True)
    )
    exits = df_1h['RSI_14'] >= 70
    
    # 3. 동적 레버리지 (변동성에 따른 투자 비중 조절)
    atr_pct = df_1h['ATR'] / df_1h['close']
    dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0)
    
    # 4. [핵심 튜닝] 트레일링 스탑이 장착된 포트폴리오
    pf = vbt.Portfolio.from_signals(
        close=df_1h['close'],
        entries=entries,
        exits=exits,
        size=dynamic_leverage,
        size_type='percent',
        
        # 🔻 기존의 고정 손절/익절을 스마트 로직으로 교체 🔻
        sl_stop=0.03,        # 고점 대비 3% 하락 시 즉시 청산 (이익 실현 또는 손절)
        sl_trail=True,       # ★ 트레일링 스탑 활성화 (수익을 악착같이 따라가며 보존함)
        tp_stop=0.20,        # 엄청난 폭등장(20%)에서는 기계적 익절 처리
        
        init_cash=10000,
        fees=0.0005,
        freq='1h'
    )
    
    print("\n📊 [스마트 트레일링 스탑 적용 성과 리포트]")
    print(pf.stats())
    
    return pf

if __name__ == "__main__":
    run_advanced_backtest()