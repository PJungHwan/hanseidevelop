import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_advanced_backtest(csv_path="BTC_USDT_1h.csv"):
    """
    시장 변동성에 맞춘 동적 레버리지와 TP/SL(익절/손절)이 결합된 고도화 엔진입니다.
    """
    print(f"🚀 [{csv_path}] 동적 레버리지 및 리스크 관리 시스템 가동...\n")
    
    df = add_indicators(csv_path)
    
    # 1. 시장 변동성 측정을 위한 ATR(Average True Range) 지표 추가
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df.dropna(inplace=True) # 새로 생긴 빈칸 제거
    
    # 2. 매수/매도 시그널 (Golden Cross / RSI 과매수)
    entries = (df['close'] > df['AVWAP_Weekly']) & (df['close'].shift(1) <= df['AVWAP_Weekly'].shift(1))
    exits = df['RSI_14'] >= 70
    
    # 3. [핵심] 변동성 기반 동적 레버리지(Dynamic Leverage) 비율 계산
    # 현재 가격 대비 변동성(ATR)의 비율을 구함
    atr_pct = df['ATR'] / df['close']
    
    # 변동성이 높으면 레버리지를 줄이고, 낮으면 레버리지를 늘리는 탄력적 유연 로직
    # np.clip을 사용해 최소 1배수 ~ 최대 10배수 사이로 제한
    dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0) 
    
    # 4. Vectorbt 포트폴리오 가동 (자본금 10,000 USDT 기준)
    pf = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=entries,
        exits=exits,
        size=dynamic_leverage,     # 실시간으로 계산된 동적 레버리지 배율을 투자 비중으로 적용
        size_type='percent',       # 자본금 대비 비율로 해석 (예: 2.0 = 2배수 레버리지)
        sl_stop=0.05,              # 5% 하락 시 기계적 손절(Stop Loss)
        tp_stop=0.15,              # 15% 상승 시 자동 익절(Take Profit)
        init_cash=10000,
        fees=0.0005,               # 바이낸스 시장가 수수료
        freq='1h' 
    )
    
    print("\n📊 [고도화된 백테스트 성과 요약 리포트]")
    print(pf.stats())
    
    return pf

if __name__ == "__main__":
    run_advanced_backtest("BTC_USDT_1h.csv")