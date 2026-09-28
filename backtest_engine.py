import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from indicators import add_indicators

def run_strategy_benchmark(csv_1h="BTC_USDT_1h.csv", csv_4h="BTC_USDT_4h.csv"):
    """
    다양한 퀀트 전략들을 동시에 평가하여 순위를 매기는 범용 백테스팅 플랫폼 모듈입니다.
    """
    print("🚀 [Step 10] 범용 퀀트 테스트베드 구동: 4대 매매 전략 성과 벤치마킹...\n")
    
    # 1. 기초 데이터 준비
    df_1h = add_indicators(csv_1h)
    df_4h = add_indicators(csv_4h)
    
    # 2. 공통 지표 연산
    df_1h['ATR'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
    df_1h['SMA_20'] = ta.sma(df_1h['close'], length=20)
    df_1h['SMA_50'] = ta.sma(df_1h['close'], length=50)
    df_1h['STD_20'] = df_1h['close'].rolling(20).std()
    df_1h['BB_Lower'] = df_1h['SMA_20'] - (2 * df_1h['STD_20'])
    
    # 우리팀 전략용 4시간봉 필터 연산
    df_4h['SMA_30'] = ta.sma(df_4h['close'], length=30)
    df_4h['Trend_4h_Bullish'] = df_4h['close'] > df_4h['SMA_30']
    df_1h['Trend_4h_Bullish'] = df_4h['Trend_4h_Bullish'].reindex(df_1h.index, method='ffill')
    
    df_1h.dropna(inplace=True)
    
    # 공통 동적 레버리지 (리스크 관리)
    atr_pct = df_1h['ATR'] / df_1h['close']
    dynamic_leverage = np.clip(0.02 / atr_pct, 1.0, 10.0)

    # 3. [모듈화] 4가지 전략의 진입(Entry) / 청산(Exit) 룰 정의
    
    # 전략 A: 래리 윌리엄스 변동성 돌파
    range_prev = df_1h['high'].shift(1) - df_1h['low'].shift(1)
    entry_larry = df_1h['close'] > (df_1h['open'] + range_prev * 0.5)
    exit_larry = df_1h['RSI_14'] >= 70 # 기본적인 RSI 익절
    
    # 전략 B: 20/50 이평선 골든크로스
    entry_gc = (df_1h['SMA_20'] > df_1h['SMA_50']) & (df_1h['SMA_20'].shift(1) <= df_1h['SMA_50'].shift(1))
    exit_gc = df_1h['SMA_20'] < df_1h['SMA_50'] # 데드크로스 청산
    
    # 전략 C: 볼린저 밴드 하단 반등
    entry_bb = df_1h['close'] < df_1h['BB_Lower']
    exit_bb = df_1h['close'] > df_1h['SMA_20'] # 중앙선 회귀 시 청산
    
    # 전략 D: [우리 팀 독자 최적화 전략] (AVWAP + 4H 필터 + RSI 90)
    entry_ours = (
        (df_1h['close'] > df_1h['AVWAP_Weekly']) & 
        (df_1h['close'].shift(1) <= df_1h['AVWAP_Weekly'].shift(1)) & 
        (df_1h['Trend_4h_Bullish'] == True)
    )
    exit_ours = df_1h['RSI_14'] >= 90
    
    # 전략 딕셔너리 구성
    strategies = {
        "1. 래리 윌리엄스 변동성 돌파": (entry_larry, exit_larry),
        "2. 이평선 골든크로스 (20/50)": (entry_gc, exit_gc),
        "3. 볼린저 밴드 하단 반등": (entry_bb, exit_bb),
        "4. [한세대 2팀] MTF 최적화 (AVWAP+RSI90)": (entry_ours, exit_ours)
    }

    results = []
    
    # 4. 각 전략들을 엔진에 순차적으로 넣고 백테스트 실행
    for name, (entries, exits) in strategies.items():
        pf = vbt.Portfolio.from_signals(
            close=df_1h['close'],
            entries=entries,
            exits=exits,
            size=dynamic_leverage,
            size_type='percent',
            sl_stop=0.05,        # 모든 전략에 동일한 5% 방어막 적용
            sl_trail=True,
            init_cash=10000,
            fees=0.0005,
            freq='1h'
        )
        
        returns = pf.total_return() * 100
        win_rate = pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0
        
        results.append({
            'Strategy_Name': name,
            'Win_Rate(%)': win_rate,
            'Total_Return(%)': returns,
            'Total_Trades': pf.trades.count()
        })
        
    # 5. 최종 리더보드 출력
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Total_Return(%)', ascending=False)
    
    print("🏆 [다중 전략 통합 백테스트 벤치마킹 결과]")
    print(res_df.to_string(index=False))
    
    return res_df

if __name__ == "__main__":
    run_strategy_benchmark()