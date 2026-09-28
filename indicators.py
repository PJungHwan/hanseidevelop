import pandas as pd
import pandas_ta as ta

def add_indicators(csv_path):
    """
    저장된 CSV 데이터를 불러와 핵심 보조지표 및 커스텀 Anchored VWAP을 추가합니다.
    """
    print(f"[{csv_path}] 보조지표 및 Anchored VWAP 연산 중...")
    
    df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
    
    # 1. 기본 지표들 (이평선, RSI, 볼린저 밴드)
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    bbands = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # 2. 커스텀 Anchored VWAP (매주 월요일 자정 기준으로 닻을 내리고 초기화)
    # 캔들의 평균 가격(Typical Price) = (고가 + 저가 + 종가) / 3
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typical_price'] * df['volume']
    
    # 주간 단위(W-MON: 매주 월요일 기준)로 그룹을 묶어서 거래량과 가격*거래량을 누적합산
    df['cum_pv'] = df.groupby(pd.Grouper(freq='W-MON'))['pv'].cumsum()
    df['cum_vol'] = df.groupby(pd.Grouper(freq='W-MON'))['volume'].cumsum()
    
    # 누적 (가격*거래량) / 누적 거래량 = Anchored VWAP
    df['AVWAP_Weekly'] = df['cum_pv'] / df['cum_vol']
    
    # 연산용으로 썼던 임시 열들은 깔끔하게 삭제
    df.drop(columns=['typical_price', 'pv', 'cum_pv', 'cum_vol'], inplace=True)
    
    # 빈칸(NaN)이 생긴 앞부분 행들 제거
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # 1시간봉 데이터에 지표를 장착하여 테스트
    df_with_indicators = add_indicators("BTC_USDT_1h.csv")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n✅ 커스텀 Anchored VWAP 장착 완료! 최신 데이터 5줄 확인:")
    print(df_with_indicators[['close', 'SMA_20', 'RSI_14', 'AVWAP_Weekly']].tail(5))