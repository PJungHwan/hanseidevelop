import ccxt
import pandas as pd
import os

def fetch_and_save_data(symbol="BTC/USDT", timeframe="1h", limit=1000):
    """
    바이낸스에서 대용량 캔들 데이터를 가져와 로컬 CSV 파일로 저장합니다.
    """
    # 윈도우 파일명 규칙에 맞게 특수문자(/)를 언더바(_)로 변경
    clean_symbol = symbol.replace('/', '_')
    filename = f"{clean_symbol}_{timeframe}.csv"
    
    print(f"[{symbol}] {timeframe} 차트 캔들 {limit}개 수집 및 저장 시작...")
    exchange = ccxt.binance()
    
    # 바이낸스 API 호출 (한 번에 최대 1000개 제한)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # 데이터를 Pandas DataFrame(표)으로 변환
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # CSV 파일로 컴퓨터에 영구 저장 (로컬 데이터베이스화)
    df.to_csv(filename)
    print(f"✅ 저장 완료! 파일명: {filename} (총 {len(df)}개)")
    
    return df

if __name__ == "__main__":
    # 백테스트 비교 분석을 위해 1시간봉(단타/스윙)과 4시간봉(스윙) 데이터를 각각 1,000개씩 다운로드합니다.
    # 1시간봉 1000개 = 약 41일치 / 4시간봉 1000개 = 약 166일치 과거 데이터
    print("백테스트용 기초 캔들 데이터 구축을 시작합니다.\n")
    
    df_1h = fetch_and_save_data(symbol="BTC/USDT", timeframe="1h", limit=1000)
    df_4h = fetch_and_save_data(symbol="BTC/USDT", timeframe="4h", limit=1000)
    
    print("\n🎉 모든 데이터 수집 및 로컬 저장소 구축이 완료되었습니다!")
    
