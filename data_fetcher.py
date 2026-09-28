import requests
import pandas as pd

def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000, filename="BTC_USDT_1h.csv"):
    """
    바이낸스 퍼블릭 API를 호출하여 최신 실시간 캔들 데이터를 수집하고 CSV로 저장하는 파이프라인입니다.
    """
    print(f"📥 바이낸스 실시간 데이터 수집 중... (종목: {symbol}, 타임프레임: {interval})")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    # 바이낸스 서버에 데이터 요청
    response = requests.get(url, params=params)
    data = response.json()
    
    # 수집한 데이터를 판다스(Pandas) 표 형태로 변환
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # 사람이 읽을 수 있는 시간으로 변환 및 필수 컬럼 추출
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # 텍스트로 들어온 가격 데이터를 숫자(float)로 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('timestamp', inplace=True)
    
    # 기존 CSV 파일 덮어쓰기 (데이터 최신화)
    df.to_csv(filename)
    print(f"✅ [{filename}] 저장 완료! (총 {len(df)}개 캔들 | 최신 캔들 시간: {df.index[-1]})")

if __name__ == "__main__":
    print("🚀 [Step 12] 실시간 바이낸스 데이터 파이프라인 가동 시작\n")
    
    # 1시간봉과 4시간봉 최신 데이터 연속 수집
    fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000, filename="BTC_USDT_1h.csv")
    fetch_binance_data(symbol="BTCUSDT", interval="4h", limit=1000, filename="BTC_USDT_4h.csv")
    
    print("\n🎉 모든 데이터가 현재 시간 기준으로 최신화되었습니다. 이제 대시보드 엔진을 구동하면 실시간 시장 성과가 반영됩니다.")