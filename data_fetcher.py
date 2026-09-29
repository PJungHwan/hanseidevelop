import requests
import pandas as pd
import time

def fetch_extensive_binance_data(symbol="BTCUSDT", interval="5m", target_candles=20000, filename="BTC_USDT_5m.csv"):
    print(f"📥 바이낸스 과거 데이터 대량 수집 시작... (목표: {target_candles}개 캔들)")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    limit = 1000

    while len(all_data) < target_candles:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data:
                break
                
            # 수집한 데이터를 리스트 앞쪽에 이어 붙이기 (과거 -> 현재 순서 유지)
            all_data = data + all_data 
            
            # 다음 루프를 위해 이번에 수집한 가장 오래된 캔들의 바로 1밀리초 전을 endTime으로 설정
            end_time = data[0][0] - 1 
            
            print(f"⏳ 진행률: {len(all_data)} / {target_candles} 캔들 병합 완료...")
            time.sleep(0.2) # 바이낸스 API 서버 밴(Ban) 방지용 딜레이
            
        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            break

    # 목표 개수만큼 정확히 자르기
    all_data = all_data[-target_candles:]

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('timestamp', inplace=True)
    df.to_csv(filename)
    
    print("-" * 50)
    print(f"✅ [{filename}] 총 {len(df)}개의 5분봉 데이터 수집 완료!")
    print(f"📅 테스트 기간: {df.index[0]} ~ {df.index[-1]}")
    print("-" * 50)

if __name__ == "__main__":
    print("🚀 [Big Data] 70일치 대규모 과거 데이터 파이프라인 가동\n")
    fetch_extensive_binance_data(target_candles=20000)