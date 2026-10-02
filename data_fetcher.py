import urllib.request
import json
import pandas as pd
import time

def fetch_extensive_binance_data(symbol="BTCUSDT", interval="1h", target_candles=10000, filename="BTC_USDT_1h.csv"):
    print(f"📥 바이낸스 1시간봉(1H) 과거 데이터 대량 수집 시작... (목표: {target_candles}개 캔들)")
    all_data = []
    end_time = None
    limit = 1000

    while len(all_data) < target_candles:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time:
            url += f"&endTime={end_time}"
            
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
            
            if not data:
                break
                
            all_data = data + all_data 
            end_time = data[0][0] - 1 
            
            print(f"⏳ 진행률: {len(all_data)} / {target_candles} 캔들 병합 완료...")
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            break

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
    print(f"✅ [{filename}] 총 {len(df)}개의 1시간봉 데이터 수집 완료!")
    print(f"📅 테스트 기간: {df.index[0]} ~ {df.index[-1]}")
    print("-" * 50)

if __name__ == "__main__":
    print("🚀 [방화벽 우회 모드] 1시간봉(1H) 스윙 데이터 파이프라인 가동\n")
    fetch_extensive_binance_data(target_candles=10000)