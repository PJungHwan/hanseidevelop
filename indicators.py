import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def get_sma(series: pd.Series, length: int = 20) -> pd.Series:
    """단순 이동평균선 (Simple Moving Average)"""
    return series.rolling(window=length).mean()


def get_ema(series: pd.Series, length: int = 20) -> pd.Series:
    """지수 이동평균선 (Exponential Moving Average)"""
    return series.ewm(span=length, adjust=False).mean()


def get_zlsma(series: pd.Series, length: int = 130) -> pd.Series:
    """영지연 이동평균선 (Zero-Lag Moving Average)"""
    lag = int((length - 1) / 2)
    data = series + (series - series.shift(lag))
    return data.ewm(span=length, adjust=False).mean()


def get_bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    볼린저 밴드 (Bollinger Bands) 산출:
    - BB_Middle : 중심선 (SMA_20)
    - BB_Upper  : 상단선 (Middle + std * 표준편차)
    - BB_Lower  : 하단선 (Middle - std * 표준편차)
    - BB_Width  : 밴드폭 ((Upper - Lower) / Middle * 100)
    """
    middle = series.rolling(window=length).mean()
    rolling_std = series.rolling(window=length).std(ddof=0)
    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)
    width = np.where(middle != 0, ((upper - lower) / middle) * 100.0, np.nan)

    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': middle,
        'BB_Lower': lower,
        'BB_Width': width
    }, index=series.index)


def get_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) 산출:
    - MACD_Line   : 단기 EMA(12) - 장기 EMA(26)
    - MACD_Signal : MACD_Line의 EMA(9)
    - MACD_Hist   : MACD_Line - MACD_Signal
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return pd.DataFrame({
        'MACD_Line': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Hist': macd_hist
    }, index=series.index)


def get_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index, Wilder's Smoothing 적용)
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where(avg_loss == 0, 100.0, rsi)
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)
    return pd.Series(rsi, index=series.index).where(series.notna() & avg_gain.notna())


def get_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    ATR (Average True Range, Wilder's Smoothing 적용)
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    return atr


def add_indicators(
    data_source,
    sma_length: int = 20,
    ema_length: int = 50,
    bb_length: int = 20,
    bb_std: float = 2.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_length: int = 14,
    atr_length: int = 14,
    dropna: bool = True
) -> pd.DataFrame:
    """
    저장된 CSV 경로(str) 또는 OHLCV DataFrame을 입력받아 핵심 보조지표
    (SMA, EMA, Bollinger Bands, MACD, RSI, ATR, Anchored VWAP)를 계산하고
    결측치(NaN) 없이 정제된 데이터프레임을 반환합니다.
    """
    if isinstance(data_source, str):
        print(f"[{data_source}] 보조지표(SMA, EMA, BB, MACD, RSI, ATR, AVWAP) 연산 중...")
        df = pd.read_csv(data_source, index_col='timestamp', parse_dates=True)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    else:
        raise ValueError("data_source는 CSV 파일 경로(str) 또는 pandas.DataFrame이어야 합니다.")

    # 1. 이동평균선 (SMA, EMA, ZLSMA)
    df[f'SMA_{sma_length}'] = get_sma(df['close'], length=sma_length)
    df[f'EMA_{ema_length}'] = get_ema(df['close'], length=ema_length)
    df['ZLSMA'] = get_zlsma(df['close'], length=130)
    df['EMA_200'] = get_ema(df['close'], length=200)  # 전략 2 추세 필터용
    df['VOL_SMA_20'] = df['volume'].rolling(window=20).mean()  # 전략 3 거래량 급증 판단용

    # 2. 볼린저 밴드 (상단, 중단, 하단, 밴드폭)
    bb_df = get_bbands(df['close'], length=bb_length, std=bb_std)
    df['BB_Upper'] = bb_df['BB_Upper']
    df['BB_Middle'] = bb_df['BB_Middle']
    df['BB_Lower'] = bb_df['BB_Lower']
    df['BB_Width'] = bb_df['BB_Width']

    # pandas_ta 호환 컬럼 별칭 유지
    df[f'BBL_{bb_length}_{bb_std}'] = df['BB_Lower']
    df[f'BBM_{bb_length}_{bb_std}'] = df['BB_Middle']
    df[f'BBU_{bb_length}_{bb_std}'] = df['BB_Upper']
    df[f'BBB_{bb_length}_{bb_std}'] = df['BB_Width']

    # 3. MACD (MACD Line, Signal Line, Histogram)
    macd_df = get_macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['MACD_Line'] = macd_df['MACD_Line']
    df['MACD_Signal'] = macd_df['MACD_Signal']
    df['MACD_Hist'] = macd_df['MACD_Hist']

    # 4. RSI (Relative Strength Index)
    df[f'RSI_{rsi_length}'] = get_rsi(df['close'], length=rsi_length)
    df['RSI'] = df[f'RSI_{rsi_length}']

    # 5. ATR (Average True Range)
    df[f'ATR_{atr_length}'] = get_atr(df, length=atr_length)
    df['ATR'] = df[f'ATR_{atr_length}']

    # 6. 커스텀 Anchored VWAP (매주 월요일 자정 기준으로 닻을 내리고 초기화)
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    pv = typical_price * df['volume']
    cum_pv = pv.groupby(pd.Grouper(freq='W-MON')).cumsum()
    cum_vol = df['volume'].groupby(pd.Grouper(freq='W-MON')).cumsum()
    df['AVWAP_Weekly'] = np.where(cum_vol != 0, cum_pv / cum_vol, df['close'])

    # 7. 무한대(inf) 및 초기 웜업 구간 결측치(NaN) 완벽 제거
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if dropna:
        df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    import os

    sample_csv = "BTC_USDT_1h.csv"
    if os.path.exists(sample_csv):
        df_ind = add_indicators(sample_csv)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(f"\n✅ [{sample_csv}] 보조지표 장착 완료! (결측치 수: {df_ind.isna().sum().sum()}개, 총 행 수: {len(df_ind):,}개)")
        cols_to_show = [
            'close', 'SMA_20', 'EMA_50',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
            'MACD_Line', 'MACD_Signal', 'MACD_Hist',
            'RSI_14', 'ATR_14', 'AVWAP_Weekly'
        ]
        print(df_ind[cols_to_show].tail(5))
    else:
        print(f"ℹ️ '{sample_csv}' 파일이 없어 샘플 데이터로 지표 연산 및 결측치 검증을 수행합니다.")
        rng = np.random.default_rng(42)
        idx = pd.date_range("2026-01-01", periods=300, freq="1h")
        close_s = 60000 + np.cumsum(rng.normal(0, 150, size=300))
        sample_df = pd.DataFrame({
            "open": close_s - rng.uniform(-50, 50, size=300),
            "high": close_s + rng.uniform(50, 200, size=300),
            "low": close_s - rng.uniform(50, 200, size=300),
            "close": close_s,
            "volume": rng.uniform(100, 1000, size=300)
        }, index=idx)
        df_ind = add_indicators(sample_df)
        print(f"✅ 샘플 데이터 검증 완료! (결측치 수: {df_ind.isna().sum().sum()}개, 남은 행 수: {len(df_ind)}개)")
        print(df_ind[['close', 'SMA_20', 'EMA_50', 'BB_Upper', 'BB_Lower', 'BB_Width', 'MACD_Line', 'RSI_14', 'ATR_14']].tail(3))
