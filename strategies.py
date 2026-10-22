"""
유튜브 대표 매매 전략 시그널 생성 모듈.

모든 전략 함수는 `add_indicators()`가 반환한 지표 장착 DataFrame을 입력받아
(entries, exits, description) 을 반환합니다. (롱 온리, 캔들 종가 기준 시그널)
"""
import pandas as pd

STRATEGY_BB_RSI = "bb_rsi"
STRATEGY_EMA_MACD = "ema_macd"
STRATEGY_BB_SQUEEZE = "bb_squeeze"

STRATEGY_LABELS = {
    STRATEGY_BB_RSI: "전략 1: 볼린저 밴드 + RSI 역추세",
    STRATEGY_EMA_MACD: "전략 2: EMA 200 필터 + MACD 골든크로스 추세추종",
    STRATEGY_BB_SQUEEZE: "전략 3: 볼린저 스퀴즈 + 거래량 급증 돌파",
}


def signals_bb_rsi(df, bb_lower_col="BB_Lower", rsi_oversold=30.0, rsi_overbought=70.0, strict_rebound=False):
    """[전략 1] BB 하단 터치 + RSI 과매도 반등 롱 / BB 상단 터치 또는 RSI 과매수 청산."""
    bb_lower_touch = (df['low'] <= df[bb_lower_col]) | (df['low'].shift(1) <= df[bb_lower_col].shift(1))
    rsi_oversold_zone = (df['RSI'] <= rsi_oversold) | (df['RSI'].shift(1) <= rsi_oversold)
    rsi_up = df['RSI'] > df['RSI'].shift(1)

    if strict_rebound:
        entries = bb_lower_touch & rsi_oversold_zone & rsi_up & (df['close'] >= df['open'])
    else:
        direct = (df['low'] <= df[bb_lower_col]) & (df['RSI'] <= rsi_oversold)
        rebound = bb_lower_touch & rsi_oversold_zone & rsi_up
        entries = direct | rebound

    exits = (df['high'] >= df['BB_Upper']) | (df['close'] >= df['BB_Upper']) | (df['RSI'] >= rsi_overbought)
    return entries.fillna(False), exits.fillna(False)


def signals_ema_macd(df, trend_col="EMA_200", require_below_zero=False):
    """
    [전략 2] EMA 200 추세 필터 + MACD 골든크로스 추세추종.
    - 진입: 종가 > EMA200 이고 MACD_Line 이 MACD_Signal 을 상향 돌파(골든크로스)
            (require_below_zero=True 이면 크로스가 0선 아래에서 발생한 경우만)
    - 청산: MACD 데드크로스 또는 종가가 EMA200 하향 이탈
    """
    macd, sig = df['MACD_Line'], df['MACD_Signal']
    golden = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    dead = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    uptrend = df['close'] > df[trend_col]

    entries = golden & uptrend
    if require_below_zero:
        entries = entries & (macd < 0)
    exits = dead | (df['close'] < df[trend_col])
    return entries.fillna(False), exits.fillna(False)


def signals_bb_squeeze(df, squeeze_lookback=120, squeeze_quantile=0.2, squeeze_window=5,
                       vol_mult=2.0, vol_col="VOL_SMA_20"):
    """
    [전략 3] 볼린저 밴드 스퀴즈 + 거래량 급증 돌파.
    - 스퀴즈: BB_Width <= 최근 squeeze_lookback 봉 BB_Width 의 squeeze_quantile 분위수
    - 진입: 직전 squeeze_window 봉 이내 스퀴즈가 있었고, 종가 > BB_Upper,
            거래량 > vol_mult * 거래량 SMA(20)
    - 청산: 종가가 BB_Middle(중심선) 하향 이탈
    """
    width = df['BB_Width']
    threshold = width.rolling(squeeze_lookback, min_periods=max(20, squeeze_lookback // 4)).quantile(squeeze_quantile)
    squeeze = (width <= threshold).astype(float)
    recent_squeeze = squeeze.rolling(squeeze_window, min_periods=1).max().shift(1) > 0

    vol_surge = df['volume'] > vol_mult * df[vol_col]
    breakout = df['close'] > df['BB_Upper']

    entries = recent_squeeze & vol_surge & breakout
    exits = df['close'] < df['BB_Middle']
    return entries.fillna(False), exits.fillna(False)

