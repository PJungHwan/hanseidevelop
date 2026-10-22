import os
import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from indicators import (
    get_atr,
    get_zlsma,
    add_indicators
)


def load_timeframe_data(timeframe="1h", csv_path=None):
    """
    지정된 타임프레임('1h' 또는 '4h')의 OHLCV 데이터를 로드합니다.
    - '4h' 요청 시 BTC_USDT_4h.csv가 없으면 BTC_USDT_1h.csv를 불러와 4시간봉으로 자동 리샘플링합니다.
    """
    tf = timeframe.lower().strip()
    if tf not in ("1h", "4h"):
        raise ValueError(f"지원하지 않는 타임프레임입니다: {timeframe} ('1h' 또는 '4h'를 사용하세요.)")

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        if tf == "4h" and "1h" in os.path.basename(csv_path).lower():
            df = resample_ohlcv(df, "4h")
        return df

    default_target = f"BTC_USDT_{tf}.csv"
    if os.path.exists(default_target):
        return pd.read_csv(default_target, index_col='timestamp', parse_dates=True)

    # 4h CSV가 없을 경우 1h CSV에서 리샘플링 지원
    fallback_1h = "BTC_USDT_1h.csv"
    if tf == "4h" and os.path.exists(fallback_1h):
        print(f"ℹ️ '{default_target}' 파일이 없어 '{fallback_1h}' 데이터를 4시간봉(4h)으로 리샘플링합니다.")
        df_1h = pd.read_csv(fallback_1h, index_col='timestamp', parse_dates=True)
        return resample_ohlcv(df_1h, "4h")

    raise FileNotFoundError(
        f"데이터 파일('{default_target}' 또는 '{fallback_1h}')을 찾을 수 없습니다. "
        "먼저 data_fetcher.py를 실행해 데이터를 수집해 주세요."
    )


def resample_ohlcv(df: pd.DataFrame, freq: str = "4h") -> pd.DataFrame:
    """1시간봉 등 하위 타임프레임 OHLCV 데이터를 상위 타임프레임(예: '4h')으로 리샘플링합니다."""
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled = df.resample(freq).agg(agg_dict).dropna()
    return resampled


def extract_backtest_metrics(df, pf, init_cash=10000.0, leverage=1.0, fee_rate=0.0004):
    """
    vectorbt Portfolio 객체로부터 매매 진입/청산 타임스탬프와
    수수료가 명확히 반영된 총수익, 총손실, 총수수료, 승률, MDD 및 상세 거래 기록(Trade Log)을 추출합니다.
    """
    records = pf.trades.records_readable
    if len(records) == 0:
        empty_summary = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_gross_profit": 0.0,
            "total_gross_loss": 0.0,
            "total_fees": 0.0,
            "net_profit": 0.0,
            "mdd_pct": 0.0,
            "final_equity": init_cash,
        }
        return empty_summary, pd.DataFrame()

    # 1. 진입/청산 타임스탬프 및 체결 가격 매핑
    raw_records = pf.trades.records
    entry_indices = raw_records['entry_idx'].values
    exit_indices = raw_records['exit_idx'].values

    entry_timestamps = df.index[entry_indices]
    exit_timestamps = df.index[exit_indices]

    entry_prices = raw_records['entry_price'].values
    exit_prices = raw_records['exit_price'].values
    directions = np.where(raw_records['direction'].values == 0, 'Long', 'Short')
    statuses = np.where(raw_records['status'].values == 1, 'Closed', 'Open')

    # 2. 수수료 차감 전 가격 변동 수익률 및 왕복 수수료(진입 + 청산) 명시적 산출
    raw_price_ret_1x = np.where(
        raw_records['direction'].values == 0,
        (exit_prices - entry_prices) / entry_prices,
        (entry_prices - exit_prices) / entry_prices
    )
    gross_return_lev = raw_price_ret_1x * leverage

    # 진입 수수료율(fee_rate) + 청산 시 변동된 가치에 대한 수수료율(fee_rate * (1 + raw_price_ret_1x))
    roundtrip_fee_1x = fee_rate + fee_rate * (1.0 + raw_price_ret_1x)
    roundtrip_fee_lev = roundtrip_fee_1x * leverage
    fee_paid = roundtrip_fee_lev * init_cash

    # 3. 수수료가 반영된 최종 수익률 (vectorbt pf.trades.returns는 수수료가 이미 반영된 1배수 수익률)
    returns_1x_net = pf.trades.returns.values
    returns_lev_net = returns_1x_net * leverage
    # 레버리지 적용 후 -100% 이하 손실 시 원금 전액 청산(-100%) 처리
    returns_lev_net = np.where(returns_lev_net <= -1.0, -1.0, returns_lev_net)

    net_pnl = returns_lev_net * init_cash
    cumulative_equity = init_cash + np.cumsum(net_pnl)

    # 4. 수수료가 반영된 총수익(Total Gross Profit), 총손실(Total Gross Loss), 순손익, 승률, MDD 계산
    total_trades = len(returns_lev_net)
    winning_mask = net_pnl > 0
    losing_mask = net_pnl <= 0

    winning_trades = int(np.sum(winning_mask))
    losing_trades = int(np.sum(losing_mask))
    win_rate = (winning_trades / total_trades) * 100.0

    total_gross_profit = float(np.sum(net_pnl[winning_mask]))
    total_gross_loss = float(np.sum(net_pnl[losing_mask]))
    total_fees = float(np.sum(fee_paid))
    net_profit = total_gross_profit + total_gross_loss

    # MDD (Maximum Drawdown) 계산 (초기 자본금 포함 누적 자산 곡선 기준)
    equity_curve = np.concatenate([[init_cash], cumulative_equity])
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(running_peak > 0, (equity_curve - running_peak) / running_peak, 0.0)
    mdd_pct = float(np.min(drawdowns) * 100.0)

    # 5. 상세 거래 기록(Trade Log) 데이터프레임 구성
    trades_df = pd.DataFrame({
        "trade_id": np.arange(1, total_trades + 1),
        "entry_timestamp": entry_timestamps,
        "exit_timestamp": exit_timestamps,
        "direction": directions,
        "entry_price": np.round(entry_prices, 2),
        "exit_price": np.round(exit_prices, 2),
        "gross_return_pct": np.round(gross_return_lev * 100.0, 4),
        "fee_paid": np.round(fee_paid, 2),
        "net_return_pct": np.round(returns_lev_net * 100.0, 4),
        "net_pnl": np.round(net_pnl, 2),
        "cumulative_equity": np.round(cumulative_equity, 2),
        "status": statuses
    })

    summary = {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_gross_profit": total_gross_profit,
        "total_gross_loss": total_gross_loss,
        "total_fees": total_fees,
        "net_profit": net_profit,
        "mdd_pct": mdd_pct,
        "final_equity": float(cumulative_equity[-1]),
    }

    return summary, trades_df


def _simulate_signals_fallback(df, entries, exits, init_cash=10000.0, leverage=1.0, fee_rate=0.0004, tp_stop=None, sl_stop=None):
    """
    vectorbt 미설치 환경에서도 동일하게 동작하는 순수 NumPy/Pandas 기반 백테스트 폴백 엔진입니다.
    """
    closes = df['close'].values
    timestamps = df.index
    entries_arr = np.asarray(entries, dtype=bool)
    exits_arr = np.asarray(exits, dtype=bool)

    in_pos = False
    entry_idx = 0
    entry_price = 0.0
    trade_records = []

    for i in range(len(closes)):
        price = closes[i]
        if not in_pos:
            if entries_arr[i]:
                in_pos = True
                entry_idx = i
                entry_price = price
        else:
            raw_ret = (price - entry_price) / entry_price
            hit_tp = (tp_stop is not None) and (raw_ret >= tp_stop)
            hit_sl = (sl_stop is not None) and (raw_ret <= -sl_stop)
            if exits_arr[i] or hit_tp or hit_sl:
                exit_price = entry_price * (1.0 + tp_stop) if hit_tp else (
                    entry_price * (1.0 - sl_stop) if hit_sl else price
                )
                trade_records.append((entry_idx, i, entry_price, exit_price, 1))
                in_pos = False

    if len(trade_records) == 0:
        empty_summary = {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate": 0.0,
            "total_gross_profit": 0.0, "total_gross_loss": 0.0, "total_fees": 0.0,
            "net_profit": 0.0, "mdd_pct": 0.0, "final_equity": init_cash,
        }
        return empty_summary, pd.DataFrame(), None

    entry_indices = np.array([r[0] for r in trade_records])
    exit_indices = np.array([r[1] for r in trade_records])
    entry_prices = np.array([r[2] for r in trade_records], dtype=float)
    exit_prices = np.array([r[3] for r in trade_records], dtype=float)

    raw_price_ret_1x = (exit_prices - entry_prices) / entry_prices
    gross_return_lev = raw_price_ret_1x * leverage

    roundtrip_fee_1x = fee_rate + fee_rate * (1.0 + raw_price_ret_1x)
    roundtrip_fee_lev = roundtrip_fee_1x * leverage
    fee_paid = roundtrip_fee_lev * init_cash

    returns_1x_net = ((1.0 - fee_rate) * (1.0 + raw_price_ret_1x) * (1.0 - fee_rate)) - 1.0
    returns_lev_net = np.where(returns_1x_net * leverage <= -1.0, -1.0, returns_1x_net * leverage)

    net_pnl = returns_lev_net * init_cash
    cumulative_equity = init_cash + np.cumsum(net_pnl)

    total_trades = len(returns_lev_net)
    winning_mask = net_pnl > 0
    losing_mask = net_pnl <= 0
    winning_trades = int(np.sum(winning_mask))
    losing_trades = int(np.sum(losing_mask))
    win_rate = (winning_trades / total_trades) * 100.0

    total_gross_profit = float(np.sum(net_pnl[winning_mask]))
    total_gross_loss = float(np.sum(net_pnl[losing_mask]))
    total_fees = float(np.sum(fee_paid))
    net_profit = total_gross_profit + total_gross_loss

    equity_curve = np.concatenate([[init_cash], cumulative_equity])
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(running_peak > 0, (equity_curve - running_peak) / running_peak, 0.0)
    mdd_pct = float(np.min(drawdowns) * 100.0)

    trades_df = pd.DataFrame({
        "trade_id": np.arange(1, total_trades + 1),
        "entry_timestamp": timestamps[entry_indices],
        "exit_timestamp": timestamps[exit_indices],
        "direction": ["Long"] * total_trades,
        "entry_price": np.round(entry_prices, 2),
        "exit_price": np.round(exit_prices, 2),
        "gross_return_pct": np.round(gross_return_lev * 100.0, 4),
        "fee_paid": np.round(fee_paid, 2),
        "net_return_pct": np.round(returns_lev_net * 100.0, 4),
        "net_pnl": np.round(net_pnl, 2),
        "cumulative_equity": np.round(cumulative_equity, 2),
        "status": ["Closed"] * total_trades
    })

    summary = {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_gross_profit": total_gross_profit,
        "total_gross_loss": total_gross_loss,
        "total_fees": total_fees,
        "net_profit": net_profit,
        "mdd_pct": mdd_pct,
        "final_equity": float(cumulative_equity[-1]),
    }
    return summary, trades_df, None


def print_backtest_report(title: str, summary: dict, trades_df: pd.DataFrame, max_log_rows: int = 10):
    """
    백테스트 성과 요약(총수익, 총손실, 지불 수수료, 승률, MDD) 및
    상세 매매기록(진입/청산 시간, 단가, 수수료, 순손익)을 출력합니다.
    """
    print("\n" + "=" * 78)
    print(f"▶ {title}")
    print("-" * 78)
    if summary["total_trades"] == 0:
        print("매매 내역이 없습니다.")
        print("=" * 78)
        return

    print(f"총 진입 횟수          : {summary['total_trades']} 번")
    print(f"승리(Win) / 패배(Loss): {summary['winning_trades']} 번 / {summary['losing_trades']} 번")
    print(f"최종 승률 (Win Rate)  : {summary['win_rate']:.2f} %")
    print(f"총 수익 (수수료 반영) : +${summary['total_gross_profit']:,.2f}")
    print(f"총 손실 (수수료 반영) : -${abs(summary['total_gross_loss']):,.2f}")
    print(f"총 지불 수수료        : ${summary['total_fees']:,.2f}")
    print(f"최종 순 손익 (Net PnL): ${summary['net_profit']:,.2f} (최종 자산: ${summary['final_equity']:,.2f})")
    print(f"최대 낙폭 (MDD)       : {summary['mdd_pct']:.2f} %")
    print("-" * 78)
    print(f"[상세 매매기록 (Trade Log) - 총 {len(trades_df)}건 중 최근 {min(max_log_rows, len(trades_df))}건]")
    display_cols = [
        "trade_id", "entry_timestamp", "exit_timestamp",
        "entry_price", "exit_price", "fee_paid", "net_return_pct", "net_pnl"
    ]
    print(trades_df[display_cols].tail(max_log_rows).to_string(index=False))
    print("=" * 78)


def run_youtube_bb_rsi_strategy(
    timeframe: str = "1h",
    csv_path: str = None,
    df_input: pd.DataFrame = None,
    bb_length: int = 20,
    bb_std: float = 2.0,
    rsi_length: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    init_cash: float = 10000.0,
    leverage: float = 1.0,
    fee_rate: float = 0.0004,
    strict_rebound: bool = False
):
    """
    [유튜브 대표 매매 전략: 볼린저 밴드 하단 터치 + RSI 과매도 반등 롱(Long) 전략]
    - 타임프레임: '1h' (1시간봉) 또는 '4h' (4시간봉) 인자 지원
    - 롱(Long) 진입 조건:
        1) 직전 또는 현재 캔들이 볼린저 밴드 하단(BB_Lower)을 터치/이탈 (low <= BB_Lower 또는 close <= BB_Lower)
        2) RSI가 과매도 구간(기본 30 이하)에 진입했거나 직전 과매도 침체 후 반등(RSI > prev_RSI & close > open)할 때 진입
    - 롱(Long) 청산 조건:
        1) 캔들이 볼린저 밴드 상단(BB_Upper)을 터치/돌파 (high >= BB_Upper 또는 close >= BB_Upper)
        2) 또는 RSI가 과매수 구간(기본 70 이상)에 도달할 때 청산 (RSI >= 70)
    """
    tf = timeframe.lower().strip()
    if tf not in ("1h", "4h"):
        raise ValueError("timeframe 인자는 '1h' 또는 '4h'여야 합니다.")

    if df_input is not None:
        raw_df = df_input.copy()
        if tf == "4h":
            # 입력 데이터 주기가 1시간봉이면 4시간봉으로 리샘플링
            inferred_diff = raw_df.index.to_series().diff().median()
            if pd.notna(inferred_diff) and inferred_diff <= pd.Timedelta(hours=1):
                raw_df = resample_ohlcv(raw_df, "4h")
    else:
        raw_df = load_timeframe_data(timeframe=tf, csv_path=csv_path)

    # indicators.py의 add_indicators()를 호출하여 결측치 없이 전체 보조지표 계산
    df = add_indicators(
        raw_df,
        bb_length=bb_length,
        bb_std=bb_std,
        rsi_length=rsi_length,
        dropna=True
    )

    # --- [진입 조건: 볼린저 밴드 하단 터치 + RSI 과매도(30 이하) 반등] ---
    bb_lower_touch = (df['low'] <= df['BB_Lower']) | (df['low'].shift(1) <= df['BB_Lower'].shift(1))
    rsi_oversold_zone = (df['RSI'] <= rsi_oversold) | (df['RSI'].shift(1) <= rsi_oversold)
    rsi_rebound = (df['RSI'] > df['RSI'].shift(1)) & (df['close'] >= df['open'])

    if strict_rebound:
        # 엄격 모드: 하단 밴드 터치 + 과매도 이력 + 양봉 & RSI 상승 반등 확인 시 진입
        entries = bb_lower_touch & rsi_oversold_zone & rsi_rebound
    else:
        # 표준 모드: (당일 하단 터치 & 과매도) 또는 (직전/당일 하단 터치 & 과매도 후 RSI 반등)
        direct_touch_oversold = (df['low'] <= df['BB_Lower']) & (df['RSI'] <= rsi_oversold)
        rebound_after_oversold = bb_lower_touch & rsi_oversold_zone & (df['RSI'] > df['RSI'].shift(1))
        entries = direct_touch_oversold | rebound_after_oversold

    # --- [청산 조건: 볼린저 밴드 상단 터치 또는 RSI 과매수(70 이상)] ---
    bb_upper_touch = (df['high'] >= df['BB_Upper']) | (df['close'] >= df['BB_Upper'])
    rsi_overbought_hit = df['RSI'] >= rsi_overbought
    exits = bb_upper_touch | rsi_overbought_hit

    strategy_title = (
        f"[유튜브 BB+RSI 전략 ({tf.upper()})] "
        f"BB({bb_length},{bb_std}) 하단터치 + RSI({rsi_length})≤{int(rsi_oversold)} 반등 진입 / "
        f"BB 상단터치 or RSI≥{int(rsi_overbought)} 청산 (레버리지 {leverage}x)"
    )

    if HAS_VBT:
        pf = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fee_rate,
            freq=tf
        )
        summary, trades_df = extract_backtest_metrics(
            df=df, pf=pf, init_cash=init_cash, leverage=leverage, fee_rate=fee_rate
        )
    else:
        summary, trades_df, pf = _simulate_signals_fallback(
            df=df, entries=entries, exits=exits,
            init_cash=init_cash, leverage=leverage, fee_rate=fee_rate
        )

    print_backtest_report(strategy_title, summary, trades_df)

    return {
        "timeframe": tf,
        "summary": summary,
        "trades": trades_df,
        "portfolio": pf,
        "data": df
    }


def run_strategy_benchmark(csv_1h="BTC_USDT_1h.csv", init_cash=10000.0, leverage=50.0, fee_rate=0.0004, freq='1h'):
    print("🚀 [1시간봉 격상] 종가 마감 vs 실시간 돌파 승률 테스트 가동...\n")
    df = pd.read_csv(csv_1h, index_col='timestamp', parse_dates=True)
    
    len_A = 130
    df['ZLSMA'] = get_zlsma(df['close'], len_A)
    
    atr_A, mult_A = 22, 3
    df['ATR'] = get_atr(df, atr_A)
    df['CE_Long'] = df['high'].rolling(atr_A).max() - (df['ATR'] * mult_A)
    df['CE_Short'] = df['low'].rolling(atr_A).min() + (df['ATR'] * mult_A)
    
    buy_close = (df['close'] > df['ZLSMA']) & (df['close'] > df['CE_Short']) & (df['close'].shift(1) <= df['CE_Short'].shift(1))
    sell_close = (df['close'] < df['CE_Long']) & (df['close'].shift(1) >= df['CE_Long'].shift(1))
    
    entry_prc_close = df['close'].where(buy_close).ffill()
    entry_atr_close = df['ATR'].where(buy_close).ffill()
    sl_close = df['close'] < (entry_prc_close - (entry_atr_close * 1.5))
    exit_close = sell_close | sl_close

    buy_high = (df['high'] > df['ZLSMA'].shift(1)) & (df['high'] > df['CE_Short'].shift(1)) & (df['close'].shift(1) <= df['CE_Short'].shift(1))
    sell_low = (df['low'] < df['CE_Long'].shift(1)) & (df['close'].shift(1) >= df['CE_Long'].shift(1))
    
    entry_prc_high = df['close'].where(buy_high).ffill() 
    entry_atr_high = df['ATR'].where(buy_high).ffill()
    sl_high = df['close'] < (entry_prc_high - (entry_atr_high * 1.5))
    exit_high = sell_low | sl_high

    target_profit = 0.004 # 0.4% 타겟 익절 (50배 기준 20% 수익)

    strategies = {
        "1. [정석] 1시간봉 종가 마감 확인 후 진입": {'entries': buy_close, 'exits': exit_close},
        "2. [공격] 1시간봉 꼬리 돌파 즉시 진입": {'entries': buy_high, 'exits': exit_high}
    }

    print(f"📊 {len(df):,}개의 캔들({freq})에서 매매 타점 및 타임스탬프/수수료 내역을 추출합니다...")
    
    benchmark_results = {}

    for name, params in strategies.items():
        if HAS_VBT:
            pf = vbt.Portfolio.from_signals(
                close=df['close'],
                entries=params['entries'],
                exits=params['exits'],
                tp_stop=target_profit,
                init_cash=init_cash,
                fees=fee_rate, 
                freq=freq
            )
            summary, trades_df = extract_backtest_metrics(
                df=df, pf=pf, init_cash=init_cash, leverage=leverage, fee_rate=fee_rate
            )
        else:
            summary, trades_df, pf = _simulate_signals_fallback(
                df=df, entries=params['entries'], exits=params['exits'],
                init_cash=init_cash, leverage=leverage, fee_rate=fee_rate, tp_stop=target_profit
            )

        benchmark_results[name] = {
            "summary": summary,
            "trades": trades_df,
            "portfolio": pf
        }

        print_backtest_report(name, summary, trades_df, max_log_rows=3)

    return benchmark_results


if __name__ == "__main__":
    print("==============================================================================")
    print("🎯 [백테스트 엔진 실행] 유튜브 볼린저 밴드 + RSI 매매 전략 (1H / 4H 비교 검증)")
    print("==============================================================================")

    if os.path.exists("BTC_USDT_1h.csv") or os.path.exists("BTC_USDT_4h.csv"):
        res_1h = run_youtube_bb_rsi_strategy(timeframe="1h", leverage=5.0)
        res_4h = run_youtube_bb_rsi_strategy(timeframe="4h", leverage=5.0)
        if os.path.exists("BTC_USDT_1h.csv"):
            run_strategy_benchmark("BTC_USDT_1h.csv")
    else:
        print("ℹ️ 로컬 CSV('BTC_USDT_1h.csv')가 없어 시뮬레이션 샘플 캔들(2,000개 1H)로 1H 및 4H 전략을 테스트합니다.")
        rng = np.random.default_rng(7)
        idx = pd.date_range("2026-01-01", periods=2000, freq="1h")
        # 평균회귀 + 파동 특성을 가진 현실적인 가격 시계열 생성
        t = np.linspace(0, 40 * np.pi, 2000)
        base_wave = 65000 + 2500 * np.sin(t) + np.cumsum(rng.normal(0, 120, size=2000))
        open_p = base_wave + rng.normal(0, 80, size=2000)
        close_p = base_wave + rng.normal(0, 80, size=2000)
        high_p = np.maximum(open_p, close_p) + rng.uniform(50, 350, size=2000)
        low_p = np.minimum(open_p, close_p) - rng.uniform(50, 350, size=2000)
        vol_p = rng.uniform(100, 1500, size=2000)

        synth_df = pd.DataFrame({
            "open": open_p, "high": high_p, "low": low_p, "close": close_p, "volume": vol_p
        }, index=idx)

        res_1h = run_youtube_bb_rsi_strategy(timeframe="1h", df_input=synth_df, leverage=5.0)
        res_4h = run_youtube_bb_rsi_strategy(timeframe="4h", df_input=synth_df, leverage=5.0)

    if os.path.exists("BTC_USDT_1h.csv") or os.path.exists("BTC_USDT_4h.csv"):
        res_1h = run_youtube_bb_rsi_strategy(timeframe="1h", leverage=5.0)
        res_4h = run_youtube_bb_rsi_strategy(timeframe="4h", leverage=5.0)
        if os.path.exists("BTC_USDT_1h.csv"):
            run_strategy_benchmark("BTC_USDT_1h.csv")
    else:
        print("ℹ️ 로컬 CSV('BTC_USDT_1h.csv')가 없어 시뮬레이션 샘플 캔들(2,000개 1H)로 1H 및 4H 전략을 테스트합니다.")
        rng = np.random.default_rng(7)
        idx = pd.date_range("2026-01-01", periods=2000, freq="1h")
        # 평균회귀 + 파동 특성을 가진 현실적인 가격 시계열 생성
        t = np.linspace(0, 40 * np.pi, 2000)
        base_wave = 65000 + 2500 * np.sin(t) + np.cumsum(rng.normal(0, 120, size=2000))
        open_p = base_wave + rng.normal(0, 80, size=2000)
        close_p = base_wave + rng.normal(0, 80, size=2000)
        high_p = np.maximum(open_p, close_p) + rng.uniform(50, 350, size=2000)
        low_p = np.minimum(open_p, close_p) - rng.uniform(50, 350, size=2000)
        vol_p = rng.uniform(100, 1500, size=2000)

        synth_df = pd.DataFrame({
            "open": open_p, "high": high_p, "low": low_p, "close": close_p, "volume": vol_p
        }, index=idx)

        res_1h = run_youtube_bb_rsi_strategy(timeframe="1h", df_input=synth_df, leverage=5.0)
        res_4h = run_youtube_bb_rsi_strategy(timeframe="4h", df_input=synth_df, leverage=5.0)

