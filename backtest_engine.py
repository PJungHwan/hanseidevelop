import os
import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from strategies import (
    STRATEGY_BB_RSI, STRATEGY_EMA_MACD, STRATEGY_BB_SQUEEZE, STRATEGY_LABELS,
    signals_bb_rsi, signals_ema_macd, signals_bb_squeeze
)
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


def _simulate_signals_fallback(df, entries, exits, init_cash=10000.0, leverage=1.0, fee_rate=0.0004,
                               tp_stop=None, sl_stop=None, sl_frac=None, tp_frac=None):
    """
    vectorbt 미설치 환경에서도 동일하게 동작하는 순수 NumPy/Pandas 기반 백테스트 폴백 엔진입니다.
    - sl_frac / tp_frac: 진입 캔들 기준 봉별 손절/익절 폭(진입가 대비 비율, 1배수 가격 기준) 배열.
      (스칼라 sl_stop / tp_stop 도 동일하게 전 구간 상수 배열로 변환되어 처리됩니다.)
    - 캔들 내 고가/저가로 손절·익절 터치를 판정하고 체결가는 손절/익절 가격으로 가정합니다.
      같은 캔들에서 둘 다 터치되면 보수적으로 손절을 우선합니다.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df.index
    n = len(closes)
    entries_arr = np.asarray(entries, dtype=bool)
    exits_arr = np.asarray(exits, dtype=bool)

    def _to_arr(arr, scalar):
        if arr is not None:
            return np.asarray(arr, dtype=float)
        if scalar is not None:
            return np.full(n, float(scalar))
        return None

    sl_arr = _to_arr(sl_frac, sl_stop)
    tp_arr = _to_arr(tp_frac, tp_stop)

    in_pos = False
    entry_idx = 0
    entry_price = 0.0
    sl_price = np.nan
    tp_price = np.nan
    trade_records = []

    for i in range(n):
        price = closes[i]
        if not in_pos:
            if entries_arr[i]:
                in_pos = True
                entry_idx = i
                entry_price = price
                sl_price = entry_price * (1.0 - sl_arr[i]) if (sl_arr is not None and np.isfinite(sl_arr[i])) else np.nan
                tp_price = entry_price * (1.0 + tp_arr[i]) if (tp_arr is not None and np.isfinite(tp_arr[i])) else np.nan
        else:
            hit_sl = np.isfinite(sl_price) and lows[i] <= sl_price
            hit_tp = np.isfinite(tp_price) and highs[i] >= tp_price
            if hit_sl:
                trade_records.append((entry_idx, i, entry_price, sl_price, 1))
                in_pos = False
            elif hit_tp:
                trade_records.append((entry_idx, i, entry_price, tp_price, 1))
                in_pos = False
            elif exits_arr[i]:
                trade_records.append((entry_idx, i, entry_price, price, 1))
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


def build_sl_tp_fractions(df, mode="atr", sl_atr_mult=1.5, tp_atr_mult=3.0,
                          sl_pct=2.0, tp_pct=4.0, atr_col="ATR_14"):
    """
    봉별 손절/익절 폭(진입가 대비 비율, 레버리지 미적용 1배수 가격 기준)을 계산합니다.
    진입 캔들(t)의 값이 해당 포지션에 고정 적용됩니다.
    - ATR 모드: SL 폭 = sl_atr_mult × ATR_t / C_t,  TP 폭 = tp_atr_mult × ATR_t / C_t
        (손절가 = C_t - sl_atr_mult × ATR_t,  익절가 = C_t + tp_atr_mult × ATR_t)
    - 고정 % 모드: SL 폭 = sl_pct / 100,  TP 폭 = tp_pct / 100
    반환: (sl_frac, tp_frac) numpy 배열 (df 와 동일 길이)
    """
    n = len(df)
    if mode == "atr":
        if atr_col not in df.columns:
            raise KeyError(f"ATR 컬럼 '{atr_col}' 이(가) 데이터에 없습니다.")
        atr_ratio = (df[atr_col] / df['close']).values.astype(float)
        sl_frac = sl_atr_mult * atr_ratio
        tp_frac = tp_atr_mult * atr_ratio
    elif mode == "pct":
        sl_frac = np.full(n, sl_pct / 100.0)
        tp_frac = np.full(n, tp_pct / 100.0)
    else:
        raise ValueError("sl_tp_mode 는 'atr' 또는 'pct' 여야 합니다.")
    # 손절폭이 100% 이상이면 의미가 없으므로 상한 처리
    sl_frac = np.minimum(sl_frac, 0.999)
    return sl_frac, tp_frac


def classify_exit_reasons(trades_df, df, sl_frac, tp_frac):
    """각 거래의 청산 사유를 'SL'(손절) / 'TP'(익절) / 'Signal'(전략 청산 시그널) / 'Open'(미청산)으로 분류합니다."""
    n_trades = len(trades_df)
    reasons = np.full(n_trades, "Signal", dtype=object)
    if sl_frac is not None and tp_frac is not None and n_trades > 0:
        entry_pos = df.index.get_indexer(pd.DatetimeIndex(trades_df["entry_timestamp"]))
        ep = trades_df["entry_price"].values.astype(float)
        xp = trades_df["exit_price"].values.astype(float)
        sl_price = ep * (1.0 - sl_frac[entry_pos])
        tp_price = ep * (1.0 + tp_frac[entry_pos])
        # 가격은 소수 2자리로 반올림되어 있으므로 허용 오차를 둠
        is_sl = np.isclose(xp, sl_price, rtol=2e-6, atol=0.011) & (xp < ep)
        is_tp = np.isclose(xp, tp_price, rtol=2e-6, atol=0.011) & (xp > ep)
        reasons[is_tp] = "TP"
        reasons[is_sl] = "SL"
    if "status" in trades_df.columns:
        reasons[(trades_df["status"] == "Open").values] = "Open"
    return reasons


def compute_sharpe_ratio(equity_curve, bar_index, timeframe, risk_free=0.0):
    """
    연환산 샤프지수.
    - 거래 청산 시점마다 갱신되는 실현 잔고 곡선을 전체 캔들 인덱스로 전방채움(ffill)해 봉 단위 손익을 구하고,
      초기 자본금 대비 수익률 r 로 환산 (거래당 고정 명목금액 모델이므로 잔고가 음수여도 안정적)
    - Sharpe = (mean(r) - rf) / std(r) × sqrt(연간 봉 수)   (1h: 8,760 / 4h: 2,190, 표본표준편차 ddof=1)
    - 변동이 없거나 표본이 부족하면 0.0
    """
    periods_per_year = 8760.0 if timeframe == "1h" else 2190.0
    eq = equity_curve[~equity_curve.index.duplicated(keep="last")].sort_index()
    eq = eq.reindex(eq.index.union(bar_index)).ffill().reindex(bar_index).bfill()
    rets = eq.diff().dropna() / float(equity_curve.iloc[0])  # 초기 자본금 대비 봉별 손익률 (고정 명목금액 모델)
    if len(rets) < 2:
        return 0.0
    std = rets.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float((rets.mean() - risk_free / periods_per_year) / std * np.sqrt(periods_per_year))


def run_strategy(
    strategy: str = STRATEGY_BB_RSI,
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
    strict_rebound: bool = False,
    vol_mult: float = 2.0,
    squeeze_quantile: float = 0.2,
    use_sl_tp: bool = False,
    sl_tp_mode: str = "atr",
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
    sl_pct: float = 2.0,
    tp_pct: float = 4.0,
    atr_col: str = "ATR_14",
    verbose: bool = True
):
    """
    3종 유튜브 매매 전략 통합 백테스트 실행기.
    - strategy: 'bb_rsi'(볼린저+RSI 역추세) / 'ema_macd'(EMA200+MACD 골든크로스) / 'bb_squeeze'(스퀴즈+거래량 돌파)
    - timeframe: '1h' 또는 '4h'
    - use_sl_tp: True 이면 손절/익절 적용 (전략의 청산 시그널과 함께 먼저 발생한 쪽으로 청산)
        · sl_tp_mode='atr': 손절 = 진입가 - sl_atr_mult × ATR(진입봉), 익절 = 진입가 + tp_atr_mult × ATR(진입봉)
        · sl_tp_mode='pct': 손절 = 진입가 × (1 - sl_pct/100), 익절 = 진입가 × (1 + tp_pct/100)
    반환: timeframe, strategy, summary, trades, portfolio, data(지표 DF), equity_curve(Series)
    """
    tf = timeframe.lower().strip()
    if tf not in ("1h", "4h"):
        raise ValueError("timeframe 인자는 '1h' 또는 '4h'여야 합니다.")
    if strategy not in STRATEGY_LABELS:
        raise ValueError(f"지원하지 않는 전략입니다: {strategy} (가능: {list(STRATEGY_LABELS)})")

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

    if strategy == STRATEGY_BB_RSI:
        entries, exits = signals_bb_rsi(
            df, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought, strict_rebound=strict_rebound
        )
        strategy_title = (
            f"[전략1 BB+RSI 역추세 ({tf.upper()})] "
            f"BB({bb_length},{bb_std}) 하단터치 + RSI({rsi_length})≤{int(rsi_oversold)} 반등 진입 / "
            f"BB 상단터치 or RSI≥{int(rsi_overbought)} 청산 (레버리지 {leverage}x)"
        )
    elif strategy == STRATEGY_EMA_MACD:
        entries, exits = signals_ema_macd(df)
        strategy_title = (
            f"[전략2 EMA200+MACD 추세추종 ({tf.upper()})] "
            f"종가>EMA200 & MACD 골든크로스 진입 / MACD 데드크로스 or 종가<EMA200 청산 (레버리지 {leverage}x)"
        )
    else:
        entries, exits = signals_bb_squeeze(df, vol_mult=vol_mult, squeeze_quantile=squeeze_quantile)
        strategy_title = (
            f"[전략3 BB 스퀴즈+거래량 돌파 ({tf.upper()})] "
            f"스퀴즈(BW 하위 {int(squeeze_quantile * 100)}%) 직후 종가>BB상단 & 거래량≥{vol_mult}x 진입 / "
            f"종가<BB중단 청산 (레버리지 {leverage}x)"
        )

    # --- 리스크 관리: 손절(SL) / 익절(TP) 폭 산출 (진입가 대비 비율, 1배수 가격 기준) ---
    sl_frac = tp_frac = None
    if use_sl_tp:
        sl_frac, tp_frac = build_sl_tp_fractions(
            df, mode=sl_tp_mode, sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
            sl_pct=sl_pct, tp_pct=tp_pct, atr_col=atr_col
        )
        if sl_tp_mode == "atr":
            strategy_title += f" + ATR SL {sl_atr_mult}x / TP {tp_atr_mult}x"
        else:
            strategy_title += f" + 고정 SL {sl_pct}% / TP {tp_pct}%"

    if HAS_VBT:
        vbt_kwargs = {}
        if use_sl_tp:
            # 봉별 sl_stop/tp_stop 은 진입 캔들 값이 적용되며, 고가/저가 기준 장중 터치를 판정
            vbt_kwargs = dict(
                open=df['open'], high=df['high'], low=df['low'],
                sl_stop=pd.Series(sl_frac, index=df.index),
                tp_stop=pd.Series(tp_frac, index=df.index),
            )
        pf = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fee_rate,
            freq=tf,
            **vbt_kwargs
        )
        summary, trades_df = extract_backtest_metrics(
            df=df, pf=pf, init_cash=init_cash, leverage=leverage, fee_rate=fee_rate
        )
    else:
        summary, trades_df, pf = _simulate_signals_fallback(
            df=df, entries=entries, exits=exits,
            init_cash=init_cash, leverage=leverage, fee_rate=fee_rate,
            sl_frac=sl_frac, tp_frac=tp_frac
        )

    # 자산 증식 곡선: 시작 시점 초기 자본금 + 각 청산 시각의 누적 잔고
    if len(trades_df) > 0:
        equity_curve = pd.concat([
            pd.Series([init_cash], index=[df.index[0]]),
            pd.Series(trades_df["cumulative_equity"].values, index=pd.DatetimeIndex(trades_df["exit_timestamp"]))
        ])
        equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]
        trades_df["exit_reason"] = classify_exit_reasons(trades_df, df, sl_frac, tp_frac)
    else:
        equity_curve = pd.Series([init_cash, init_cash], index=[df.index[0], df.index[-1]])

    # 추가 성과 지표: 총수익률, 샤프지수, 프로핏 팩터, 청산 사유별 횟수
    summary["total_return_pct"] = (summary["net_profit"] / init_cash) * 100.0 if init_cash > 0 else 0.0
    summary["sharpe"] = compute_sharpe_ratio(equity_curve, df.index, tf)
    loss_abs = abs(summary["total_gross_loss"])
    summary["profit_factor"] = (summary["total_gross_profit"] / loss_abs) if loss_abs > 0 else (
        float("inf") if summary["total_gross_profit"] > 0 else 0.0
    )
    reasons = trades_df["exit_reason"] if "exit_reason" in trades_df.columns else pd.Series(dtype=object)
    summary["sl_exits"] = int((reasons == "SL").sum())
    summary["tp_exits"] = int((reasons == "TP").sum())
    summary["signal_exits"] = int((reasons == "Signal").sum())

    if verbose:
        print_backtest_report(strategy_title, summary, trades_df)

    return {
        "timeframe": tf,
        "strategy": strategy,
        "strategy_title": strategy_title,
        "summary": summary,
        "trades": trades_df,
        "portfolio": pf,
        "data": df,
        "equity_curve": equity_curve
    }


def run_youtube_bb_rsi_strategy(timeframe: str = "1h", **kwargs):
    """[하위 호환] 전략 1(볼린저 밴드 + RSI 역추세) 실행 래퍼."""
    return run_strategy(strategy=STRATEGY_BB_RSI, timeframe=timeframe, **kwargs)


def run_all_strategies(timeframe: str = "1h", df_input: pd.DataFrame = None, csv_path: str = None, **kwargs):
    """
    3대 유튜브 전략을 동일한 데이터/타임프레임/비용/SL·TP 조건으로 모두 실행합니다.
    반환: {strategy_key: run_strategy 결과 dict}
    (kwargs 는 run_strategy 의 파라미터를 그대로 전달하며, 리포트 출력은 기본적으로 생략합니다.)
    """
    kwargs.setdefault("verbose", False)
    return {
        key: run_strategy(strategy=key, timeframe=timeframe, df_input=df_input, csv_path=csv_path, **kwargs)
        for key in STRATEGY_LABELS
    }


def build_benchmark_table(results: dict) -> pd.DataFrame:
    """run_all_strategies() 결과를 전략별 성과 비교 표(DataFrame, 숫자형)로 변환합니다."""
    rows = []
    for key, res in results.items():
        s = res["summary"]
        rows.append({
            "전략": STRATEGY_LABELS[key],
            "총수익률(%)": s["total_return_pct"],
            "순손익($)": s["net_profit"],
            "승률(%)": s["win_rate"],
            "MDD(%)": s["mdd_pct"],
            "총수수료($)": s["total_fees"],
            "샤프지수": s["sharpe"],
            "프로핏팩터": s["profit_factor"],
            "거래횟수": s["total_trades"],
            "SL/TP/시그널 청산": f"{s['sl_exits']}/{s['tp_exits']}/{s['signal_exits']}",
        })
    return pd.DataFrame(rows).set_index("전략")


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
        for _strat in STRATEGY_LABELS:
            for _tf in ("1h", "4h"):
                run_strategy(strategy=_strat, timeframe=_tf, leverage=5.0)
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

        for _strat in STRATEGY_LABELS:
            for _tf in ("1h", "4h"):
                run_strategy(strategy=_strat, timeframe=_tf, df_input=synth_df, leverage=5.0)
