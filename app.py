import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_fetcher import fetch_extensive_binance_data
from backtest_engine import (
    load_timeframe_data,
    resample_ohlcv,
    run_strategy,
    run_all_strategies,
    build_benchmark_table
)
from strategies import (
    STRATEGY_BB_RSI,
    STRATEGY_EMA_MACD,
    STRATEGY_BB_SQUEEZE,
    STRATEGY_LABELS
)

# --- 웹페이지 기본 세팅 (TradingView Strategy Tester 스타일) ---
st.set_page_config(
    page_title="TradingView Strategy Tester Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TradingView 다크 테마 커스텀 CSS ---
st.markdown(
    """
    <style>
    .tv-kpi-grid {
        display: grid;
        grid-template-columns: repeat(7, minmax(120px, 1fr));
        gap: 10px;
        margin-bottom: 18px;
    }
    @media (max-width: 1200px) {
        .tv-kpi-grid {
            grid-template-columns: repeat(3, minmax(120px, 1fr));
        }
    }
    .tv-kpi-card {
        background: #131722;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    .tv-kpi-label {
        font-size: 12px;
        color: #787b86;
        margin-bottom: 6px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .tv-kpi-value {
        font-size: 20px;
        font-weight: 700;
        line-height: 1.2;
        color: #d1d4dc;
    }
    .tv-kpi-sub {
        font-size: 12px;
        margin-top: 4px;
        font-weight: 500;
    }
    .tv-pos { color: #089981 !important; }
    .tv-neg { color: #f23645 !important; }
    .tv-neu { color: #b2b5be !important; }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data(show_spinner=False)
def ensure_and_load_ohlcv(timeframe: str) -> pd.DataFrame:
    """
    선택된 타임프레임('1h' 또는 '4h') 데이터를 로드합니다.
    - 로컬에 해당 CSV 파일이 없으면:
      1) '4h' 선택 시 'BTC_USDT_1h.csv'가 존재하면 이를 4시간봉으로 자동 변환(리샘플링) 및 저장
      2) 그렇지 않으면 data_fetcher.fetch_extensive_binance_data()를 호출하여 바이낸스에서 자동 수집
    """
    tf = timeframe.lower().strip()
    target_csv = f"BTC_USDT_{tf}.csv"
    fallback_1h_csv = "BTC_USDT_1h.csv"

    if os.path.exists(target_csv):
        return pd.read_csv(target_csv, index_col="timestamp", parse_dates=True)

    if tf == "4h" and os.path.exists(fallback_1h_csv):
        df_1h = pd.read_csv(fallback_1h_csv, index_col="timestamp", parse_dates=True)
        df_4h = resample_ohlcv(df_1h, "4h")
        df_4h.to_csv(target_csv)
        return df_4h

    target_candles = 10000 if tf == "1h" else 3000
    fetch_extensive_binance_data(
        symbol="BTCUSDT",
        interval=tf,
        target_candles=target_candles,
        filename=target_csv
    )
    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 0:
        df_fetched = pd.read_csv(target_csv, index_col="timestamp", parse_dates=True)
        if len(df_fetched) > 0:
            return df_fetched

    return load_timeframe_data(timeframe=tf)


def compute_max_drawdown_usd(equity_curve: pd.Series, init_cash: float) -> float:
    """자산 증식 곡선으로부터 고점 대비 최대 손실폭 금액($)을 계산합니다."""
    if equity_curve is None or len(equity_curve) == 0:
        return 0.0
    eq_vals = np.concatenate([[init_cash], equity_curve.values.astype(float)])
    peaks = np.maximum.accumulate(eq_vals)
    dd_usd = eq_vals - peaks
    return float(np.min(dd_usd))


# ==============================================================================
# 1. 사이드바 컨트롤 패널 (캔들/보조지표 체크박스 완전 제거, 전략/비용/SL·TP 유지)
# ==============================================================================
st.sidebar.title("⚡ 전략 테스터 설정")
st.sidebar.markdown("---")

# [1-1] 매매 전략 선택 (3종)
st.sidebar.subheader("🎯 매매 전략 선택")
selected_strategy_label = st.sidebar.selectbox(
    "유튜브 매매 전략",
    options=list(STRATEGY_LABELS.values()),
    index=0,
    help="선택 즉시 해당 전략으로 백테스팅이 재실행됩니다."
)
selected_strategy = next(k for k, v in STRATEGY_LABELS.items() if v == selected_strategy_label)

st.sidebar.markdown("---")

# [1-2] 타임프레임 선택 (1h / 4h)
st.sidebar.subheader("🕒 타임프레임 설정")
tf_label = st.sidebar.selectbox(
    "분석 타임프레임 선택",
    options=["1시간봉 (1h)", "4시간봉 (4h)"],
    index=0,
    help="선택한 타임프레임 데이터가 로컬에 없으면 data_fetcher를 통해 자동 수집하거나 변환합니다."
)
selected_tf = "1h" if "1h" in tf_label else "4h"

if st.sidebar.button("🔄 바이낸스 최신 데이터 재수집", use_container_width=True):
    target_csv = f"BTC_USDT_{selected_tf}.csv"
    with st.spinner(f"바이낸스에서 {selected_tf.upper()} 데이터를 새로 수집하는 중..."):
        fetch_extensive_binance_data(
            symbol="BTCUSDT",
            interval=selected_tf,
            target_candles=10000 if selected_tf == "1h" else 3000,
            filename=target_csv
        )
        st.cache_data.clear()
    st.sidebar.success(f"{target_csv} 업데이트 완료!")

st.sidebar.markdown("---")

# [1-3] 백테스팅 비용 & 레버리지 파라미터
st.sidebar.subheader("💰 자본금 · 레버리지 · 수수료")
init_cash = st.sidebar.number_input(
    "초기 자본금 (USDT)",
    min_value=100.0,
    max_value=10000000.0,
    value=10000.0,
    step=1000.0,
    format="%.0f"
)
fee_pct = st.sidebar.number_input(
    "편도 수수료율 (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    format="%.2f",
    help="기본값 0.05% (바이낸스 선물 테이커 기준). 진입 및 청산 시 각각 부과됩니다."
)
fee_rate = fee_pct / 100.0

leverage = st.sidebar.slider(
    "레버리지 배율 (x)",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

rsi_oversold, rsi_overbought = st.sidebar.slider(
    "RSI 과매도 / 과매수 기준값",
    min_value=10,
    max_value=90,
    value=(30, 70),
    step=1,
    help="전략 1(BB+RSI)에서 RSI 과매도 진입 및 과매수 청산 기준으로 사용됩니다."
)

st.sidebar.markdown("---")

# [1-4] 리스크 관리 (SL / TP)
st.sidebar.subheader("🛡️ 리스크 관리 (SL / TP)")
use_sl_tp = st.sidebar.checkbox(
    "손절/익절(SL/TP) 적용",
    value=False,
    help="켜면 전략의 청산 시그널과 별개로 진입가 기준 SL/TP 가격에 먼저 닿는 쪽으로 청산됩니다."
)
sl_tp_mode_label = st.sidebar.radio(
    "SL/TP 산출 방식",
    options=["ATR 변동성 기반", "고정 % 기반"],
    index=0,
    disabled=not use_sl_tp,
    horizontal=True
)
sl_tp_mode = "atr" if sl_tp_mode_label.startswith("ATR") else "pct"
sl_atr_mult, tp_atr_mult, sl_pct, tp_pct = 1.5, 3.0, 2.0, 4.0
if sl_tp_mode == "atr":
    sl_atr_mult = st.sidebar.slider("손절 배수 (× ATR)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, disabled=not use_sl_tp)
    tp_atr_mult = st.sidebar.slider("익절 배수 (× ATR)", min_value=0.5, max_value=10.0, value=3.0, step=0.1, disabled=not use_sl_tp)
    st.sidebar.caption("손절가 = 진입가 − 손절배수 × ATR(14) / 익절가 = 진입가 + 익절배수 × ATR(14)")
else:
    sl_pct = st.sidebar.slider("손절 (%) - 1배수 가격 기준", min_value=0.5, max_value=20.0, value=2.0, step=0.5, disabled=not use_sl_tp)
    tp_pct = st.sidebar.slider("익절 (%) - 1배수 가격 기준", min_value=0.5, max_value=40.0, value=4.0, step=0.5, disabled=not use_sl_tp)
    st.sidebar.caption(f"레버리지 {leverage}x 적용 시 자본 대비 손절 ≈ {sl_pct * leverage:.1f}% / 익절 ≈ {tp_pct * leverage:.1f}%")

with st.sidebar.expander("🔧 전략 세부 파라미터 설정", expanded=False):
    bb_length = st.number_input("볼린저 밴드 기간 (Length)", min_value=5, max_value=100, value=20, step=1)
    bb_std = st.number_input("볼린저 밴드 표준편차 (Std)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    rsi_length = st.number_input("RSI 기간 (Length)", min_value=5, max_value=50, value=14, step=1)
    strict_rebound = st.checkbox("[전략1] 엄격 반등 확인 후 진입 (양봉 + RSI 상승)", value=False)
    vol_mult = st.number_input("[전략3] 거래량 급증 배수 (20봉 평균 대비)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    squeeze_quantile_pct = st.slider("[전략3] 스퀴즈 기준 밴드폭 하위 분위 (%)", min_value=5, max_value=50, value=20, step=5)
    squeeze_quantile = squeeze_quantile_pct / 100.0


# ==============================================================================
# 2. 데이터 로드 및 백테스팅 엔진 실행
# ==============================================================================
st.title("⚡ TradingView 스타일 전략 테스터 (초경량 고성능 모드)")
STRATEGY_DESCRIPTIONS = {
    STRATEGY_BB_RSI: (
        f"BB(`{bb_length}`, `{bb_std}`) 하단 터치 + RSI(`{rsi_length}`) ≤ `{rsi_oversold}` 반등 롱 진입 "
        f"/ BB 상단 터치 또는 RSI ≥ `{rsi_overbought}` 청산"
    ),
    STRATEGY_EMA_MACD: "종가 > EMA 200 상태에서 MACD 골든크로스 롱 진입 / MACD 데드크로스 또는 종가 < EMA 200 청산",
    STRATEGY_BB_SQUEEZE: (
        f"BB 밴드폭 스퀴즈(하위 `{int(squeeze_quantile * 100)}%`) 직후 종가 > BB 상단 & "
        f"거래량 ≥ 20봉 평균의 `{vol_mult}`배 돌파 롱 진입 / 종가 < BB 중단 청산"
    ),
}
if not use_sl_tp:
    risk_desc = "OFF"
elif sl_tp_mode == "atr":
    risk_desc = f"ATR SL `{sl_atr_mult}x` / TP `{tp_atr_mult}x`"
else:
    risk_desc = f"고정 SL `{sl_pct}%` / TP `{tp_pct}%`"

st.caption(
    f"**전략:** {STRATEGY_LABELS[selected_strategy]} ({STRATEGY_DESCRIPTIONS[selected_strategy]}) | "
    f"**타임프레임:** `{selected_tf.upper()}` | **자본금:** `${init_cash:,.0f}` | "
    f"**레버리지:** `{leverage}x` | **수수료:** `{fee_pct:.2f}%` | **SL/TP:** {risk_desc}"
)

try:
    with st.spinner(f"BTCUSDT ({selected_tf.upper()}) 전략 시뮬레이션 연산 중..."):
        raw_df = ensure_and_load_ohlcv(selected_tf)
        results = run_all_strategies(
            timeframe=selected_tf,
            df_input=raw_df,
            vol_mult=float(vol_mult),
            squeeze_quantile=float(squeeze_quantile),
            bb_length=int(bb_length),
            bb_std=float(bb_std),
            rsi_length=int(rsi_length),
            rsi_oversold=float(rsi_oversold),
            rsi_overbought=float(rsi_overbought),
            init_cash=float(init_cash),
            leverage=float(leverage),
            fee_rate=float(fee_rate),
            strict_rebound=bool(strict_rebound),
            use_sl_tp=bool(use_sl_tp),
            sl_tp_mode=sl_tp_mode,
            sl_atr_mult=float(sl_atr_mult),
            tp_atr_mult=float(tp_atr_mult),
            sl_pct=float(sl_pct),
            tp_pct=float(tp_pct)
        )
    bt_result = results[selected_strategy]

    tab1, tab2 = st.tabs(["📊 탭 1: 전략 테스터 리포트 (Overview & Trades)", "🏁 탭 2: 3대 유튜브 전략 벤치마크 비교"])

    with tab1:
        df = bt_result["data"]
        summary = bt_result["summary"]
        trades_df = bt_result["trades"]
        equity_curve = bt_result["equity_curve"]

        net_profit = float(summary["net_profit"])
        total_return_pct = (net_profit / init_cash) * 100.0 if init_cash > 0 else 0.0
        mdd_pct = float(summary["mdd_pct"])
        mdd_usd = compute_max_drawdown_usd(equity_curve, init_cash)
        total_trades = int(summary["total_trades"])
        win_trades = int(summary["winning_trades"])
        loss_trades = int(summary["losing_trades"])
        win_rate = float(summary["win_rate"])
        pf_val = float(summary["profit_factor"])
        pf_str = "∞" if np.isinf(pf_val) else f"{pf_val:.2f}"
        gross_profit = float(summary["total_gross_profit"])
        gross_loss = float(summary["total_gross_loss"])
        total_fees = float(summary["total_fees"])

        pnl_cls = "tv-pos" if net_profit >= 0 else "tv-neg"
        pf_cls = "tv-pos" if (np.isinf(pf_val) or pf_val >= 1.0) else "tv-neg"

        # ======================================================================
        # [1] 상단: 주요 통계 KPI 헤더 (TradingView 스타일 메트릭 카드)
        # ======================================================================
        kpi_html = f"""
        <div class="tv-kpi-grid">
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">총 PnL (Total PnL)</div>
                <div class="tv-kpi-value {pnl_cls}">${net_profit:+,.2f}</div>
                <div class="tv-kpi-sub {pnl_cls}">{total_return_pct:+.2f}%</div>
            </div>
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">최대 손실폭 (Max Drawdown)</div>
                <div class="tv-kpi-value tv-neg">-${abs(mdd_usd):,.2f}</div>
                <div class="tv-kpi-sub tv-neg">{mdd_pct:.2f}%</div>
            </div>
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">수익성 거래 (Win Rate)</div>
                <div class="tv-kpi-value">{win_rate:.2f}%</div>
                <div class="tv-kpi-sub tv-neu">{win_trades} / {total_trades} 거래</div>
            </div>
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">수익 팩터 (Profit Factor)</div>
                <div class="tv-kpi-value {pf_cls}">{pf_str}</div>
                <div class="tv-kpi-sub tv-neu">Sharpe {summary['sharpe']:.2f}</div>
            </div>
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">총수익 (Gross Profit)</div>
                <div class="tv-kpi-value tv-pos">+${gross_profit:,.2f}</div>
                <div class="tv-kpi-sub tv-pos">승리 {win_trades}건</div>
            </div>
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">총손실 (Gross Loss)</div>
                <div class="tv-kpi-value tv-neg">-${abs(gross_loss):,.2f}</div>
                <div class="tv-kpi-sub tv-neg">패배 {loss_trades}건</div>
            </div>
            <div class="tv-kpi-card">
                <div class="tv-kpi-label">총 수수료 (Total Fees)</div>
                <div class="tv-kpi-value">${total_fees:,.2f}</div>
                <div class="tv-kpi-sub tv-neu">편도 {fee_pct:.2f}% × {leverage}x</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ======================================================================
        # [2] 중단: 누적 성과 차트 (초경량 Equity Line + 건별 손익 Bar 결합)
        # ======================================================================
        st.subheader("📈 누적 성과 차트 (Equity Curve & Trade PnL)")
        if total_trades > 0:
            trade_nums = trades_df["trade_id"].values
            cum_pnl_series = trades_df["cumulative_equity"].values - init_cash
            trade_pnl_vals = trades_df["net_pnl"].values
            trade_ret_vals = trades_df["net_return_pct"].values
            exit_times_str = pd.to_datetime(trades_df["exit_timestamp"]).dt.strftime("%Y-%m-%d %H:%M").values

            eq_color = "#089981" if net_profit >= 0 else "#f23645"
            eq_fill = "rgba(8, 153, 129, 0.14)" if net_profit >= 0 else "rgba(242, 54, 69, 0.14)"
            bar_colors = np.where(trade_pnl_vals >= 0, "#089981", "#f23645")

            perf_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.65, 0.35],
                subplot_titles=(
                    "누적 PnL / 자산 증식 곡선 (USDT)",
                    "건별 실현 손익 (Trade PnL, USDT)"
                )
            )

            # Row 1: 누적 자산(Equity) 및 누적 PnL 단일 라인
            perf_fig.add_trace(
                go.Scatter(
                    x=np.concatenate([[0], trade_nums]),
                    y=np.concatenate([[init_cash], trades_df["cumulative_equity"].values]),
                    mode="lines",
                    name="누적 자산 (Equity)",
                    line=dict(color=eq_color, width=2.4),
                    fill="tozeroy",
                    fillcolor=eq_fill,
                    customdata=np.concatenate([
                        [["시작 시점", 0.0]],
                        np.stack([exit_times_str, cum_pnl_series], axis=-1)
                    ]),
                    hovertemplate=(
                        "<b>거래 #%{x}</b> (%{customdata[0]})<br>"
                        "계좌 잔고: $%{y:,.2f}<br>"
                        "누적 PnL: $%{customdata[1]:+,.2f}<extra></extra>"
                    )
                ),
                row=1, col=1
            )
            perf_fig.add_hline(
                y=init_cash, line_dash="dash", line_color="#787b86",
                annotation_text=f"초기 자본 (${init_cash:,.0f})", row=1, col=1
            )

            # Row 2: 건별 손익 Bar 차트 (Green: 이익, Red: 손실)
            perf_fig.add_trace(
                go.Bar(
                    x=trade_nums,
                    y=trade_pnl_vals,
                    name="건별 손익 ($)",
                    marker_color=bar_colors,
                    customdata=np.stack([exit_times_str, trade_ret_vals], axis=-1),
                    hovertemplate=(
                        "<b>거래 #%{x}</b> (%{customdata[0]})<br>"
                        "실현 손익: $%{y:+,.2f} (%{customdata[1]:+.2f}%)<extra></extra>"
                    )
                ),
                row=2, col=1
            )
            perf_fig.add_hline(y=0, line_color="#434651", line_width=1, row=2, col=1)

            perf_fig.update_xaxes(title_text="거래 번호 (Trade #)", row=2, col=1)
            perf_fig.update_yaxes(title_text="잔고 ($)", row=1, col=1)
            perf_fig.update_yaxes(title_text="건별 PnL ($)", row=2, col=1)
            perf_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131722",
                plot_bgcolor="#131722",
                height=480,
                hovermode="x unified",
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(perf_fig, use_container_width=True)
        else:
            st.info("거래 내역이 없어 누적 성과 차트를 표시할 수 없습니다.")

        st.markdown("---")

        # ======================================================================
        # [3] 거래 분석 섹션 (2열 분할: 좌측 수익/손실 분포 & 기대값 | 우측 도넛 차트)
        # ======================================================================
        st.subheader("🔍 거래 분석 (Trade Analysis)")
        if total_trades > 0:
            win_mask = trades_df["net_pnl"] > 0
            loss_mask = trades_df["net_pnl"] <= 0

            avg_win_pct = float(trades_df.loc[win_mask, "net_return_pct"].mean()) if win_trades > 0 else 0.0
            avg_loss_pct = float(trades_df.loc[loss_mask, "net_return_pct"].mean()) if loss_trades > 0 else 0.0
            avg_win_usd = float(trades_df.loc[win_mask, "net_pnl"].mean()) if win_trades > 0 else 0.0
            avg_loss_usd = float(trades_df.loc[loss_mask, "net_pnl"].mean()) if loss_trades > 0 else 0.0

            win_prob = win_trades / total_trades
            loss_prob = loss_trades / total_trades
            expectancy_pct = (win_prob * avg_win_pct) + (loss_prob * avg_loss_pct)
            expectancy_usd = float(trades_df["net_pnl"].mean())
            payoff_ratio = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct != 0 else 0.0

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("##### 📊 수익 / 손실 분포 및 기대값 (Expectancy)")
                c1, c2, c3 = st.columns(3)
                c1.metric("평균 수익 (Avg Win)", f"{avg_win_pct:+.2f}%", f"+${avg_win_usd:,.2f}")
                c2.metric("평균 손실 (Avg Loss)", f"{avg_loss_pct:.2f}%", f"-${abs(avg_loss_usd):,.2f}", delta_color="inverse")
                c3.metric("1회 기대값 (Expectancy)", f"{expectancy_pct:+.2f}%", f"${expectancy_usd:+,.2f} (손익비 {payoff_ratio:.2f})")

                dist_fig = go.Figure()
                dist_fig.add_trace(
                    go.Bar(
                        x=["평균 수익 (%)", "평균 손실 (%)", "기대값 (Expectancy %)"],
                        y=[avg_win_pct, abs(avg_loss_pct), expectancy_pct],
                        text=[f"+{avg_win_pct:.2f}%", f"{avg_loss_pct:.2f}% (절대폭)", f"{expectancy_pct:+.2f}%"],
                        textposition="auto",
                        marker_color=[
                            "#089981",
                            "#f23645",
                            "#089981" if expectancy_pct >= 0 else "#f23645"
                        ]
                    )
                )
                dist_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#131722",
                    plot_bgcolor="#131722",
                    height=260,
                    yaxis_title="비율 (%)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False
                )
                st.plotly_chart(dist_fig, use_container_width=True)

            with col_right:
                st.markdown("##### 🍩 거래 승/패 분포 (Win / Loss Donut)")
                donut_fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=["승자 (Winning)", "패자 (Losing)"],
                            values=[win_trades, loss_trades],
                            hole=0.64,
                            marker=dict(
                                colors=["#089981", "#f23645"],
                                line=dict(color="#131722", width=2)
                            ),
                            textinfo="label+percent",
                            hovertemplate="<b>%{label}</b><br>거래 수: %{value}건 (%{percent})<extra></extra>"
                        )
                    ]
                )
                donut_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#131722",
                    plot_bgcolor="#131722",
                    height=335,
                    margin=dict(l=20, r=20, t=20, b=20),
                    annotations=[
                        dict(
                            text=f"<b>총 {total_trades}건</b><br>승률 {win_rate:.1f}%",
                            x=0.5, y=0.5, font_size=15, showarrow=False, font_color="#d1d4dc"
                        )
                    ],
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(donut_fig, use_container_width=True)
        else:
            st.info("분석할 거래 데이터가 없습니다.")

        st.markdown("---")

        # ======================================================================
        # [4] 하단: 트레이딩뷰 스타일 '거래목록 (Trade List)' 테이블 & CSV 내보내기
        # ======================================================================
        header_col, btn_col = st.columns([3, 1])
        with header_col:
            st.subheader("📋 거래목록 (List of Trades)")
            st.caption(
                f"테스트 구간: `{df.index[0]}` ~ `{df.index[-1]}` | "
                f"총 **{total_trades:,}건** (포지션 명목 크기: `${init_cash * leverage:,.0f}` / {leverage}x)"
            )

        if total_trades > 0:
            notional_usd = init_cash * leverage
            pos_label = trades_df["direction"].map(lambda d: "롱 (Long)" if str(d).lower().startswith("long") else "숏 (Short)")
            btc_qty = np.round(notional_usd / trades_df["entry_price"].astype(float), 4)
            size_col = [f"{q:.4f} BTC (${notional_usd:,.0f})" for q in btc_qty]
            exit_reason_col = trades_df["exit_reason"] if "exit_reason" in trades_df.columns else "Signal"

            trade_list_df = pd.DataFrame({
                "거래 번호": trades_df["trade_id"].astype(int),
                "포지션": pos_label,
                "진입 시간": pd.to_datetime(trades_df["entry_timestamp"]).dt.strftime("%Y-%m-%d %H:%M"),
                "진입 가격": trades_df["entry_price"].astype(float),
                "청산 시간": pd.to_datetime(trades_df["exit_timestamp"]).dt.strftime("%Y-%m-%d %H:%M"),
                "청산 가격": trades_df["exit_price"].astype(float),
                "크기": size_col,
                "순손익($)": trades_df["net_pnl"].astype(float),
                "수익률(%)": trades_df["net_return_pct"].astype(float),
                "청산 사유": exit_reason_col
            })

            with btn_col:
                csv_bytes = trade_list_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 CSV 내보내기",
                    data=csv_bytes,
                    file_name=f"tradingview_trades_{selected_strategy}_{selected_tf}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            def _style_pnl(val):
                if isinstance(val, (int, float, np.number)):
                    if val > 0:
                        return "color: #089981; font-weight: 700;"
                    elif val < 0:
                        return "color: #f23645; font-weight: 700;"
                return ""

            styled_trades = (
                trade_list_df.style
                .map(_style_pnl, subset=["순손익($)", "수익률(%)"])
                .format({
                    "진입 가격": "${:,.2f}",
                    "청산 가격": "${:,.2f}",
                    "순손익($)": "{:+,.2f}",
                    "수익률(%)": "{:+.2f}%"
                })
            )
            st.dataframe(styled_trades, use_container_width=True, height=420, hide_index=True)
        else:
            st.warning("현재 설정된 파라미터 조건에서는 매매 진입 내역이 발생하지 않았습니다.")

    with tab2:
        st.subheader("🏁 3대 유튜브 매매 전략 한눈에 비교 (Benchmark)")
        any_df = results[selected_strategy]["data"]
        st.caption(
            f"**동일 조건 비교** — 타임프레임 `{selected_tf.upper()}` | 기간 `{any_df.index[0]}` ~ `{any_df.index[-1]}` | "
            f"초기 자본금 `${init_cash:,.0f}` | 수수료 `{fee_pct:.2f}%` | 레버리지 `{leverage}x` | SL/TP: {risk_desc}"
        )

        bench_df = build_benchmark_table(results)

        traded = bench_df[bench_df["거래횟수"] > 0]
        if len(traded) > 0:
            b1, b2, b3 = st.columns(3)
            b1.metric("🏆 최고 총수익률", traded["총수익률(%)"].idxmax().split(":")[0], f"{traded['총수익률(%)'].max():+.2f}%")
            b2.metric("📐 최고 샤프지수", traded["샤프지수"].idxmax().split(":")[0], f"{traded['샤프지수'].max():.2f}")
            b3.metric("🛡️ 최소 MDD", traded["MDD(%)"].idxmax().split(":")[0], f"{traded['MDD(%)'].max():.2f}%")

        show_df = bench_df.copy()
        show_df["프로핏팩터"] = show_df["프로핏팩터"].apply(lambda v: "∞" if np.isinf(v) else f"{v:.2f}")
        st.dataframe(
            show_df,
            use_container_width=True,
            column_config={
                "총수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "순손익($)": st.column_config.NumberColumn(format="$%.2f"),
                "승률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "MDD(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "총수수료($)": st.column_config.NumberColumn(format="$%.2f"),
                "샤프지수": st.column_config.NumberColumn(format="%.2f"),
                "거래횟수": st.column_config.NumberColumn(format="%d"),
            }
        )
        st.download_button(
            label="📥 벤치마크 비교표 CSV 다운로드",
            data=bench_df.to_csv().encode("utf-8-sig"),
            file_name=f"benchmark_BTCUSDT_{selected_tf}.csv",
            mime="text/csv"
        )

        st.subheader("📈 전략별 누적 수익률(%) 곡선 비교")
        cmp_fig = go.Figure()
        for key, res in results.items():
            eq = res["equity_curve"]
            cmp_fig.add_trace(
                go.Scatter(
                    x=eq.index,
                    y=(eq.values / init_cash - 1.0) * 100.0,
                    mode="lines",
                    line=dict(shape="hv", width=2),
                    name=STRATEGY_LABELS[key]
                )
            )
        cmp_fig.add_hline(y=0, line_dash="dash", line_color="#787b86")
        cmp_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131722",
            plot_bgcolor="#131722",
            height=400,
            hovermode="x unified",
            yaxis_title="누적 수익률 (%)",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(cmp_fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ 데이터를 불러오거나 백테스팅을 실행하는 중 오류가 발생했습니다: {e}")
    st.info("네트워크 연결 상태를 확인하거나 터미널에서 `python data_fetcher.py`를 실행해 데이터를 먼저 확보해 주세요.")
