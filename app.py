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
    run_youtube_bb_rsi_strategy
)

# --- 웹페이지 기본 세팅 ---
st.set_page_config(
    page_title="Quant Backtesting Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
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

    # 로컬 파일이 전혀 없는 경우 data_fetcher를 통해 자동 수집 시도
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


# ==============================================================================
# 1. 사이드바 컨트롤 패널 (Sidebar Control Panel)
# ==============================================================================
st.sidebar.title("⚙️ 퀀트 백테스팅 컨트롤 패널")
st.sidebar.markdown("---")

# [1-1] 타임프레임 선택 드롭다운
st.sidebar.subheader("🕒 타임프레임 설정")
tf_label = st.sidebar.selectbox(
    "분석 타임프레임 선택",
    options=["1시간봉 (1h)", "4시간봉 (4h)"],
    index=0,
    help="선택한 타임프레임 데이터가 로컬에 없으면 data_fetcher를 통해 자동 수집하거나 변환합니다."
)
selected_tf = "1h" if "1h" in tf_label else "4h"

# 데이터 수동 갱신 버튼
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

# [1-2] 보조지표 표시 체크박스
st.sidebar.subheader("📐 차트 보조지표 표시")
show_ma = st.sidebar.checkbox("이동평균선 (SMA 20 / EMA 50)", value=True)
show_bb = st.sidebar.checkbox("볼린저 밴드 (BB Upper/Mid/Lower)", value=True)
show_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)", value=True)
show_macd = st.sidebar.checkbox("MACD (Line / Signal / Hist)", value=True)

st.sidebar.markdown("---")

# [1-3] 백테스팅 파라미터 조절
st.sidebar.subheader("💰 백테스팅 & 전략 파라미터")
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
    help="볼린저 하단 터치 + RSI 과매도 이하 반등 시 롱 진입 / 볼린저 상단 터치 또는 RSI 과매수 이상 도달 시 청산"
)

with st.sidebar.expander("🔧 세부 지표 주기 및 차트 표시 설정", expanded=False):
    bb_length = st.number_input("볼린저 밴드 기간 (Length)", min_value=5, max_value=100, value=20, step=1)
    bb_std = st.number_input("볼린저 밴드 표준편차 (Std)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    rsi_length = st.number_input("RSI 기간 (Length)", min_value=5, max_value=50, value=14, step=1)
    strict_rebound = st.checkbox("엄격 반등 확인 후 진입 (양봉 + RSI 상승)", value=False)
    max_chart_candles = st.slider(
        "차트 렌더링 최근 캔들 수",
        min_value=200,
        max_value=5000,
        value=1000,
        step=100,
        help="전체 기간 백테스트 후 차트 렌더링 속도를 위해 최근 N개 캔들과 해당 구간 타점을 표시합니다."
    )


# ==============================================================================
# 2. 데이터 로드 및 백테스팅 엔진 실행
# ==============================================================================
st.title("📊 퀀트 트레이딩 백테스팅 전용 대시보드")
st.caption(
    f"**적용 전략:** 볼린저 밴드(`{bb_length}`, `{bb_std}`) 하단 터치 + RSI(`{rsi_length}`) ≤ `{rsi_oversold}` 반등 롱(Long) 진입 "
    f"/ 볼린저 상단 터치 또는 RSI ≥ `{rsi_overbought}` 청산 | "
    f"**타임프레임:** `{selected_tf.upper()}` | **초기 자본금:** `${init_cash:,.0f}` | "
    f"**수수료율:** `{fee_pct:.2f}%` | **레버리지:** `{leverage}x`"
)

try:
    with st.spinner(f"BTCUSDT ({selected_tf.upper()}) 데이터 준비 및 백테스팅 시뮬레이션 연산 중..."):
        raw_df = ensure_and_load_ohlcv(selected_tf)
        bt_result = run_youtube_bb_rsi_strategy(
            timeframe=selected_tf,
            df_input=raw_df,
            bb_length=int(bb_length),
            bb_std=float(bb_std),
            rsi_length=int(rsi_length),
            rsi_oversold=float(rsi_oversold),
            rsi_overbought=float(rsi_overbought),
            init_cash=float(init_cash),
            leverage=float(leverage),
            fee_rate=float(fee_rate),
            strict_rebound=bool(strict_rebound)
        )

    df = bt_result["data"]
    summary = bt_result["summary"]
    trades_df = bt_result["trades"]

    total_return_pct = (summary["net_profit"] / init_cash) * 100.0 if init_cash > 0 else 0.0

    # ==========================================================================
    # 3. 상단 요약 카드 (KPI Metrics Row)
    # ==========================================================================
    st.subheader("📌 백테스팅 핵심 성과 요약 (Performance Metrics)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric(
        label="총 수익률 (Total Return)",
        value=f"{total_return_pct:+.2f}%",
        delta=f"순손익 ${summary['net_profit']:+,.2f}"
    )
    m2.metric(
        label="총 수익금 (Gross Profit)",
        value=f"+${summary['total_gross_profit']:,.2f}",
        delta=f"승리 {summary['winning_trades']}회"
    )
    m3.metric(
        label="총 손실금 (Gross Loss)",
        value=f"-${abs(summary['total_gross_loss']):,.2f}",
        delta=f"패배 {summary['losing_trades']}회",
        delta_color="inverse"
    )
    m4.metric(
        label="총 지불 수수료 (Fees)",
        value=f"${summary['total_fees']:,.2f}",
        delta=f"편도 {fee_pct:.2f}% × {leverage}x",
        delta_color="off"
    )
    m5.metric(
        label="승률 (Win Rate)",
        value=f"{summary['win_rate']:.2f}%",
        delta=f"총 {summary['total_trades']}회 진입",
        delta_color="off"
    )
    m6.metric(
        label="최대 낙폭 (MDD)",
        value=f"{summary['mdd_pct']:.2f}%",
        delta=f"최종자산 ${summary['final_equity']:,.2f}",
        delta_color="inverse"
    )

    st.markdown("---")

    # ==========================================================================
    # 4. 메인 캔들차트 (진입/청산 마커 + BB/MA 오버레이) & 서브 차트 (RSI / MACD)
    # ==========================================================================
    st.subheader("📈 인터랙티브 캔들차트 & 매매 타점 / 보조지표 연동 뷰")

    plot_df = df.tail(max_chart_candles).copy()
    plot_start_ts = plot_df.index[0]

    # 활성화된 서브 차트 개수 동적 계산
    sub_panels = []
    if show_rsi:
        sub_panels.append("RSI")
    if show_macd:
        sub_panels.append("MACD")

    num_rows = 1 + len(sub_panels)
    if num_rows == 3:
        row_heights = [0.58, 0.21, 0.21]
    elif num_rows == 2:
        row_heights = [0.72, 0.28]
    else:
        row_heights = [1.0]

    subplot_titles = [f"BTC/USDT ({selected_tf.upper()}) 캔들차트 & 매매 타점"]
    for panel in sub_panels:
        if panel == "RSI":
            subplot_titles.append(f"RSI ({rsi_length}) [과매도: {rsi_oversold} / 과매수: {rsi_overbought}]")
        elif panel == "MACD":
            subplot_titles.append("MACD (12, 26, 9)")

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # [Row 1] 메인 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="BTC/USDT",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # [Row 1 오버레이] 이동평균선 (SMA 20 / EMA 50)
    if show_ma:
        sma_col = next((c for c in plot_df.columns if c.startswith("SMA_")), None)
        ema_col = next((c for c in plot_df.columns if c.startswith("EMA_")), None)
        if sma_col:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[sma_col],
                    mode="lines",
                    name=sma_col,
                    line=dict(color="#ffca28", width=1.5)
                ),
                row=1, col=1
            )
        if ema_col:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[ema_col],
                    mode="lines",
                    name=ema_col,
                    line=dict(color="#29b6f6", width=1.5)
                ),
                row=1, col=1
            )

    # [Row 1 오버레이] 볼린저 밴드 (상단, 중단, 하단)
    if show_bb and {"BB_Upper", "BB_Middle", "BB_Lower"}.issubset(plot_df.columns):
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["BB_Upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="rgba(171, 71, 188, 0.75)", width=1.2, dash="dot")
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["BB_Lower"],
                mode="lines",
                name="BB Lower",
                fill="tonexty",
                fillcolor="rgba(171, 71, 188, 0.08)",
                line=dict(color="rgba(171, 71, 188, 0.75)", width=1.2, dash="dot")
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["BB_Middle"],
                mode="lines",
                name="BB Middle",
                line=dict(color="rgba(206, 147, 216, 0.6)", width=1.0, dash="dash")
            ),
            row=1, col=1
        )

    # [Row 1 오버레이] 롱 진입(초록색 위 화살표) & 청산(X 마커) 타점 시각화
    if len(trades_df) > 0:
        visible_entries = trades_df[trades_df["entry_timestamp"] >= plot_start_ts]
        visible_exits = trades_df[trades_df["exit_timestamp"] >= plot_start_ts]

        if len(visible_entries) > 0:
            fig.add_trace(
                go.Scatter(
                    x=visible_entries["entry_timestamp"],
                    y=visible_entries["entry_price"] * 0.996,
                    mode="markers",
                    name="Long 진입 (Entry)",
                    marker=dict(
                        symbol="triangle-up",
                        size=13,
                        color="#00e676",
                        line=dict(color="#ffffff", width=1)
                    ),
                    customdata=np.stack([
                        visible_entries["trade_id"],
                        visible_entries["entry_price"]
                    ], axis=-1),
                    hovertemplate=(
                        "<b>🟢 Long 진입 (#%{customdata[0]})</b><br>"
                        "시각: %{x}<br>"
                        "진입 단가: $%{customdata[1]:,.2f}<extra></extra>"
                    )
                ),
                row=1, col=1
            )

        if len(visible_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=visible_exits["exit_timestamp"],
                    y=visible_exits["exit_price"] * 1.004,
                    mode="markers",
                    name="포지션 청산 (Exit)",
                    marker=dict(
                        symbol="x",
                        size=11,
                        color="#ff5252",
                        line=dict(color="#ffffff", width=1)
                    ),
                    customdata=np.stack([
                        visible_exits["trade_id"],
                        visible_exits["exit_price"],
                        visible_exits["net_return_pct"],
                        visible_exits["net_pnl"],
                        visible_exits["fee_paid"]
                    ], axis=-1),
                    hovertemplate=(
                        "<b>❌ 포지션 청산 (#%{customdata[0]})</b><br>"
                        "시각: %{x}<br>"
                        "청산 단가: $%{customdata[1]:,.2f}<br>"
                        "순수익률: %{customdata[2]:+.2f}%<br>"
                        "순손익금: $%{customdata[3]:+,.2f} (수수료: $%{customdata[4]:,.2f})<extra></extra>"
                    )
                ),
                row=1, col=1
            )

    # [서브 차트 연동] RSI 및 MACD
    current_row = 2
    if show_rsi and "RSI" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["RSI"],
                mode="lines",
                name=f"RSI ({rsi_length})",
                line=dict(color="#b388ff", width=1.6)
            ),
            row=current_row, col=1
        )
        fig.add_hline(
            y=rsi_overbought, line_dash="dash", line_color="#ff5252",
            annotation_text=f"과매수 ({rsi_overbought})", row=current_row, col=1
        )
        fig.add_hline(
            y=rsi_oversold, line_dash="dash", line_color="#00e676",
            annotation_text=f"과매도 ({rsi_oversold})", row=current_row, col=1
        )
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    if show_macd and {"MACD_Line", "MACD_Signal", "MACD_Hist"}.issubset(plot_df.columns):
        hist_colors = np.where(plot_df["MACD_Hist"] >= 0, "#26a69a", "#ef5350")
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df["MACD_Hist"],
                name="MACD Hist",
                marker_color=hist_colors
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["MACD_Line"],
                mode="lines",
                name="MACD Line",
                line=dict(color="#42a5f5", width=1.4)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["MACD_Signal"],
                mode="lines",
                name="Signal Line",
                line=dict(color="#ffa726", width=1.4)
            ),
            row=current_row, col=1
        )

    chart_height = 550 if num_rows == 1 else (720 if num_rows == 2 else 880)
    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 누적 자산 곡선 (Equity Curve) 확장 탭
    with st.expander("💹 누적 계좌 잔고 곡선 (Equity Curve) 보기", expanded=False):
        if len(trades_df) > 0:
            eq_fig = go.Figure()
            eq_fig.add_trace(
                go.Scatter(
                    x=trades_df["exit_timestamp"],
                    y=trades_df["cumulative_equity"],
                    mode="lines+markers",
                    name="누적 자산 (USDT)",
                    line=dict(color="#00e676", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 230, 118, 0.08)"
                )
            )
            eq_fig.add_hline(y=init_cash, line_dash="dash", line_color="#9e9e9e", annotation_text="초기 자본금")
            eq_fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(eq_fig, use_container_width=True)
        else:
            st.info("거래 내역이 없어 누적 자산 곡선을 표시할 수 없습니다.")

    st.markdown("---")

    # ==========================================================================
    # 5. 하단 상세 매매 기록 (Trade Log 테이블 및 CSV 다운로드 버튼)
    # ==========================================================================
    st.subheader("📋 상세 매매 기록 (Trade Log)")

    if len(trades_df) > 0:
        display_log = trades_df.rename(columns={
            "trade_id": "거래번호",
            "entry_timestamp": "진입 시각 (Entry Time)",
            "exit_timestamp": "청산 시각 (Exit Time)",
            "direction": "포지션",
            "entry_price": "진입가 (USDT)",
            "exit_price": "청산가 (USDT)",
            "gross_return_pct": "수수료전 수익률 (%)",
            "fee_paid": "지불 수수료 (USDT)",
            "net_return_pct": "최종 수익률 (%)",
            "net_pnl": "실현 손익금 (USDT)",
            "cumulative_equity": "누적 잔고 (USDT)",
            "status": "상태"
        })

        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(
                f"총 **{len(trades_df):,}건**의 매매가 기록되었습니다. "
                f"(전체 테스트 구간: `{df.index[0]}` ~ `{df.index[-1]}`)"
            )
        with col_btn:
            csv_bytes = trades_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Trade Log CSV 다운로드",
                data=csv_bytes,
                file_name=f"trade_log_BTCUSDT_{selected_tf}_lev{leverage}x.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(display_log, use_container_width=True, height=400)
    else:
        st.warning("현재 설정된 파라미터 조건에서는 매매 진입 내역이 발생하지 않았습니다. RSI 기준값 등을 조정해 보세요.")

except Exception as e:
    st.error(f"⚠️ 데이터를 불러오거나 백테스팅을 실행하는 중 오류가 발생했습니다: {e}")
    st.info("네트워크 연결 상태를 확인하거나 터미널에서 `python data_fetcher.py`를 실행해 데이터를 먼저 확보해 주세요.")
