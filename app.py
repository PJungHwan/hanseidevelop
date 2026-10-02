import streamlit as st
import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go

# --- 웹페이지 기본 세팅 ---
st.set_page_config(page_title="Quant Swing Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 순수 수학 지표 함수 ---
def get_atr(df, length):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def get_zlsma(series, length):
    lag = int((length - 1) / 2)
    data = series + (series - series.shift(lag))
    return data.ewm(span=length, adjust=False).mean()

# --- 사이드바 UI ---
st.sidebar.title("⚙️ 퀀트 엔진 설정")
st.sidebar.markdown("---")
leverage = st.sidebar.slider("레버리지 배율", min_value=1, max_value=50, value=50, step=1)
target_profit_pct = st.sidebar.number_input("목표 익절(TP) % (1배수 기준)", value=0.4, step=0.1) / 100.0

st.title("📈 1시간봉(1H) 스윙 퀀트 대시보드")
st.markdown(f"**현재 설정:** {leverage}배 레버리지 / 1H 캔들 타임프레임")

# --- 데이터 로드 및 지표 연산 ---
try:
    df = pd.read_csv("BTC_USDT_1h.csv", index_col='timestamp', parse_dates=True)
    
    with st.spinner('10,000개 캔들 수학 연산 및 타점 추출 중...'):
        len_A = 130
        df['ZLSMA'] = get_zlsma(df['close'], len_A)
        
        atr_A, mult_A = 22, 3
        df['ATR'] = get_atr(df, atr_A)
        df['CE_Long'] = df['high'].rolling(atr_A).max() - (df['ATR'] * mult_A)
        df['CE_Short'] = df['low'].rolling(atr_A).min() + (df['ATR'] * mult_A)
        
        # [전략 1] 종가 마감 확인
        buy_close = (df['close'] > df['ZLSMA']) & (df['close'] > df['CE_Short']) & (df['close'].shift(1) <= df['CE_Short'].shift(1))
        sell_close = (df['close'] < df['CE_Long']) & (df['close'].shift(1) >= df['CE_Long'].shift(1))
        entry_prc_close = df['close'].where(buy_close).ffill()
        entry_atr_close = df['ATR'].where(buy_close).ffill()
        sl_close = df['close'] < (entry_prc_close - (entry_atr_close * 1.5))
        exit_close = sell_close | sl_close

        # [전략 2] 실시간 고점 돌파
        buy_high = (df['high'] > df['ZLSMA'].shift(1)) & (df['high'] > df['CE_Short'].shift(1)) & (df['close'].shift(1) <= df['CE_Short'].shift(1))
        sell_low = (df['low'] < df['CE_Long'].shift(1)) & (df['close'].shift(1) >= df['CE_Long'].shift(1))
        entry_prc_high = df['close'].where(buy_high).ffill() 
        entry_atr_high = df['ATR'].where(buy_high).ffill()
        sl_high = df['close'] < (entry_prc_high - (entry_atr_high * 1.5))
        exit_high = sell_low | sl_high

        strategies = {
            "1. [정석] 종가 마감 진입": {'entries': buy_close, 'exits': exit_close},
            "2. [공격] 고점 돌파 진입": {'entries': buy_high, 'exits': exit_high}
        }

        fig = go.Figure()
        results_data = []

        for name, params in strategies.items():
            pf = vbt.Portfolio.from_signals(
                close=df['close'], entries=params['entries'], exits=params['exits'],
                tp_stop=target_profit_pct, init_cash=10000, fees=0.0004, freq='1h'
            )
            
            if pf.trades.count() > 0:
                returns_1x = pf.trades.returns.values
                returns_lev = returns_1x * leverage 
                # 청산 로직 (레버리지 적용 후 -100% 손실 시 원금 전액 손실)
                returns_lev = np.where(returns_lev <= -1.0, -1.0, returns_lev)
                
                winning_trades = np.sum(returns_lev > 0)
                win_rate = (winning_trades / len(returns_lev)) * 100
                
                trade_pnl = returns_lev * 10000
                cumulative_pnl = np.cumsum(trade_pnl)
                net_profit = np.sum(trade_pnl)
                
                fig.add_trace(go.Scatter(
                    y=cumulative_pnl, mode='lines+markers', 
                    name=f"{name} (승률: {win_rate:.1f}%)", marker=dict(size=4)
                ))
                
                results_data.append({
                    "전략명": name,
                    "진입 횟수": len(returns_lev),
                    "승률 (%)": f"{win_rate:.2f}%",
                    "순손익 ($)": f"${net_profit:,.2f}"
                })

        # UI 레이아웃 출력
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 누적 수익률 차트 비교")
            fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("📋 성과 영수증 요약")
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            st.error(
                "⚠️ **손익비 파탄 경고!**\n\n"
                "승률이 60%임에도 순손익이 마이너스입니다. "
                f"익절폭({target_profit_pct*100}%)이 너무 짧은 반면, 패배 시 {leverage}배 레버리지로 인해 한 번에 원금(-100%)을 모두 청산당하기 때문입니다."
            )

except FileNotFoundError:
    st.error("데이터 파일('BTC_USDT_1h.csv')을 찾을 수 없습니다. 먼저 터미널에서 데이터를 수집해 주세요.")