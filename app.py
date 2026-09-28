import streamlit as st
import plotly.express as px
from backtest_engine import run_advanced_backtest

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="퀀트 백테스팅 대시보드", layout="wide", page_icon="🚀")

st.title("📈 퀀트 백테스팅 엔진 대시보드")
st.markdown("**전략:** 주간 AVWAP 기준 트렌드 팔로잉 + 동적 레버리지 결합 모델 (1시간봉)")
st.caption("개발 환경: Python 3.12 | Vectorbt | Streamlit")

# 2. 실행 버튼 생성
if st.button("🚀 백테스트 엔진 구동", type="primary"):
    # 버튼을 누르면 로딩 스피너가 돌며 엔진 실행
    with st.spinner("과거 데이터 스캔 및 매매 시뮬레이션 중..."):
        # 만들어둔 엔진을 그대로 호출
        pf = run_advanced_backtest("BTC_USDT_1h.csv")
        stats = pf.stats()
        
        # 3. 최상단 핵심 지표 (KPI) 블록 생성
        st.subheader("📊 핵심 성과 지표 (KPI)")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("총 수익률", f"{stats['Total Return [%]']:.2f}%")
        col2.metric("승률 (Win Rate)", f"{stats['Win Rate [%]']:.2f}%")
        col3.metric("최대 낙폭 (MDD)", f"{stats['Max Drawdown [%]']:.2f}%")
        col4.metric("총 거래 횟수", f"{stats['Total Trades']} 회")
        
        st.divider()
        
        # 4. 수익금 누적 차트 (Equity Curve) 시각화
        st.subheader("💰 자산 가치 변화 차트 (Equity Curve)")
        
        # Vectorbt 자산 데이터를 Plotly 차트용으로 변환
        equity_df = pf.value().reset_index()
        equity_df.columns = ['시간', '자산(USDT)']
        
        # 인터랙티브 차트 생성
        fig_equity = px.line(equity_df, x='시간', y='자산(USDT)', template="plotly_dark")
        fig_equity.update_layout(xaxis_title="날짜", yaxis_title="포트폴리오 가치 ($)", hovermode="x unified")
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # 5. 하단 상세 통계표
        st.subheader("📋 상세 통계 리포트")
        st.dataframe(stats, use_container_width=True)
        