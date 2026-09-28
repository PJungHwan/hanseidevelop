import streamlit as st
import pandas as pd
import plotly.express as px
from backtest_engine import run_strategy_benchmark
from data_fetcher import fetch_binance_data

def main():
    # 1. 브라우저 탭 설정 및 전체 레이아웃 세팅
    st.set_page_config(page_title="실시간 퀀트 테스트베드", page_icon="📈", layout="wide")
    
    # 2. 메인 타이틀 및 프로젝트 설명
    st.title("📈 실시간 다중 퀀트 매매 전략 벤치마킹 플랫폼")
    st.markdown("""
    **[한세대 AI 졸업프로젝트 2팀]**  
    본 플랫폼은 실시간 바이낸스(Binance) 데이터를 원클릭으로 즉각 수집하여, 
    사용자가 정의한 다양한 퀀트 전략들의 현재 장세 기준 승률과 수익률을 
    객관적으로 검증하고 서열화하는 올인원(All-in-One) 퀀트 엔진입니다.
    """)
    
    st.divider()
    
    # 3. 자동화 엔진 구동 섹션
    st.subheader("💡 실시간 벤치마크 엔진 가동")
    st.markdown("아래 버튼을 누르면 **최신 시장 데이터를 다운로드**한 뒤, 즉시 **다중 전략 백테스트**를 연속 수행하여 결과를 산출합니다.")
    
    # [핵심 통합] 데이터 수집 + 백테스트 동시 실행 버튼
    if st.button("📥 실시간 데이터 갱신 및 전체 전략 백테스트 실행 (클릭)", width="stretch"):
        
        # 1단계: 실시간 데이터 파이프라인 가동
        with st.spinner("1단계: 바이낸스 실시간 시세 파이프라인 가동 중..."):
            try:
                fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000, filename="BTC_USDT_1h.csv")
                fetch_binance_data(symbol="BTCUSDT", interval="4h", limit=1000, filename="BTC_USDT_4h.csv")
                st.toast("✅ 실시간 시장 데이터 동기화 완료!", icon="📥")
            except Exception as e:
                st.error(f"데이터 수집 중 오류가 발생했습니다: {e}")
                return # 데이터 수집 실패 시 여기서 멈춤
                
        # 2단계: 퀀트 엔진 병렬 연산 가동
        with st.spinner("2단계: 퀀트 벤치마크 엔진 수천 번의 병렬 연산 중... 잠시만 기다려주세요."):
            try:
                res_df = run_strategy_benchmark("BTC_USDT_1h.csv", "BTC_USDT_4h.csv")
                st.success("✅ 모든 전략의 실시간 장세 벤치마킹이 완료되었습니다!")
                
                # 3단계: 화면 시각화 (좌우 분할)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 실시간 전략 성과 리더보드")
                    st.dataframe(
                        res_df.style.highlight_max(subset=['Total_Return(%)', 'Win_Rate(%)'], color='lightgreen'), 
                        width="stretch",
                        hide_index=True
                    )
                    
                with col2:
                    st.subheader("📊 실시간 수익률 비교 차트")
                    fig = px.bar(
                        res_df, 
                        x='Strategy_Name', 
                        y='Total_Return(%)',
                        color='Total_Return(%)',
                        color_continuous_scale='Viridis',
                        text_auto='.2f',
                        title="전략별 총 수익률(%) 비교 (최신 데이터 기준)"
                    )
                    fig.update_layout(xaxis_title="매매 전략", yaxis_title="총 수익률 (%)")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"백테스트 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()