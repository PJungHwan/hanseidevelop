import streamlit as st
import pandas as pd
import plotly.express as px
from backtest_engine import run_strategy_benchmark

def main():
    # 1. 브라우저 탭 설정 및 전체 레이아웃 세팅
    st.set_page_config(page_title="퀀트 벤치마크 테스트베드", page_icon="📈", layout="wide")
    
    # 2. 메인 타이틀 및 프로젝트 설명
    st.title("📈 다중 퀀트 매매 전략 벤치마킹 테스트베드")
    st.markdown("""
    **[한세대 AI 졸업프로젝트 2팀]**  
    본 플랫폼은 특정 매매법에 종속된 단순 자동매매 봇이 아닙니다. 
    사용자가 원하는 다양한 퀀트 트레이딩 전략을 과거 데이터(비트코인 1h/4h)에 대입하여 
    **승률, 총 거래 횟수, 총 수익률을 객관적으로 검증하고 서열화**하는 범용 평가 엔진입니다.
    """)
    
    st.divider()
    
    # 3. 백테스트 구동 섹션
    st.subheader("💡 벤치마크 엔진 가동")
    st.markdown("아래 버튼을 누르면 4가지 주요 매매 전략이 동시에 과거 데이터를 관통하며 성과를 산출합니다.")
    
    # [수정] 최신 Streamlit 문법 적용: width="stretch"
    if st.button("🚀 전체 전략 백테스트 실행 (클릭)", width="stretch"):
        with st.spinner("퀀트 엔진 구동 중... 수천 번의 병렬 연산을 수행하고 있습니다. 잠시만 기다려주세요."):
            try:
                # 4. 백테스트 엔진에서 결과 데이터프레임(df) 받아오기
                res_df = run_strategy_benchmark("BTC_USDT_1h.csv", "BTC_USDT_4h.csv")
                st.success("✅ 백테스트 완료! 각 전략의 퍼포먼스 랭킹을 확인하세요.")
                
                # 5. 화면을 좌우 2분할하여 표와 차트 동시 배치
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 전략 성과 리더보드")
                    # [수정] 최신 Streamlit 문법 적용: width="stretch"
                    st.dataframe(
                        res_df.style.highlight_max(subset=['Total_Return(%)', 'Win_Rate(%)'], color='lightgreen'), 
                        width="stretch",
                        hide_index=True
                    )
                    
                with col2:
                    st.subheader("📊 수익률 비교 차트")
                    # Plotly를 이용한 반응형 막대 차트 생성
                    fig = px.bar(
                        res_df, 
                        x='Strategy_Name', 
                        y='Total_Return(%)',
                        color='Total_Return(%)',
                        color_continuous_scale='Viridis',
                        text_auto='.2f',
                        title="전략별 총 수익률(%) 비교"
                    )
                    fig.update_layout(xaxis_title="매매 전략", yaxis_title="총 수익률 (%)")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"백테스트 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()