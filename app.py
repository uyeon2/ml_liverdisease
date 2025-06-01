import streamlit as st
import joblib
import time

from ml_final_text import preprocess
from liver_advise import liver_feedback

model = joblib.load("liver_model_final.pkl")

st.set_page_config(                  #브라우저 탭의 이름설정
    page_title='간 질환 예측기',
    page_icon='🧬',
    layout='centered'
)

st.title("🧬 간 질환 위험도 예측기")

st.header('1️⃣ 건강 상태 입력')
user_input=st.text_area(
    "## 아래에 본인의 나이, 성별, 몸무게, 음주량, 흡연여부, 운동량, 당뇨, 고혈압, 유전력, 간수치 등 건강상태를 입력해보세요.",
    placeholder='''예: 저는 55세 남자입니다. 술은 주말에만 조금씩 마시고, 흡연은 하지 않습니다. 
    가족 중에 간 질환 병력이 있었고, 당뇨는 없습니다. 운동은 가끔 산책하는 정도입니다. 
    혈압은 정상이지만, 전반적으로 건강이 걱정돼서 한번 확인해보고 싶어요.
''',
height=200)
#placeholder은 회색으로 나타나는 예시문장

if st.button('예측하기'):
    if user_input.strip() == '':       #공백만 쳤을 수도 있으니까 확인용
        st.warning('건강 상태를 입력해주세요.')
    else:
        try:
            with st.spinner('간 질환 위험도 계산 중...'):
                input_vector=preprocess(user_input)
                probability = model.predict_proba(input_vector)[0][1]  # 양성일 확률
            st.markdown("---")
            st.header('2️⃣ 간 질환 위험도 결과')
            st.info(f"예측 결과: 당신은 {probability * 100:.1f}%의 확률로 간 질환이 있을 수 있습니다.")
            with st.spinner('건강 피드백 생성중...'):
                time.sleep(1.5)
                summary, tips = liver_feedback(probability)
            st.markdown("---")
            st.header('3️⃣ 사용자 맞춤 건강 피드백')
            st.subheader(summary)
            for t in tips:
                st.markdown(f"- {t}")
        except Exception as e:     #오류 떠도 프로그램이 멈추지 않고 대신 이렇게 처리해라
            st.error(f'예측 중 오류가 발생했습니다: {e}')  #에러 메시지를 e로 담아서 출력함
            
with st.sidebar:
    st.markdown("## ℹ️ 정보")   #write가 아닌 markdown으로 글씨체 조절 가능
    st.markdown("- 이 앱은 **자연어 입력 기반**으로 간 질환 위험도를 예측합니다.")
    st.markdown("- 입력 내용은 정규표현식으로 분석되어 **10가지 건강 요소**를 추출합니다.")
    st.markdown("- 결과는 머신러닝 모델(`CatBoost`) 기반으로 계산됩니다.")  #물결 밑에 있는 백틱으로 감싸면 인라인으로 표시됨
    st.markdown("- 이 앱은 참고용이며, **의학적 진단은 아닙니다.**")
	
    st.markdown("---")
    st.markdown("### 🧪 사용법 요약")
    st.markdown("1. 자신의 건강 상태를 자연어로 입력하세요.")
    st.markdown("2. [예측하기] 버튼을 누르면 결과가 표시됩니다.")
    
    st.markdown('---')
    st.markdown("### 👥 팀 정보")
    st.markdown("**Made by 남혜원, 이민하, 정가은**")
    st.markdown("*2025-1 Machine Learning Term Project*")