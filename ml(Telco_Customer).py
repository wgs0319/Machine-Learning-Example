import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# 1. 모델 & 스케일러 로드
# ===============================
with open("model/final_best_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="고객 이탈 예측 시스템", layout="centered")

st.title("📊 통신사 고객 이탈 예측 시스템")
st.write("고객 정보를 입력하면 **이탈 가능성(Churn Probability)**을 예측합니다.")

# ===============================
# 2. 사용자 입력 UI
# ===============================
st.subheader("👤 고객 정보 입력")

tenure = st.slider(
    "가입 기간 (개월)",
    min_value=0,
    max_value=72,
    value=12,
    step=1
)

monthly_charges = st.number_input(
    "월 요금 (Monthly Charges, $)",
    min_value=0.0,
    max_value=150.0,
    value=20.0,     # 기본값 20달러
    step=0.05       # 0.05달러 단위
)

senior = st.selectbox("고령자 여부", ["No", "Yes"])
partner = st.selectbox("배우자 여부", ["No", "Yes"])
dependents = st.selectbox("부양가족 여부", ["No", "Yes"])

contract = st.selectbox(
    "계약 유형",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "인터넷 서비스",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "결제 방식",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ===============================
# 3. 입력값 → DataFrame 변환
# ===============================
input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "Contract": [{
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }[contract]],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method]
})

# ===============================
# 4. One-Hot Encoding (학습과 동일)
# ===============================
input_df = pd.get_dummies(
    input_df,
    columns=["InternetService", "PaymentMethod"],
    drop_first=True
)

# 👉 학습 당시 컬럼과 맞추기
expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_cols]

# ===============================
# 5. 스케일링
# ===============================
input_df[["tenure", "MonthlyCharges"]] = scaler.transform(
    input_df[["tenure", "MonthlyCharges"]]
)

# ===============================
# 6. 예측
# ===============================
if st.button("🔍 이탈 예측 실행"):
    churn_prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 예측 결과")

    st.metric(
        label="이탈 확률",
        value=f"{churn_prob * 100:.2f}%"
    )

    if churn_prob >= 0.5:
        st.error("⚠️ 이 고객은 **이탈 가능성이 높습니다**")
    else:
        st.success("✅ 이 고객은 **이탈 가능성이 낮습니다**")
