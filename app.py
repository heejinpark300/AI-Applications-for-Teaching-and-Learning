
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)

# 파일 경로
BASE = Path(__file__).resolve().parent

# 저장된 예측 모델
model = joblib.load(BASE / "trained_model.joblib")

# 웹앱 기본 설정
st.set_page_config(
    page_title="AI Feedback Acceptance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Feedback Acceptance Predictor")
st.subheader("AI 영어학습 피드백 수용 행동 예측")

st.info(
    "본 서비스는 합성 데이터 300건을 활용한 "
    "머신러닝 프로토타입입니다."
)

# 세 가지 탭
tab1, tab2, tab3 = st.tabs([
    "① 행동 예측",
    "② 모델 성능",
    "③ 변수 중요도"
])

labels = {
    0: "AI 피드백을 그대로 수용",
    1: "AI 피드백을 수정 후 사용",
    2: "AI 피드백을 거부"
}


# -----------------------------------
# 모델 평가 데이터 준비
# -----------------------------------
@st.cache_resource
def prepare_models():
    df = pd.read_csv(BASE / "sample_data.csv")

    X = df[
        ["trust_score", "english_level", "ai_experience"]
    ]
    y = df["decision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1-score": f1_score(
                y_test, pred, average="macro"
            )
        })

    results_df = pd.DataFrame(results)

    return (
        results_df,
        models["Random Forest"],
        X_test,
        y_test,
        X.columns
    )


# -----------------------------------
# 탭 1. 행동 예측
# -----------------------------------
with tab1:
    st.header("1. 학습자 정보 입력")

    trust = st.slider(
        "AI 답변을 얼마나 신뢰하나요?",
        min_value=0,
        max_value=100,
        value=70
    )

    level = st.selectbox(
        "영어 수준",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "초급",
            2: "중급",
            3: "고급"
        }[x]
    )

    experience = st.selectbox(
        "AI 영어학습 사용 경험",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "적음",
            2: "보통",
            3: "많음"
        }[x]
    )

    st.divider()
    st.header("2. AI 피드백 수용 행동 예측")

    if st.button("🔍 행동 예측하기", type="primary"):

        input_data = pd.DataFrame([{
            "trust_score": trust,
            "english_level": level,
            "ai_experience": experience
        }])

        prediction = int(model.predict(input_data)[0])

        st.success(
            f"예측 결과: {labels[prediction]}"
        )

        probabilities = model.predict_proba(
            input_data
        )[0]

        chart_data = pd.DataFrame({
            "행동 유형": [
                labels[int(c)] for c in model.classes_
            ],
            "예측 확률": probabilities
        })

        st.subheader("행동 유형별 예측 확률")
        st.bar_chart(
            chart_data.set_index("행동 유형")
        )


# -----------------------------------
# 탭 2. 모델 성능
# -----------------------------------
with tab2:
    st.header("머신러닝 모델 성능 비교")

    results_df, rf_model, X_test, y_test, features = (
        prepare_models()
    )

    st.subheader("1. 모델별 성능 비교")

    st.dataframe(
        results_df.round(4),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("2. Random Forest 혼동행렬")

    fig, ax = plt.subplots(figsize=(6, 4))

    ConfusionMatrixDisplay.from_estimator(
        rf_model,
        X_test,
        y_test,
        display_labels=["Accept", "Edit", "Reject"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Random Forest - Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------------
# 탭 3. 변수 중요도
# -----------------------------------
with tab3:
    st.header("Random Forest 변수 중요도")

    _, rf_model, _, _, features = prepare_models()

    importance = pd.Series(
        rf_model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    importance_df = pd.DataFrame({
        "변수": importance.index,
        "중요도": importance.values
    })

    st.subheader("1. 변수 중요도 수치")

    st.dataframe(
        importance_df.round(6),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("2. 변수 중요도 시각화")

    fig, ax = plt.subplots(figsize=(7, 4))

    importance.sort_values().plot(
        kind="barh",
        ax=ax,
        color="steelblue"
    )

    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest - Feature Importance")

    st.pyplot(fig)
    plt.close(fig)
