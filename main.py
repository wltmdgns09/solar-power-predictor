import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="태양광 발전량 예측 실습실",
    page_icon="☀️",
    layout="wide"
)

WEATHER_URL = "https://raw.githubusercontent.com/wltmdgns09/solar-power-predictor/refs/heads/main/%ED%83%9C%EC%96%91%EA%B4%91_%EB%B0%9C%EC%A0%84%EC%98%88%EC%B8%A1_%EA%B8%B0%EC%83%81_2025_%EC%A7%80%EC%97%AD%EB%B3%84%EC%8B%9C%EA%B0%84%ED%8F%89%EA%B7%A0.csv"
POWER_URL = "https://raw.githubusercontent.com/wltmdgns09/solar-power-predictor/refs/heads/main/%ED%95%9C%EA%B5%AD%EC%A0%84%EB%A0%A5%EA%B1%B0%EB%9E%98%EC%86%8C_%EC%A7%80%EC%97%AD%EB%B3%84_%EC%8B%9C%EA%B0%84%EB%B3%84_%ED%83%9C%EC%96%91%EA%B4%91_%EB%B0%8F_%ED%92%8D%EB%A0%A5_%EB%B0%9C%EC%A0%84%EB%9F%89_UTF-8.csv"


@st.cache_data
def load_data():
    weather = pd.read_csv(WEATHER_URL)
    power = pd.read_csv(POWER_URL)

    # 기상 데이터 열 이름과 자료형 정리
    weather.columns = weather.columns.str.strip()
    power.columns = power.columns.str.strip()

    weather["날짜"] = pd.to_datetime(weather["BASIC_DT"].astype(str), errors="coerce")
    weather["시간"] = pd.to_numeric(weather["HH"], errors="coerce")
    weather["지역"] = weather["PLACE"].astype(str).str.strip()

    # 발전량 데이터에서 태양광 자료만 선택
    power = power[power["연료원"].astype(str).str.strip() == "태양광"].copy()
    power["날짜"] = pd.to_datetime(power["거래일"].astype(str), errors="coerce")
    power["시간"] = pd.to_numeric(power["거래시간"], errors="coerce") - 1
    power["지역"] = power["지역"].astype(str).str.strip()
    power["발전량"] = pd.to_numeric(power["전력거래량(MWh)"], errors="coerce")

    # 지역 이름 표기 차이를 일부 통일
    region_map = {
        "강원도": "강원", "경기도": "경기", "충청북도": "충북",
        "충청남도": "충남", "전라북도": "전북", "전라남도": "전남",
        "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
        "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
        "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
        "울산광역시": "울산", "세종특별자치시": "세종"
    }
    weather["지역"] = weather["지역"].replace(region_map)
    power["지역"] = power["지역"].replace(region_map)

    weather["시간"] = pd.to_numeric(weather["시간"], errors="coerce")
    power["시간"] = pd.to_numeric(power["시간"], errors="coerce")

    weather_cols = ["날짜", "시간", "지역", "GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP"]
    power_cols = ["날짜", "시간", "지역", "발전량"]

    for col in ["GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather[weather_cols].dropna(subset=["날짜", "시간", "지역"])
    power = power[power_cols].dropna(subset=["날짜", "시간", "지역", "발전량"])

    # 같은 날짜·시간·지역에 여러 행이 있으면 평균값으로 통일
    weather = weather.groupby(["날짜", "시간", "지역"], as_index=False)[
        ["GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP"]
    ].mean()
    power = power.groupby(["날짜", "시간", "지역"], as_index=False)["발전량"].mean()

    df = pd.merge(weather, power, on=["날짜", "시간", "지역"], how="inner")
    if df.empty:
        raise ValueError(
            "기상 데이터와 발전량 데이터가 결합되지 않았습니다. "
            "지역 이름 또는 거래시간(1~24와 0~23)의 기준을 확인해야 합니다."
        )

    df["월"] = df["날짜"].dt.month
    df["계절"] = df["월"].map({
        3: "봄", 4: "봄", 5: "봄",
        6: "여름", 7: "여름", 8: "여름",
        9: "가을", 10: "가을", 11: "가을",
        12: "겨울", 1: "겨울", 2: "겨울"
    })
    df["시간대"] = pd.cut(
        df["시간"],
        bins=[-1, 5, 10, 14, 18, 23],
        labels=["새벽", "오전", "정오 무렵", "오후", "저녁"]
    )
    return df.sort_values(["날짜", "시간", "지역"]).reset_index(drop=True)


st.title("☀️ 태양광 발전량 예측 실습실")
st.write("날씨 조건에 따라 태양광 발전량이 어떻게 달라지는지 데이터를 통해 알아봅니다.")

try:
    with st.spinner("GitHub에서 데이터를 불러오고 결합하는 중입니다..."):
        df = load_data()
except Exception as e:
    st.error("데이터를 불러오거나 결합하는 중 문제가 발생했습니다.")
    st.code(str(e))
    st.info(
        "GitHub에 두 CSV 파일이 올라가 있는지, 파일 이름과 Raw 주소가 정확한지 확인해 주세요. "
        "결합 데이터가 0개라면 지역 이름과 시간 기준도 확인해야 합니다."
    )
    st.stop()

page = st.sidebar.radio(
    "메뉴를 선택하세요",
    ["데이터 소개", "탐구 질문 분석", "발전량 예측"]
)

if page == "데이터 소개":
    st.header("1. 데이터 소개")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("결합된 데이터 수", f"{len(df):,}개")
    c2.metric("분석 지역 수", f"{df['지역'].nunique():,}개")
    c3.metric("발전량 0인 행", f"{(df['발전량'] == 0).sum():,}개")
    c4.metric("분석 변수 수", "6개")

    st.subheader("데이터 설명")
    st.write("""
    - 기상 데이터: 시간과 지역에 따른 일사량, 구름 관련 값, 기온 정보
    - 발전량 데이터: 한국전력거래소의 지역별 시간별 태양광 전력거래량
    - 분석 방법: 날짜·시간·지역을 기준으로 두 데이터를 결합한 뒤 관계를 분석
    """)
    st.warning(
        "전력거래량은 한국전력거래소 자료의 값이며, 국내 모든 태양광 설비의 "
        "전체 발전량과 반드시 같지는 않습니다."
    )
    st.subheader("결합 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("결측치 확인")
    st.dataframe(
        df.isna().sum().rename("결측치 개수").to_frame(),
        use_container_width=True
    )

elif page == "탐구 질문 분석":
    st.header("2. 탐구 질문 분석")
    question = st.selectbox("분석할 질문을 선택하세요", [
        "① 시간대별 태양광 발전량은 어떻게 달라질까?",
        "② 월별·계절별 발전량에는 어떤 차이가 있을까?",
        "③ 일사량과 발전량은 어떤 관계가 있을까?",
        "④ 기온과 구름량 중 발전량과 더 관련이 높은 변수는 무엇일까?",
        "⑤ 발전량이 0이거나 낮은 값은 언제 나타날까?"
    ])

    if question.startswith("①"):
        chart_data = df.groupby("시간대", observed=False)["발전량"].mean().reset_index()
        fig = px.bar(chart_data, x="시간대", y="발전량", title="시간대별 평균 태양광 발전량")
        st.plotly_chart(fig, use_container_width=True)
        st.write("시간대별 평균을 비교해 발전량이 높은 시간대를 확인합니다.")

    elif question.startswith("②"):
        tab1, tab2 = st.tabs(["월별", "계절별"])
        with tab1:
            monthly = df.groupby("월")["발전량"].mean().reset_index()
            st.plotly_chart(
                px.line(monthly, x="월", y="발전량", markers=True, title="월별 평균 발전량"),
                use_container_width=True
            )
        with tab2:
            seasonal = (
                df.groupby("계절", observed=False)["발전량"].mean()
                .reindex(["봄", "여름", "가을", "겨울"]).reset_index()
            )
            st.plotly_chart(
                px.bar(seasonal, x="계절", y="발전량", title="계절별 평균 발전량"),
                use_container_width=True
            )

    elif question.startswith("③"):
        sample = df.sample(min(len(df), 12000), random_state=42)
        fig = px.scatter(
            sample, x="GHI", y="발전량", opacity=0.45,
            title="일사량(GHI)과 태양광 발전량의 관계"
        )
        st.plotly_chart(fig, use_container_width=True)
        corr = df[["GHI", "발전량"]].corr().iloc[0, 1]
        st.metric("일사량과 발전량의 상관계수", f"{corr:.3f}")

    elif question.startswith("④"):
        corr = df[["AIR_TEMP", "CLOUD_OPACITY", "GHI", "발전량"]].corr()
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", aspect="auto", title="기상 변수와 발전량의 상관관계"),
            use_container_width=True
        )
        st.write(
            "상관계수의 절댓값이 클수록 두 변수 사이의 선형 관계가 강한 편입니다. "
            "상관관계가 인과관계를 뜻하지는 않습니다."
        )

    else:
        zero_count = int((df["발전량"] == 0).sum())
        low_threshold = df["발전량"].quantile(0.1)
        low = df[df["발전량"] <= low_threshold]
        c1, c2 = st.columns(2)
        c1.metric("발전량이 0인 행", f"{zero_count:,}개")
        c2.metric("하위 10% 기준 발전량", f"{low_threshold:.4f} MWh")
        by_hour = (
            df.assign(영발전=df["발전량"] == 0)
            .groupby("시간")["영발전"].mean().reset_index()
        )
        st.plotly_chart(
            px.bar(by_hour, x="시간", y="영발전", title="시간별 발전량 0인 데이터 비율"),
            use_container_width=True
        )
        st.write(
            "야간에는 일사량이 거의 없어 발전량이 0에 가까울 수 있습니다. "
            "따라서 0인 값을 무조건 오류로 판단하지 않고 시간과 날씨 조건을 함께 살펴봐야 합니다."
        )
        st.dataframe(low.head(20), use_container_width=True)

else:
    st.header("3. 태양광 발전량 예측")
    st.write("일사량, 기온, 구름 관련 값 등을 이용해 선형회귀 모델을 학습합니다.")
    features = ["GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP", "시간", "월"]
    model_df = (
        df[features + ["발전량", "날짜"]]
        .dropna()
        .sort_values("날짜")
    )

    if len(model_df) < 20:
        st.warning("예측 모델을 학습하기 위한 데이터가 충분하지 않습니다.")
        st.stop()

    split = int(len(model_df) * 0.8)
    train = model_df.iloc[:split]
    test = model_df.iloc[split:]
    X_train, y_train = train[features], train["발전량"]
    X_test, y_test = test[features], test["발전량"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("R² 점수", f"{r2:.3f}")
    st.caption("시간 순서에 따라 앞의 80%로 학습하고 뒤의 20%로 평가했습니다.")

    plot_df = pd.DataFrame({
        "실제 발전량": y_test.values,
        "예측 발전량": pred
    })
    st.plotly_chart(
        px.scatter(plot_df, x="실제 발전량", y="예측 발전량", title="실제 발전량과 예측 발전량 비교"),
        use_container_width=True
    )

    st.subheader("직접 조건을 입력해 발전량 예측하기")
    col1, col2, col3 = st.columns(3)
    with col1:
        ghi = st.number_input("일사량 GHI", min_value=0.0, value=float(df["GHI"].median()))
        dni = st.number_input("직달일사량 DNI", min_value=0.0, value=float(df["DNI"].median()))
        dhi = st.number_input("산란일사량 DHI", min_value=0.0, value=float(df["DHI"].median()))
    with col2:
        cloud = st.number_input(
            "구름 관련 값 CLOUD_OPACITY",
            min_value=0.0,
            value=float(df["CLOUD_OPACITY"].median())
        )
        temp = st.number_input("기온 AIR_TEMP (℃)", value=float(df["AIR_TEMP"].median()))
    with col3:
        hour = st.slider("시간 (0~23시)", 0, 23, 12)
        month = st.slider("월", 1, 12, 6)

    if st.button("발전량 예측하기", type="primary"):
        input_df = pd.DataFrame(
            [[ghi, dni, dhi, cloud, temp, hour, month]],
            columns=features
        )
        result = model.predict(input_df)[0]
        st.success(f"예측 발전량: {max(0, result):.3f} MWh")
        st.caption("선형회귀 모델의 예측값이며 실제 발전량과 차이가 있을 수 있습니다.")
