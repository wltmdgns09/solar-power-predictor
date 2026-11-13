import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="태양광 발전량 탐구실",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

WEATHER_URL = "https://raw.githubusercontent.com/wltjsdn09/solar-power-predictor/refs/heads/main/%ED%83%9C%EC%96%91%EA%B4%91_%EB%B0%9C%EC%A0%84%EC%98%88%EC%B8%A1_%EA%B8%B0%EC%83%81_2025_%EC%A7%80%EC%97%AD%EB%B3%84%EC%8B%9C%EA%B0%84%ED%8F%89%EA%B7%A0.csv"
POWER_URL = "https://raw.githubusercontent.com/wltjsdn09/solar-power-predictor/refs/heads/main/%ED%95%9C%EA%B5%AD%EC%A0%84%EB%A0%A5%EA%B1%B0%EB%9E%98%EC%86%8C_%EC%A7%80%EC%97%AD%EB%B3%84_%EC%8B%9C%EA%B0%84%EB%B3%84_%ED%83%9C%EC%96%91%EA%B4%91_%EB%B0%8F_%ED%92%8D%EB%A0%A5_%EB%B0%9C%EC%A0%84%EB%9F%89_UTF-8.csv"

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #f5f9ff 0%, #ffffff 42%);}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(115deg,#123b66,#1976a5); color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin-bottom: .35rem;}
.hero p {color: #e8f4ff; margin-bottom: 0;}
div[data-testid="stMetric"] {background: white; border: 1px solid #e6edf5; padding: 14px 16px; border-radius: 14px; box-shadow: 0 2px 8px #15385b0a;}
div[data-testid="stTabs"] button {font-weight: 650;}
.note {padding: 12px 15px; border-left: 4px solid #2692bd; background: #eef7fc; border-radius: 8px; margin: 8px 0 16px;}
.small-muted {color: #62748a; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    weather = pd.read_csv(WEATHER_URL)
    power = pd.read_csv(POWER_URL)
    weather.columns = weather.columns.str.strip()
    power.columns = power.columns.str.strip()

    required_w = {"BASIC_DT", "HH", "PLACE", "GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP"}
    required_p = {"거래일", "거래시간", "지역", "연료원", "전력거래량(MWh)"}
    if not required_w.issubset(weather.columns):
        raise ValueError(f"기상 CSV 열 이름을 확인해 주세요. 필요한 열: {sorted(required_w)}")
    if not required_p.issubset(power.columns):
        raise ValueError(f"발전량 CSV 열 이름을 확인해 주세요. 필요한 열: {sorted(required_p)}")

    weather["날짜"] = pd.to_datetime(weather["BASIC_DT"].astype(str), errors="coerce")
    weather["시간"] = pd.to_numeric(weather["HH"], errors="coerce")
    weather["지역"] = weather["PLACE"].astype(str).str.strip()

    power = power[power["연료원"].astype(str).str.strip() == "태양광"].copy()
    power["날짜"] = pd.to_datetime(power["거래일"].astype(str), errors="coerce")
    # 원본 거래시간이 1~24이고 1이 00시를 뜻한다는 가정으로 1을 뺍니다.
    power["시간"] = pd.to_numeric(power["거래시간"], errors="coerce") - 1
    power["지역"] = power["지역"].astype(str).str.strip()
    power["발전량"] = pd.to_numeric(power["전력거래량(MWh)"], errors="coerce")

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

    weather_vars = ["GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP"]
    for col in weather_vars:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather[["날짜", "시간", "지역"] + weather_vars].dropna(subset=["날짜", "시간", "지역"])
    power = power[["날짜", "시간", "지역", "발전량"]].dropna()
    weather = weather.groupby(["날짜", "시간", "지역"], as_index=False)[weather_vars].mean()
    power = power.groupby(["날짜", "시간", "지역"], as_index=False)["발전량"].mean()

    data = pd.merge(weather, power, on=["날짜", "시간", "지역"], how="inner")
    if data.empty:
        raise ValueError("두 CSV를 결합한 결과가 0행입니다. 지역 이름과 시간 기준(기상 0~23시, 발전량 거래시간 1~24)을 확인해야 합니다.")

    data["월"] = data["날짜"].dt.month
    data["계절"] = data["월"].map({
        3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
        9: "가을", 10: "가을", 11: "가을", 12: "겨울", 1: "겨울", 2: "겨울"
    })
    data["시간대"] = pd.cut(
        data["시간"], bins=[-1, 5, 10, 14, 18, 23],
        labels=["새벽", "오전", "정오 무렵", "오후", "저녁"]
    )
    return data.sort_values(["날짜", "시간", "지역"]).reset_index(drop=True)


def explain_corr(value):
    if pd.isna(value):
        return "상관계수를 계산할 수 없습니다."
    strength = "강한" if abs(value) >= 0.7 else "중간 정도의" if abs(value) >= 0.4 else "약한"
    direction = "양의" if value > 0 else "음의" if value < 0 else "거의 없는"
    return f"발전량과 {strength} {direction} 상관관계가 관찰됩니다 (상관계수 {value:.2f})."


st.markdown("""
<div class="hero">
  <h1>☀️ 날씨에 따른 태양광 발전량 탐구실</h1>
  <p>기상 데이터와 발전량 데이터를 연결해 발전량의 변화 원인을 살펴보고 예측해 봅니다.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("탐구 메뉴")
    page = st.radio(
        "원하는 항목을 선택하세요",
        ["한눈에 보기", "데이터 소개", "탐구 질문 분석", "지역별 비교", "발전량 예측", "탐구 결론"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("프로젝트 주제")
    st.write("날씨에 따른 태양광 발전량 예측")
    st.caption("분석 데이터: 2025년 기상 자료 + 한국전력거래소 자료")

try:
    with st.spinner("데이터를 불러오고 날짜·시간·지역을 연결하고 있습니다..."):
        df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.info("GitHub 저장소의 CSV 파일명과 Raw 주소가 코드의 주소와 정확히 같은지 확인해 주세요.")
    st.stop()

# 공통 필터
regions = sorted(df["지역"].dropna().unique().tolist())
with st.sidebar:
    selected_regions = st.multiselect("분석할 지역", regions, default=regions)
    if not selected_regions:
        st.warning("지역을 하나 이상 선택해 주세요.")

view = df[df["지역"].isin(selected_regions)].copy()
if view.empty:
    st.warning("선택된 지역의 데이터가 없습니다.")
    st.stop()

if page == "한눈에 보기":
    st.header("전체 분석 요약")
    st.write("먼저 결합된 데이터의 규모와 태양광 발전량의 기본적인 특징을 확인합니다.")
    a, b, c, d = st.columns(4)
    a.metric("분석 데이터", f"{len(view):,}개")
    b.metric("분석 지역", f"{view['지역'].nunique()}개")
    c.metric("평균 발전량", f"{view['발전량'].mean():.3f} MWh")
    d.metric("발전량 0인 비율", f"{(view['발전량'].eq(0).mean() * 100):.1f}%")

    left, right = st.columns(2)
    with left:
        hourly = view.groupby("시간")["발전량"].mean().reset_index()
        st.plotly_chart(px.line(hourly, x="시간", y="발전량", markers=True, title="시간별 평균 태양광 발전량"), use_container_width=True)
    with right:
        monthly = view.groupby("월")["발전량"].mean().reset_index()
        st.plotly_chart(px.bar(monthly, x="월", y="발전량", title="월별 평균 태양광 발전량"), use_container_width=True)

    st.markdown('<div class="note">읽는 방법: 시간별·월별 평균은 선택한 지역과 결합된 데이터 안에서 계산한 값입니다. 그래프만으로 원인을 단정하지 말고 일사량·구름 관련 값도 함께 비교해야 합니다.</div>', unsafe_allow_html=True)
    st.subheader("데이터 미리보기")
    st.dataframe(view.head(12), use_container_width=True)

elif page == "데이터 소개":
    st.header("1. 데이터 소개")
    st.write("서로 다른 두 자료를 날짜, 시간, 지역 기준으로 연결해 분석용 데이터를 만들었습니다.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("결합된 데이터 수", f"{len(view):,}개")
    c2.metric("분석 지역 수", f"{view['지역'].nunique()}개")
    c3.metric("발전량 0인 행", f"{view['발전량'].eq(0).sum():,}개")
    c4.metric("분석 변수 수", "기상 5개 + 시간·월 등")
    st.subheader("사용한 데이터")
    st.markdown("""
    - **기상 데이터:** 시간·지역별 일사량(GHI, DNI, DHI), 구름 관련 값(CLOUD_OPACITY), 기온(AIR_TEMP)
    - **발전량 데이터:** 한국전력거래소의 지역별 시간별 태양광 전력거래량
    - **결합 방법:** 날짜·시간·지역이 같은 행끼리 연결하는 내부 결합(inner join)
    """)
    st.warning("전력거래량 자료는 한국전력거래소 자료에 포함된 발전사업자의 거래량이며, 국내 모든 태양광 설비의 전체 발전량과 같다고 볼 수 없습니다.")
    st.subheader("변수 설명")
    dictionary = pd.DataFrame([
        ["GHI", "수평면 전일사량", "태양광 에너지와 관련된 핵심 기상 변수"],
        ["DNI", "직달일사량", "태양에서 직접 도달하는 일사량"],
        ["DHI", "산란일사량", "대기에서 산란되어 도달하는 일사량"],
        ["CLOUD_OPACITY", "구름 관련 값", "구름의 광학적 불투명도 관련 값"],
        ["AIR_TEMP", "기온(℃)", "주변 공기의 온도"],
        ["발전량", "전력거래량(MWh)", "분석 대상인 태양광 거래량"]
    ], columns=["변수", "의미", "분석에서의 역할"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.subheader("결측치 확인")
    st.dataframe(view.isna().sum().rename("결측치 개수").to_frame(), use_container_width=True)
    st.download_button("현재 분석 데이터 CSV 다운로드", view.to_csv(index=False).encode("utf-8-sig"), "solar_analysis_data.csv", "text/csv")

elif page == "탐구 질문 분석":
    st.header("2. 탐구 질문 분석")
    question = st.selectbox("분석할 탐구 질문을 선택하세요", [
        "① 시간대별 태양광 발전량은 어떻게 달라질까?",
        "② 월별·계절별 발전량에는 어떤 차이가 있을까?",
        "③ 일사량과 발전량은 어떤 관계가 있을까?",
        "④ 기온과 구름량 중 발전량과 더 관련이 높은 변수는 무엇일까?",
        "⑤ 발전량이 0이거나 낮은 값은 언제 나타날까?"
    ])

    if question.startswith("①"):
        hourly = view.groupby("시간")["발전량"].mean().reset_index()
        peak = hourly.loc[hourly["발전량"].idxmax()]
        st.plotly_chart(px.line(hourly, x="시간", y="발전량", markers=True, title="시간대별 평균 발전량"), use_container_width=True)
        st.metric("평균 발전량이 가장 높은 시간", f"{int(peak['시간'])}시", f"{peak['발전량']:.3f} MWh")
        st.info("해석: 태양광 발전량은 일반적으로 낮 시간대에 높아질 것으로 예상됩니다. 실제 자료에서 정점이 언제 나타나는지 그래프로 확인하세요.")

    elif question.startswith("②"):
        t1, t2 = st.tabs(["월별 비교", "계절별 비교"])
        with t1:
            monthly = view.groupby("월")["발전량"].mean().reset_index()
            st.plotly_chart(px.line(monthly, x="월", y="발전량", markers=True, title="월별 평균 발전량"), use_container_width=True)
            best = monthly.loc[monthly["발전량"].idxmax()]
            st.write(f"선택한 데이터에서는 **{int(best['월'])}월**의 평균 발전량이 가장 높게 나타났습니다.")
        with t2:
            order = ["봄", "여름", "가을", "겨울"]
            seasonal = view.groupby("계절", observed=False)["발전량"].mean().reindex(order).reset_index()
            st.plotly_chart(px.bar(seasonal, x="계절", y="발전량", title="계절별 평균 발전량"), use_container_width=True)
            st.write("계절별 차이는 일사량, 낮의 길이, 구름, 기온 등이 함께 작용한 결과일 수 있습니다.")

    elif question.startswith("③"):
        sample = view.sample(min(len(view), 10000), random_state=42)
        st.plotly_chart(px.scatter(sample, x="GHI", y="발전량", opacity=0.45, title="일사량(GHI)과 발전량의 관계"), use_container_width=True)
        corr = view[["GHI", "발전량"]].corr().iloc[0, 1]
        st.metric("GHI와 발전량의 상관계수", f"{corr:.3f}")
        st.write(explain_corr(corr))
        st.caption("상관관계는 두 값이 함께 변하는 정도이며, 일사량이 발전량 변화의 유일한 원인이라는 뜻은 아닙니다.")

    elif question.startswith("④"):
        cols = ["AIR_TEMP", "CLOUD_OPACITY", "GHI", "DNI", "DHI", "발전량"]
        corr = view[cols].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="기상 변수와 발전량의 상관관계"), use_container_width=True)
        compare = corr["발전량"].drop("발전량").abs().sort_values(ascending=False)
        strongest = compare.index[0]
        st.metric("발전량과 상관관계 절댓값이 가장 큰 변수", strongest, f"{corr.loc[strongest, '발전량']:.3f}")
        st.write("상관계수 절댓값이 클수록 선형 관계가 강한 편입니다. 상관관계가 인과관계를 의미하지는 않습니다.")

    else:
        zero_rate = view.assign(영발전=view["발전량"].eq(0)).groupby("시간")["영발전"].mean().reset_index()
        st.plotly_chart(px.bar(zero_rate, x="시간", y="영발전", title="시간별 발전량 0인 데이터 비율"), use_container_width=True)
        threshold = view["발전량"].quantile(0.1)
        low = view[view["발전량"] <= threshold]
        c1, c2 = st.columns(2)
        c1.metric("발전량이 0인 행", f"{view['발전량'].eq(0).sum():,}개")
        c2.metric("하위 10% 기준", f"{threshold:.4f} MWh")
        st.write("야간에는 일사량이 거의 없어 발전량이 0에 가까울 수 있습니다. 0인 값을 무조건 오류로 처리하지 말고 시간과 기상 조건을 함께 확인해야 합니다.")
        st.dataframe(low.head(20), use_container_width=True)

elif page == "지역별 비교":
    st.header("3. 지역별 발전량 비교")
    st.write("지역별 평균 발전량과 일사량을 비교해 지역에 따른 차이를 살펴봅니다.")
    region_summary = view.groupby("지역").agg(
        평균발전량=("발전량", "mean"),
        평균GHI=("GHI", "mean"),
        데이터수=("발전량", "size")
    ).reset_index().sort_values("평균발전량", ascending=False)
    st.plotly_chart(px.bar(region_summary, x="지역", y="평균발전량", color="평균GHI", title="지역별 평균 발전량과 평균 GHI"), use_container_width=True)
    st.dataframe(region_summary, use_container_width=True, hide_index=True)
    st.caption("지역별 설비 규모와 거래 참여 범위가 다를 수 있으므로 평균 거래량 차이를 지역의 태양광 자원량 차이로 단정할 수 없습니다.")

elif page == "발전량 예측":
    st.header("4. 태양광 발전량 예측")
    st.write("기상 조건과 시간 정보를 입력해 선형회귀 모델로 태양광 전력거래량을 예측합니다.")
    features = ["GHI", "DNI", "DHI", "CLOUD_OPACITY", "AIR_TEMP", "시간", "월"]
    model_df = view[features + ["발전량", "날짜"]].dropna().sort_values("날짜")
    if len(model_df) < 30:
        st.warning("예측 모델을 학습하기 위한 데이터가 충분하지 않습니다.")
        st.stop()
    split = int(len(model_df) * 0.8)
    train, test = model_df.iloc[:split], model_df.iloc[split:]
    X_train, y_train = train[features], train["발전량"]
    X_test, y_test = test[features], test["발전량"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.3f} MWh", help="예측값과 실제값 차이의 평균적인 크기")
    m2.metric("RMSE", f"{rmse:.3f} MWh", help="큰 오차에 더 큰 벌점을 주는 지표")
    m3.metric("R² 점수", f"{r2:.3f}", help="모델이 자료의 변동을 설명하는 정도")
    st.caption("시간 순서 기준 앞의 80%로 학습하고 뒤의 20%로 평가했습니다. 미래 예측 성능을 완벽히 보장하지는 않습니다.")
    result_plot = pd.DataFrame({"실제 발전량": y_test, "예측 발전량": pred})
    st.plotly_chart(px.scatter(result_plot, x="실제 발전량", y="예측 발전량", title="실제 발전량과 예측 발전량 비교"), use_container_width=True)
    st.subheader("어떤 변수가 모델에 크게 반영됐을까?")
    coef = pd.DataFrame({"변수": features, "회귀계수": model.coef_})
    coef["영향 방향"] = coef["회귀계수"].apply(lambda x: "양(+)의 방향" if x >= 0 else "음(-)의 방향")
    st.plotly_chart(px.bar(coef.sort_values("회귀계수"), x="회귀계수", y="변수", orientation="h", title="선형회귀 모델의 변수별 계수"), use_container_width=True)
    st.caption("회귀계수 크기는 변수의 단위와 범위에 영향을 받으므로, 계수만으로 변수 중요도를 단정하지 마세요.")
    st.subheader("직접 날씨 조건을 입력해 예측하기")
    c1, c2, c3 = st.columns(3)
    with c1:
        ghi = st.number_input("GHI 일사량", min_value=0.0, value=float(view["GHI"].median()))
        dni = st.number_input("DNI 직달일사량", min_value=0.0, value=float(view["DNI"].median()))
        dhi = st.number_input("DHI 산란일사량", min_value=0.0, value=float(view["DHI"].median()))
    with c2:
        cloud = st.number_input("CLOUD_OPACITY 구름 관련 값", min_value=0.0, value=float(view["CLOUD_OPACITY"].median()))
        temp = st.number_input("AIR_TEMP 기온(℃)", value=float(view["AIR_TEMP"].median()))
    with c3:
        hour = st.slider("시간", 0, 23, 12)
        month = st.slider("월", 1, 12, 6)
    if st.button("태양광 발전량 예측하기", type="primary"):
        row = pd.DataFrame([[ghi, dni, dhi, cloud, temp, hour, month]], columns=features)
        estimate = max(0.0, float(model.predict(row)[0]))
        st.success(f"예측 발전량은 약 **{estimate:.3f} MWh**입니다.")
        st.caption("이 값은 선택한 지역 데이터로 학습한 단순 선형회귀 모델의 추정치입니다.")

else:
    st.header("5. 탐구 결론 및 한계")
    st.subheader("탐구 질문에 대한 정리")
    st.markdown("""
    1. **시간대별 차이:** 태양의 위치와 일사량 변화에 따라 낮 시간대 발전량이 달라질 수 있습니다.
    2. **월별·계절별 차이:** 계절에 따른 낮의 길이, 일사량, 구름, 기온 등이 함께 영향을 줄 수 있습니다.
    3. **일사량과 발전량:** 일사량은 태양광 발전과 직접 관련된 중요한 기상 요인입니다.
    4. **기온과 구름:** 두 변수의 상관계수를 비교해 어느 쪽이 자료에서 더 강한 관계를 보이는지 판단할 수 있습니다.
    5. **0 또는 낮은 발전량:** 야간이나 일사량이 낮은 상황에서 자연스럽게 나타날 수 있어 무조건 오류로 보면 안 됩니다.
    """)
    st.subheader("이 분석의 한계")
    st.markdown("""
    - 한국전력거래소의 전력거래량은 전국 모든 태양광 설비의 전체 발전량과 다를 수 있습니다.
    - 기상 관측 지역과 발전량 집계 지역의 범위가 완전히 일치하지 않을 수 있습니다.
    - 거래시간을 기상 시간에 맞추기 위해 거래시간에서 1을 빼는 가정을 사용했으므로 원자료의 시간 정의를 확인해야 합니다.
    - 발전 설비 용량, 패널 방향·각도, 고장·정비, 음영 등 설비 요인은 모델에 포함하지 않았습니다.
    - 선형회귀는 복잡한 날씨와 발전량의 관계를 모두 설명하지 못할 수 있습니다.
    """)
    st.subheader("후속 탐구 방향")
    st.markdown("""
    - 설비 용량 자료를 추가해 지역별 발전량을 설비 규모 대비로 비교하기
    - 더 긴 기간의 데이터를 모아 연도별·계절별 패턴 비교하기
    - 랜덤포레스트 등 다른 예측 모델과 선형회귀의 성능 비교하기
    - 시간대와 지역별 데이터 정합성을 원자료 정의에 맞춰 재검증하기
    """)
    st.info("위 결론 문장은 분석 방향을 설명하는 기본 해석입니다. 실제 발표에서는 그래프와 상관계수, 예측 성능 수치를 확인한 뒤 본인의 데이터 결과에 맞게 고쳐 쓰세요.")

st.divider()
st.markdown('<p class="small-muted">탐구 주제: 날씨에 따른 태양광 발전량 예측 · 데이터 기반 탐구 프로젝트</p>', unsafe_allow_html=True)
