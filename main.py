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
