import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

model = joblib.load("risk_prediction_model_rangeupdated.pkl")
# model = joblib.load("risk_prediction_model_diver.pkl")

market_data = pd.read_csv("market_monthly_features.csv")

st.title("Hybrid Portfolio Risk & Behavioral Stability Engine")

st.write(
"""
This system predicts **portfolio instability risk (%)** using:

Financial risk metrics
+
Investor behavioral characteristics
+
Machine Learning
"""
)

st.markdown("---")

st.header("Portfolio Allocation")

col1, col2, col3 = st.columns(3)

with col1:
    spy_weight = st.slider("Equity (SPY)", 0.0, 1.0, 0.5, 0.01)

with col2:
    gld_weight = st.slider("Gold (GLD)", 0.0, 1.0, 0.2, 0.01)

with col3:
    agg_weight = st.slider("Bonds (AGG)", 0.0, 1.0, 0.3, 0.01)

total_weight = spy_weight + gld_weight + agg_weight

st.write(f"Total Weight: {total_weight:.2f}")

if abs(total_weight - 1.0) > 0.01:
    st.warning("⚠ Portfolio weights must sum to 1")

st.markdown("---")


st.header("Investment Duration")

horizon = st.slider(
"Investment Horizon (Years)",
1,
20,
10
)

st.markdown("---")
st.header("Investor Behavioral Profile")

col1, col2 = st.columns(2)

with col1:

    risk_tol = st.slider(
        "Risk Tolerance",
        0.0, 1.0, 0.5
    )

    panic_threshold = st.slider(
        "Panic Threshold (loss trigger)",
        0.05, 0.25, 0.15
    )

with col2:

    exit_tendency = st.slider(
        "Exit Tendency (probability of selling)",
        0.0, 1.0, 0.4
    )

    rebalance_score = st.slider(
        "Rebalancing Discipline",
        0.0, 1.0, 0.6
    )

st.markdown("---")

if st.button("Calculate Portfolio Risk"):

    if abs(total_weight - 1.0) > 0.01:

        st.error("Portfolio weights must sum to 1")

    else:

        portfolio_returns = (
            spy_weight * market_data["SPY_Return_M"] +
            gld_weight * market_data["GLD_Return_M"] +
            agg_weight * market_data["AGG_Return_M"]
        )

        # portfolio_return = portfolio_returns.iloc[-1]

        # weights = np.array([spy_weight, gld_weight, agg_weight])

        # returns_matrix = market_data[
        # ["SPY_Return_M","GLD_Return_M","AGG_Return_M"]
        # ]

        # cov_matrix = returns_matrix.tail(12).cov()

        # portfolio_volatility = np.sqrt(
        #     np.dot(weights.T, np.dot(cov_matrix, weights))
        # )
        # portfolio_volatility = portfolio_volatility * np.sqrt(12)

        # downside_returns = portfolio_returns.clip(upper=0)

        # downside_risk = downside_returns.rolling(6).std().iloc[-1]

        # cumulative = (1 + portfolio_returns).cumprod()

        # peak = cumulative.cummax()

        # drawdown = (cumulative - peak) / peak

        # portfolio_drawdown = drawdown.iloc[-1]

        # equity_horizon_risk = spy_weight / np.sqrt(horizon)
        portfolio_return = portfolio_returns.tail(6).mean()

        weights = np.array([spy_weight, gld_weight, agg_weight])

        returns_matrix = market_data[
        ["SPY_Return_M","GLD_Return_M","AGG_Return_M"]
        ]

        cov_matrix = returns_matrix.tail(24).cov()

        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

        portfolio_volatility = portfolio_volatility * np.sqrt(12)

        downside_returns = portfolio_returns.clip(upper=0)

        downside_risk = downside_returns.tail(12).std()

        cumulative = (1 + portfolio_returns).cumprod()

        peak = cumulative.cummax()

        drawdown = (cumulative - peak) / peak

        portfolio_drawdown = drawdown.min()

        equity_horizon_risk = spy_weight / np.sqrt(horizon)

        concentration = (
            spy_weight**2 +
            gld_weight**2 +
            agg_weight**2
        )
        equity_horizon_risk = spy_weight / np.sqrt(horizon)

        diversification = 1 - concentration

        input_data = pd.DataFrame({

            "SPY_weight":[spy_weight],
            "GLD_weight":[gld_weight],
            "AGG_weight":[agg_weight],

            "Portfolio_Return":[portfolio_return],
            "Portfolio_Volatility":[portfolio_volatility],
            "Portfolio_Drawdown":[portfolio_drawdown],
            "Downside_Risk":[downside_risk],

            "Concentration":[concentration],
            "Diversification":[diversification],

            "Investment_Horizon":[horizon],
            "Equity_Horizon_Risk":[equity_horizon_risk],

            "Risk_Tolerance_Score":[risk_tol],
            "Exit_Tendency":[exit_tendency],
            "Rebalance_Score":[rebalance_score],
            "Panic_Threshold":[panic_threshold]
            })

        prediction = model.predict(input_data)[0]

        st.subheader(f"Predicted Portfolio Risk: {prediction:.2f}%")

        if prediction < 20:
            risk_level = "Very Low Risk"
            color = "green"

        elif prediction < 40:
            risk_level = "Low Risk"
            color = "blue"

        elif prediction < 60:
            risk_level = "Moderate Risk"
            color = "orange"

        elif prediction < 80:
            risk_level = "High Risk"
            color = "red"

        else:
            risk_level = "Very High Risk"
            color = "violet"

        st.metric("Predicted Portfolio Risk", f"{prediction:.2f}%")
        st.markdown(f"### Risk Category: :{color}[{risk_level}]")
        st.progress(prediction/100)

        st.header("Portfolio Allocation")

        labels = ["Equity", "Gold", "Bonds"]

        sizes = [spy_weight, gld_weight, agg_weight]

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")

        st.pyplot(fig, use_container_width=False)

        st.markdown("---")

        st.header("Historical Portfolio Performance")

        portfolio_growth = (1 + portfolio_returns).cumprod()

        fig2, ax2 = plt.subplots()

        ax2.plot(portfolio_growth)

        ax2.set_title("Portfolio Growth Over Time")

        ax2.set_xlabel("Time")

        ax2.set_ylabel("Portfolio Value")

        st.pyplot(fig2, use_container_width=False)

        st.markdown("---")

        st.header("Model Explainability (SHAP)")

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(input_data)

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[0],
            data=input_data.iloc[0],
            feature_names=input_data.columns
        )

        fig3, ax3 = plt.subplots()

        shap.plots.waterfall(explanation, show=False)

        st.pyplot(fig3, use_container_width=False)

        st.markdown("---")

        st.header("Model Feature Importance")

        importance = pd.Series(
            model.feature_importances_,
            index=input_data.columns
        ).sort_values(ascending=False)

        fig4, ax4 = plt.subplots(figsize=(6,3))

        importance.plot.bar(ax=ax4)

        ax4.set_title("Feature Importance")

        st.pyplot(fig4, use_container_width=False)

st.markdown("---")

st.write(
"""
Model: Random Forest Regressor

Data Source: Yahoo Finance via yfinance  
Time Period: 2000–2024

This hybrid model integrates **financial risk metrics and behavioral stability indicators**
to detect portfolio instability.
"""
)