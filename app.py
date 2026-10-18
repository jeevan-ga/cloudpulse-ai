
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="CloudPulse AI",
    page_icon="",
    layout="wide"
)

st.title("CloudPulse AI")
st.subheader("Cloud Cost Forecasting and Anomaly Detection")

st.write(
    "Analyze cloud expenditure, explore spending patterns, "
    "and identify unusual cost changes."
)

# Sample billing data for the initial dashboard
dates = pd.date_range(
    end=pd.Timestamp.today().normalize(),
    periods=30,
    freq="D"
)

np.random.seed(42)

costs = np.random.normal(
    loc=1500,
    scale=150,
    size=30
).clip(min=500)

df = pd.DataFrame({
    "Date": dates,
    "Cloud Cost": costs
})

# Dashboard metrics
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total spending (30 days)",
        f"₹{df['Cloud Cost'].sum():,.2f}"
    )

with col2:
    st.metric(
        "Average daily spending",
        f"₹{df['Cloud Cost'].mean():,.2f}"
    )

# Spending chart
st.subheader("Daily Cloud Spending")

fig = px.line(
    df,
    x="Date",
    y="Cloud Cost",
    markers=True,
    title="Cloud Spending Over the Last 30 Days"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Prototype dashboard using synthetic sample data. "
    "Forecasting and anomaly detection will be added next."
)
