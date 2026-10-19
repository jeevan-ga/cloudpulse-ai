
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="CloudPulse AI",
    layout="wide"
)

st.title("CloudPulse AI")
st.subheader("Cloud Cost Forecasting and Anomaly Detection")

st.write(
    "Upload cloud billing data to analyze expenditure "
    "and identify unusual spending patterns."
)

st.sidebar.header("Billing Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a billing CSV file",
    type=["csv"]
)

def validate_billing_data(data):
    required_columns = {"date", "service", "cost"}

    data.columns = data.columns.str.strip().str.lower()

    missing = required_columns - set(data.columns)
    if missing:
        return None, (
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()
    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )
    data["cost"] = pd.to_numeric(
        data["cost"], errors="coerce"
    )

    if data["date"].isna().any():
        return None, "Some dates are invalid."

    if data["cost"].isna().any():
        return None, "Some costs are missing or non-numeric."

    if (data["cost"] < 0).any():
        return None, "Costs cannot be negative."

    if data["service"].isna().any():
        return None, "Some service names are missing."

    data["service"] = data["service"].astype(str).str.strip()

    if data["service"].eq("").any():
        return None, "Some service names are empty."

    return data.sort_values("date"), None


if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        billing_data, error = validate_billing_data(raw_data)

        if error:
            st.error(error)
        else:
            st.success("Billing data passed validation.")

            total_cost = billing_data["cost"].sum()
            average_cost = billing_data.groupby("date")["cost"].sum().mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total expenditure", f"{total_cost:,.2f}")
            col2.metric("Billing records", len(billing_data))
            col3.metric("Services", billing_data["service"].nunique())

            daily_cost = (
                billing_data.groupby("date", as_index=False)["cost"]
                .sum()
            )

            st.subheader("Daily Cloud Spending")
            fig = px.line(
                daily_cost,
                x="date",
                y="cost",
                markers=True,
                labels={"date": "Date", "cost": "Cloud Cost"}
            )
            st.plotly_chart(fig, width="stretch")

            service_cost = (
                billing_data.groupby("service", as_index=False)["cost"]
                .sum()
                .sort_values("cost", ascending=False)
            )

            st.subheader("Spending by Service")
            st.bar_chart(service_cost.set_index("service"))

            st.subheader("Validated Billing Records")
            st.dataframe(billing_data, width="stretch")
            
            st.subheader("7-Day Cloud Cost Forecast")

            daily_cost = (
                billing_data.groupby("date")["cost"]
                .sum()
            )

            daily_cost.index = pd.to_datetime(daily_cost.index)
            daily_cost = daily_cost.groupby(level=0).sum().sort_index()
            daily_cost = daily_cost.asfreq("D", fill_value=0)

            if len(daily_cost) >= 14:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_absolute_error

                values = daily_cost.to_numpy(dtype=float)
                window = 7
                X, y = [], []

                # Use the previous 7 days to predict the next day.
                for i in range(window, len(values)):
                    X.append(values[i - window:i])
                    y.append(values[i])

                X = np.asarray(X)
                y = np.asarray(y)

                # Chronological train/test split.
                split = int(len(X) * 0.8)

                if split > 0 and split < len(X):
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    rf_model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    )
                    rf_model.fit(X_train, y_train)

                    ml_predictions = rf_model.predict(X_test)
                    baseline_predictions = X_test.mean(axis=1)

                    ml_mae = mean_absolute_error(
                        y_test, ml_predictions
                    )
                    baseline_mae = mean_absolute_error(
                        y_test, baseline_predictions
                    )

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Random Forest MAE",
                        f"{ml_mae:,.2f}"
                    )
                    col2.metric(
                        "7-Day Baseline MAE",
                        f"{baseline_mae:,.2f}"
                    )

                    st.caption(
                        "Lower MAE is better. Evaluation uses historical "
                        "data held out in chronological order. The ML model "
                        "may not outperform the baseline."
                    )

                    # Train on all available history for future forecasting.
                    final_model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    )
                    final_model.fit(X, y)

                    recent = list(values[-window:])
                    predictions = []

                    for _ in range(7):
                        next_cost = max(
                            0.0,
                            float(final_model.predict(
                                np.asarray(
                                    recent[-window:]
                                ).reshape(1, -1)
                            )[0])
                        )
                        predictions.append(next_cost)
                        recent.append(next_cost)

                    future_dates = pd.date_range(
                        start=daily_cost.index.max()
                        + pd.Timedelta(days=1),
                        periods=7,
                        freq="D"
                    )

                    forecast = pd.DataFrame({
                        "date": future_dates,
                        "predicted_cost": predictions
                    })

                    st.metric(
                        "Estimated spending for next 7 days",
                        f"{forecast['predicted_cost'].sum():,.2f}"
                    )

                    st.line_chart(
                        forecast.set_index("date")["predicted_cost"]
                    )

                    st.dataframe(forecast, width="stretch")

                else:
                    st.info(
                        "More billing history is needed to evaluate "
                        "the forecasting model."
                    )

            else:
                st.info(
                    "At least 14 calendar days of billing history are "
                    "needed for this forecasting experiment."
                )

            st.subheader("Cloud Spending Anomaly Detection")
            from sklearn.ensemble import IsolationForest

            daily_service_cost = (
                billing_data.groupby(
                    ["date", "service"], as_index=False
                )["cost"].sum()
            )

            if len(daily_service_cost) >= 10:
                sensitivity = st.slider(
                    "Anomaly sensitivity",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.10,
                    step=0.01
                )

                model = IsolationForest(
                    contamination=sensitivity,
                    random_state=42
                )

                daily_service_cost["anomaly"] = model.fit_predict(
                    daily_service_cost[["cost"]]
                )

                anomalies = daily_service_cost[
                    daily_service_cost["anomaly"] == -1
                ].copy()

                if anomalies.empty:
                    st.success("No unusual spending records detected.")
                else:
                    st.warning(
                        f"Detected {len(anomalies)} potentially unusual "
                        "spending records."
                    )

                    st.dataframe(
                        anomalies[["date", "service", "cost"]].sort_values(
                            "cost", ascending=False
                        ),
                        width="stretch"
                    )

                    st.caption(
                        "These records differ from the overall cost "
                        "distribution. Review them before treating them "
                        "as genuine incidents."
                    )
            else:
                st.info(
                    "At least 10 date-service records are required "
                    "for anomaly detection."
                )


    except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("Could not read this CSV. Check that it is a valid CSV file.")

else:
    st.info("Upload a CSV file in the sidebar to begin.")

    st.subheader("Required CSV format")
    example = pd.DataFrame({
        "date": ["2026-10-01", "2026-10-01", "2026-10-02"],
        "service": ["Compute", "Storage", "Compute"],
        "cost": [1500.00, 300.00, 1650.00]
    })
    st.dataframe(example, width="stretch")

    st.caption(
        "Use columns named date, service, and cost. "
        "Costs must be numeric and non-negative."
    )
