import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Load preprocessed data
df = pd.read_csv("D:/OneDrive - Beyond Key Systems Pvt. Ltd/Desktop/sales_prediction_app/clean_data.csv")



# Convert date column to datetime format
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
# df["Date"] = pd.to_datetime(df["Date"], format="%m/%y", errors="coerce")


# Pivot the data to have customers as rows and dates as columns
wide_df = df.pivot(index="Customer", columns="Date", values="Sales").fillna(0)
wide_df.columns = [col.strftime("%m/%y") for col in wide_df.columns]  

wide_df = wide_df.fillna(0)
# Function to train SARIMA and forecast
def forecast_sales(customer, start_date="2024/06/01", end_date="2026/12/01"):
    sales_data = wide_df.loc[customer].T
    sales_data.index = pd.to_datetime(sales_data.index)

    train_data = sales_data[:start_date]
    model = SARIMAX(train_data, order=(2, 0, 0), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()

    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    forecast = results.get_forecast(steps=len(forecast_dates))
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Sales": forecast_values})
    return sales_data, forecast_df, forecast_conf_int

# Streamlit Interface
st.title("Sales Prediction")

# Customer selection dropdown
customer_name = st.selectbox("Select Customer", wide_df.index)

# Date range selector
date_range = st.date_input("Select Forecast Range", [pd.to_datetime("2025-01-01"), pd.to_datetime("2026-12-01")])

# Run forecast on button click
if st.button("Generate Forecast"):
    actual_data, forecast_data, conf_int = forecast_sales(customer_name, "2024-06-01", date_range[1].strftime("%Y-%m-%d"))

    # Plot actual and forecasted sales
    plt.figure(figsize=(10, 5))
    plt.plot(actual_data.index, actual_data.values, label="Actual Sales", marker="o")
    plt.plot(forecast_data["Date"], forecast_data["Predicted Sales"], label="Forecasted Sales", linestyle="dashed", marker="o", color="red")
    plt.fill_between(forecast_data["Date"], conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
    plt.axvline(x=pd.to_datetime("2025-01-01"), color='gray', linestyle='dotted')
    plt.title(f"Sales Forecast for {customer_name}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    st.pyplot(plt)

    # Save forecast to Excel
    excel_filename = f"{customer_name}_sales_forecast.xlsx"
    forecast_data.to_excel(excel_filename, index=False)

    # Download button for Excel file
    with open(excel_filename, "rb") as file:
        st.download_button(label="Download Forecast Data", data=file, file_name=excel_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
