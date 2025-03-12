import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide", initial_sidebar_state="expanded")

# Sidebar navigation for multi-page functionality
page = st.sidebar.selectbox("Navigation Panel", ["Prediction", "Project Info"])

if page == "Prediction":
    st.title("📈 Stock Price Prediction App")
    st.markdown("Predict stock prices using our trained model.")

    # Sidebar inputs for manual prediction
    st.sidebar.header("Input Features")
    mar_2021_net_profit = st.sidebar.number_input("Mar 2021 Net Profit", value=1000.0, step=100.0, min_value=0.0)
    mar_2023_fixed_assets = st.sidebar.number_input("Mar 2023 Fixed Assets", value=5000.0, step=100.0, min_value=0.0)
    mar_2022_reserves = st.sidebar.number_input("Mar 2022 Reserves", value=3000.0, step=100.0, min_value=0.0)
    price_input = st.sidebar.number_input("Current Price", value=1500.0, step=50.0, min_value=0.0)
    rsi_input = st.sidebar.number_input("Current RSI", value=50.0, step=1.0, min_value=0.0, max_value=100.0)

    # Select feature for iteration
    st.sidebar.subheader("Iterative Prediction")
    feature_to_iterate = st.sidebar.selectbox("Choose feature to iterate", ["RSI", "Price"])
    num_iterations = st.sidebar.slider("Number of Iterations", 5, 20, 10)

    # Create a DataFrame for the manual input (order must match training)
    input_data = pd.DataFrame({
        "Mar_2021_Net_Profit": [mar_2021_net_profit],
        "Mar_2023_Fixed_Assets": [mar_2023_fixed_assets],
        "Mar_2022_Reserves": [mar_2022_reserves],
        "Price": [price_input],
        "RSI": [rsi_input]
    })

    st.markdown("### 📋 Input Data Summary")
    st.dataframe(input_data)

    # Load the trained model (cached to avoid re-loading)
    @st.cache_resource
    def load_model():
        with open("stock_price_model_5.pkl", "rb") as f:
            return pickle.load(f)
    model = load_model()

    # Iterative prediction
    if st.button("🔮 Predict Stock Price"):
        predictions = []
        feature_values = []

        for i in range(num_iterations):
            temp_input = input_data.copy()
            if feature_to_iterate == "RSI":
                temp_input["RSI"] = rsi_input + i
            else:
                temp_input["Price"] = price_input + (i * 50)

            pred = model.predict(temp_input)[0]
            predictions.append(pred)
            feature_values.append(temp_input[feature_to_iterate].values[0])

        st.success(f"Predicted Stock Price at {feature_to_iterate}={feature_values[0]:.2f}: ₹{predictions[0]:.2f}")

        # Plot predictions
        fig = px.line(
            x=feature_values,
            y=predictions,
            title=f"📊 Stock Price Predictions over {feature_to_iterate}",
            markers=True
        )
        fig.update_layout(
            xaxis_title=feature_to_iterate,
            yaxis_title="Predicted Price",
            width=800,
            height=400
        )
        fig.add_vline(x=input_data[feature_to_iterate].values[0], line_dash="dash", line_color="green")
        st.plotly_chart(fig)

    st.markdown("---")
    st.header("📂 Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV for Bulk Predictions", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        required_cols = ["Mar_2021_Net_Profit", "Mar_2023_Fixed_Assets", "Mar_2022_Reserves", "Price", "RSI"]
        if not all(col in data.columns for col in required_cols):
            st.error(f"Uploaded CSV must contain these columns: {required_cols}")
        else:
            data_selected = data[required_cols]
            predictions = model.predict(data_selected)
            data["Predicted_Price"] = predictions
            st.markdown("### 🧾 Bulk Data Preview (with Predictions)")
            st.dataframe(data.head())

            fig2 = px.line(
                data,
                x=data.index,
                y="Predicted_Price",
                title="📈 Bulk Predicted Stock Prices",
                markers=True
            )
            fig2.update_layout(xaxis_title="Index", yaxis_title="Predicted Price", width=800, height=400)
            st.plotly_chart(fig2)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )

elif page == "Project Info":
    st.title("📚 Project Information")
    st.markdown("## 🧠 Model Information")
    st.write("**Model Name:** Stock Price Prediction Model using XGBoost")

    st.markdown("## 🔍 Project Steps")
    st.write("""
    1. **Data Collection:** Gathered stock market data from reliable sources.
    2. **Data Preprocessing:** Cleaned data, handled missing values, and selected 5 key features.
    3. **Feature Engineering:** Created technical indicators such as moving averages and RSI.
    4. **Model Training:** Trained an XGBoost Regressor using the selected features.
    5. **Model Evaluation:** Evaluated performance using RMSE, MAE, and R² metrics.
    6. **Deployment:** Built this interactive web app using Streamlit.
    """)

    st.markdown("## 📊 Dataset Overview")
    st.write("The original dataset contained 132 columns covering various financial metrics. For this model, 5 key features were selected: **Mar_2021_Net_Profit, Mar_2023_Fixed_Assets, Mar_2022_Reserves, Price, RSI**")

    st.markdown("## 👩‍💻 Developer Information")
    st.write("Developed by **Riddhi Suvariya**")
    st.write("Contact: **rsuvariya1510@gmail.com**")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit | Model: XGBoost")