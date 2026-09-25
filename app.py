import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    IsolationForest
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Sales Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 20px;
    }

    .kpi-title {
        font-size: 15px;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }

    .section-header {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 8px 18px;
    }

    .stDataFrame {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI-Powered Sales Intelligence & Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Data Science Project | Sales Analytics | Machine Learning | AI Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv("sales_data.csv")
except FileNotFoundError:
    st.error("❌ sales_data.csv file not found!")
    st.stop()

# ============================================================
# DATA CLEANING
# ============================================================

df = df.drop_duplicates()

# Convert numeric columns
if "Quantity" in df.columns:
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

if "Price" in df.columns:
    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

# Convert Date if available
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

# Remove invalid numeric rows
df = df.dropna(
    subset=["Quantity", "Price"]
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Total_Sales"] = (
    df["Quantity"] * df["Price"]
)

df["Revenue_Per_Unit"] = (
    df["Total_Sales"] / df["Quantity"]
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

show_dataset = st.sidebar.checkbox(
    "Show Dataset",
    value=True
)

show_analysis = st.sidebar.checkbox(
    "Show Analysis",
    value=True
)

show_charts = st.sidebar.checkbox(
    "Show Charts",
    value=True
)

show_ml = st.sidebar.checkbox(
    "Show Machine Learning",
    value=True
)

show_forecast = st.sidebar.checkbox(
    "Show AI Forecast",
    value=True
)

# ============================================================
# DATASET
# ============================================================

if show_dataset:

    st.header("📋 Sales Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Total Products",
        df["Product"].nunique()
        if "Product" in df.columns else 0
    )

    col3.metric(
        "Total Cities",
        df["City"].nunique()
        if "City" in df.columns else 0
    )

# ============================================================
# KPI SECTION
# ============================================================

st.header("📊 Business KPIs")

total_revenue = df["Total_Sales"].sum()

average_sale = df["Total_Sales"].mean()

total_quantity = df["Quantity"].sum()

if "Product" in df.columns:
    product_sales = (
        df.groupby("Product")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    top_product = product_sales.index[0]

else:
    top_product = "N/A"

if "City" in df.columns:
    city_sales = (
        df.groupby("City")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    best_city = city_sales.index[0]

else:
    best_city = "N/A"


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Revenue",
    f"₹{total_revenue:,.2f}"
)

col2.metric(
    "🏆 Top Product",
    top_product
)

col3.metric(
    "🌆 Best City",
    best_city
)

col4.metric(
    "📈 Average Sale",
    f"₹{average_sale:,.2f}"
)

# ============================================================
# PRODUCT ANALYSIS
# ============================================================

if show_analysis:

    st.markdown("---")

    st.header("📦 Product Analysis")

    if "Product" in df.columns:

        product_analysis = (
            df.groupby("Product")
            .agg(
                Quantity=("Quantity", "sum"),
                Revenue=("Total_Sales", "sum")
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
        )

        st.dataframe(
            product_analysis,
            use_container_width=True
        )

# ============================================================
# CITY ANALYSIS
# ============================================================

    st.header("🌆 City Analysis")

    if "City" in df.columns:

        city_analysis = (
            df.groupby("City")
            .agg(
                Quantity=("Quantity", "sum"),
                Revenue=("Total_Sales", "sum")
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
        )

        st.dataframe(
            city_analysis,
            use_container_width=True
        )

# ============================================================
# CATEGORY ANALYSIS
# ============================================================

    if "Category" in df.columns:

        st.header("🗂️ Category Analysis")

        category_analysis = (
            df.groupby("Category")
            .agg(
                Quantity=("Quantity", "sum"),
                Revenue=("Total_Sales", "sum")
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
        )

        st.dataframe(
            category_analysis,
            use_container_width=True
        )

# ============================================================
# CHARTS
# ============================================================

if show_charts:

    st.markdown("---")

    st.header("📈 Sales Visualizations")

    # --------------------------------------------------------
    # Product Revenue Chart
    # --------------------------------------------------------

    if "Product" in df.columns:

        product_chart = (
            df.groupby("Product")["Total_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        fig1, ax1 = plt.subplots(
            figsize=(10, 5)
        )

        product_chart.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_title(
            "Product-wise Revenue"
        )

        ax1.set_xlabel(
            "Product"
        )

        ax1.set_ylabel(
            "Revenue"
        )

        plt.xticks(rotation=45)

        st.pyplot(fig1)

    # --------------------------------------------------------
    # City Revenue Chart
    # --------------------------------------------------------

    if "City" in df.columns:

        city_chart = (
            df.groupby("City")["Total_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        city_chart.plot(
            kind="bar",
            ax=ax2
        )

        ax2.set_title(
            "City-wise Revenue"
        )

        ax2.set_xlabel(
            "City"
        )

        ax2.set_ylabel(
            "Revenue"
        )

        plt.xticks(rotation=45)

        st.pyplot(fig2)

    # --------------------------------------------------------
    # Category Chart
    # --------------------------------------------------------

    if "Category" in df.columns:

        category_chart = (
            df.groupby("Category")["Total_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        fig3, ax3 = plt.subplots(
            figsize=(8, 5)
        )

        category_chart.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax3
        )

        ax3.set_title(
            "Category Revenue Distribution"
        )

        ax3.set_ylabel("")

        st.pyplot(fig3)

# ============================================================
# MACHINE LEARNING
# ============================================================

if show_ml:

    st.markdown("---")

    st.header(
        "🤖 Machine Learning Sales Prediction"
    )

    st.write(
        "Multiple Machine Learning models are trained "
        "to predict Total Sales."
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = [
        "Quantity",
        "Price"
    ]

    X = df[features]

    y = df["Total_Sales"]

    # --------------------------------------------------------
    # Train Test Split
    # --------------------------------------------------------

    if len(df) >= 5:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        models = {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        }

        results = []

        predictions = {}

        trained_models = {}

        # ----------------------------------------------------
        # Train Models
        # ----------------------------------------------------

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                pred
            )

            mse = mean_squared_error(
                y_test,
                pred
            )

            rmse = np.sqrt(mse)

            r2 = r2_score(
                y_test,
                pred
            )

            results.append(
                {
                    "Model": name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2 Score": r2
                }
            )

            predictions[name] = pred

            trained_models[name] = model

        # ----------------------------------------------------
        # Model Comparison
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        st.subheader(
            "🏆 Model Comparison"
        )

        st.dataframe(
            results_df.round(2),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Best Model
        # ----------------------------------------------------

        best_model_name = results_df.loc[
            results_df["R2 Score"].idxmax(),
            "Model"
        ]

        best_model = trained_models[
            best_model_name
        ]

        st.success(
            f"🏆 Highest R² model in this test split: "
            f"{best_model_name}"
        )

        # ----------------------------------------------------
        # Model Selector
        # ----------------------------------------------------

        selected_model_name = st.selectbox(
            "Select Prediction Model",
            list(trained_models.keys())
        )

        selected_model = trained_models[
            selected_model_name
        ]

        # ----------------------------------------------------
        # Prediction Input
        # ----------------------------------------------------

        st.subheader(
            "🔮 Predict Sales"
        )

        col1, col2 = st.columns(2)

        with col1:

            input_quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1
            )

        with col2:

            input_price = st.number_input(
                "Price",
                min_value=1.0,
                value=1000.0
            )

        if st.button(
            "🚀 Predict Sales"
        ):

            input_data = pd.DataFrame(
                {
                    "Quantity": [input_quantity],
                    "Price": [input_price]
                }
            )

            prediction = selected_model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                prediction
            )

            st.success(
                f"💰 Predicted Sales: "
                f"₹{prediction:,.2f}"
            )

        # ----------------------------------------------------
        # Actual vs Predicted
        # ----------------------------------------------------

        st.subheader(
            "📊 Actual vs Predicted Sales"
        )

        selected_predictions = predictions[
            selected_model_name
        ]

        comparison_df = pd.DataFrame(
            {
                "Actual": y_test.values,
                "Predicted": selected_predictions
            }
        )

        st.dataframe(
            comparison_df.round(2),
            use_container_width=True
        )

        fig4, ax4 = plt.subplots(
            figsize=(10, 5)
        )

        ax4.plot(
            y_test.values,
            marker="o",
            label="Actual"
        )

        ax4.plot(
            selected_predictions,
            marker="x",
            label="Predicted"
        )

        ax4.set_title(
            f"Actual vs Predicted - "
            f"{selected_model_name}"
        )

        ax4.set_xlabel(
            "Test Sample"
        )

        ax4.set_ylabel(
            "Sales"
        )

        ax4.legend()

        st.pyplot(fig4)

        # ----------------------------------------------------
        # Random Forest Feature Importance
        # ----------------------------------------------------

        if (
            selected_model_name
            == "Random Forest"
        ):

            st.subheader(
                "🌲 Feature Importance"
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": selected_model.feature_importances_
                }
            ).sort_values(
                "Importance",
                ascending=False
            )

            st.dataframe(
                importance_df.round(4),
                use_container_width=True
            )


# ============================================================
# EXPLAINABLE AI — FEATURE IMPACT
# ============================================================

st.markdown("---")
st.header("🧠 Explainable AI — Feature Impact")
st.write(
    "This section explains which input features have the strongest "
    "influence on the selected Random Forest sales prediction model."
)

if selected_model_name == "Random Forest" and hasattr(selected_model, "feature_importances_"):

    xai_df = pd.DataFrame({
        "Feature": features,
        "Importance": selected_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    col_x1, col_x2 = st.columns([1.4, 1])

    with col_x1:
        st.subheader("📊 Feature Impact Chart")

        fig_xai, ax_xai = plt.subplots(figsize=(8, 4.5))
        ax_xai.barh(
            xai_df["Feature"][::-1],
            xai_df["Importance"][::-1]
        )
        ax_xai.set_xlabel("Relative Importance")
        ax_xai.set_ylabel("Feature")
        ax_xai.set_title("Random Forest Feature Importance")
        plt.tight_layout()
        st.pyplot(fig_xai)

    with col_x2:
        st.subheader("📋 Impact Table")
        st.dataframe(
            xai_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        top_feature = xai_df.iloc[0]["Feature"]
        top_importance = xai_df.iloc[0]["Importance"]

        st.info(
            f"**Highest-impact feature:** {top_feature}\n\n"
            f"Relative importance: **{top_importance:.2%}**"
        )

    st.subheader("💡 AI Interpretation")

    for _, row in xai_df.head(3).iterrows():
        st.write(
            f"• **{row['Feature']}** contributes "
            f"approximately **{row['Importance']:.2%}** "
            f"of the Random Forest feature importance."
        )

else:
    st.info(
        "Select **Random Forest** from the model selector to view "
        "the Explainable AI feature-impact analysis."
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.markdown("---")

st.header(
    "🔍 AI Sales Anomaly Detection"
)

st.write(
    "Isolation Forest is used to identify unusual sales transactions."
)

if len(df) >= 5:

    anomaly_features = df[
        ["Quantity", "Price", "Total_Sales"]
    ]

    anomaly_model = IsolationForest(
        contamination="auto",
        random_state=42
    )

    df["Anomaly"] = anomaly_model.fit_predict(
        anomaly_features
    )

    df["Anomaly_Status"] = df[
        "Anomaly"
    ].map(
        {
            1: "Normal",
            -1: "Anomaly"
        }
    )

    anomaly_count = (
        df["Anomaly"] == -1
    ).sum()

    normal_count = (
        df["Anomaly"] == 1
    ).sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "✅ Normal Transactions",
        normal_count
    )

    col2.metric(
        "⚠️ Anomalous Transactions",
        anomaly_count
    )

    st.dataframe(
        df[
            [
                "Order_ID",
                "Product",
                "Quantity",
                "Price",
                "Total_Sales",
                "Anomaly_Status"
            ]
        ],
        use_container_width=True
    )

# ============================================================
# AI SALES FORECASTING
# ============================================================

if show_forecast:

    st.markdown("---")

    st.header(
        "🔮 AI Sales Forecasting"
    )

    st.write(
        "Future sales are estimated using historical "
        "date-based sales data."
    )

    if "Date" in df.columns:

        try:

            forecast_df = df.copy()

            forecast_df["Date"] = pd.to_datetime(
                forecast_df["Date"],
                errors="coerce"
            )

            forecast_df = forecast_df.dropna(
                subset=["Date"]
            )

            daily_sales = (
                forecast_df
                .groupby("Date")["Total_Sales"]
                .sum()
                .reset_index()
            )

            prophet_df = daily_sales.rename(
                columns={
                    "Date": "ds",
                    "Total_Sales": "y"
                }
            )

            st.subheader(
                "📊 Historical Sales"
            )

            st.dataframe(
                daily_sales,
                use_container_width=True
            )

            if len(prophet_df) >= 2:

                from prophet import Prophet

                # --------------------------------------------
                # Forecast period
                # --------------------------------------------

                future_days = st.slider(
                    "Forecast Period (Days)",
                    min_value=7,
                    max_value=90,
                    value=30
                )

                # --------------------------------------------
                # Prophet Model
                # --------------------------------------------

                forecast_model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )

                forecast_model.fit(
                    prophet_df
                )

                # --------------------------------------------
                # Future Dates
                # --------------------------------------------

                future = (
                    forecast_model
                    .make_future_dataframe(
                        periods=future_days
                    )
                )

                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                forecast = forecast_model.predict(
                    future
                )

                # --------------------------------------------
                # Forecast Chart
                # --------------------------------------------

                st.subheader(
                    "📈 AI Sales Forecast"
                )

                fig5, ax5 = plt.subplots(
                    figsize=(12, 6)
                )

                ax5.plot(
                    prophet_df["ds"],
                    prophet_df["y"],
                    marker="o",
                    label="Actual Sales"
                )

                ax5.plot(
                    forecast["ds"],
                    forecast["yhat"],
                    label="Forecast Sales"
                )

                ax5.fill_between(
                    forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    alpha=0.2,
                    label="Confidence Interval"
                )

                ax5.set_title(
                    "AI-Powered Sales Forecast"
                )

                ax5.set_xlabel(
                    "Date"
                )

                ax5.set_ylabel(
                    "Revenue"
                )

                ax5.legend()

                st.pyplot(fig5)

                # --------------------------------------------
                # Future Forecast Table
                # --------------------------------------------

                future_forecast = forecast[
                    forecast["ds"]
                    > prophet_df["ds"].max()
                ][
                    [
                        "ds",
                        "yhat",
                        "yhat_lower",
                        "yhat_upper"
                    ]
                ].copy()

                future_forecast.columns = [
                    "Date",
                    "Predicted Sales",
                    "Lower Estimate",
                    "Upper Estimate"
                ]

                future_forecast[
                    "Predicted Sales"
                ] = future_forecast[
                    "Predicted Sales"
                ].clip(
                    lower=0
                )

                future_forecast[
                    "Lower Estimate"
                ] = future_forecast[
                    "Lower Estimate"
                ].clip(
                    lower=0
                )

                future_forecast[
                    "Upper Estimate"
                ] = future_forecast[
                    "Upper Estimate"
                ].clip(
                    lower=0
                )

                st.subheader(
                    "🔮 Future Sales Prediction"
                )

                st.dataframe(
                    future_forecast.round(2),
                    use_container_width=True
                )

                # --------------------------------------------
                # Forecast KPIs
                # --------------------------------------------

                total_forecast = (
                    future_forecast[
                        "Predicted Sales"
                    ].sum()
                )

                average_forecast = (
                    future_forecast[
                        "Predicted Sales"
                    ].mean()
                )

                highest_forecast = (
                    future_forecast[
                        "Predicted Sales"
                    ].max()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "🔮 Forecast Revenue",
                    f"₹{total_forecast:,.2f}"
                )

                col2.metric(
                    "📊 Average Daily Forecast",
                    f"₹{average_forecast:,.2f}"
                )

                col3.metric(
                    "📈 Highest Forecast",
                    f"₹{highest_forecast:,.2f}"
                )

                # --------------------------------------------
                # FORECAST VALIDATION & LIMITATIONS
                # --------------------------------------------

                st.subheader("🔍 Forecast Validation")

                st.info(
                    "⚠️ Demo/portfolio forecast: the current dataset contains "
                    f"only {len(prophet_df)} historical dates. With such a small "
                    "sample, forecast accuracy and seasonal patterns should not "
                    "be treated as production-level estimates."
                )

                historical_mean = prophet_df["y"].mean()
                forecast_mean = future_forecast["Predicted Sales"].mean()
                change_pct = ((forecast_mean - historical_mean) / historical_mean * 100) if historical_mean != 0 else 0

                v1, v2, v3 = st.columns(3)
                v1.metric("Historical Avg / Day", f"₹{historical_mean:,.2f}")
                v2.metric("Forecast Avg / Day", f"₹{forecast_mean:,.2f}")
                v3.metric("Forecast vs Historical", f"{change_pct:+.2f}%")

                # Forecast uncertainty
                future_forecast["Uncertainty Range"] = (
                    future_forecast["Upper Estimate"] - future_forecast["Lower Estimate"]
                )

                st.write("**Average forecast uncertainty range:** "
                         f"₹{future_forecast['Uncertainty Range'].mean():,.2f}")

                st.caption(
                    "For a production forecasting system, use a substantially larger "
                    "historical dataset and evaluate forecasts with time-based validation "
                    "(for example MAE, RMSE and MAPE on a held-out time period)."
                )

                # AI Insight
                # --------------------------------------------

                st.subheader(
                    "🤖 AI Forecast Insight"
                )

                st.info(
                    f"""
                    Based on the available historical data,
                    the model estimates an average future
                    daily revenue of approximately
                    ₹{average_forecast:,.2f}.

                    This forecast can support:

                    • Inventory planning
                    • Revenue planning
                    • Sales target setting
                    • Business decision support
                    """
                )

            else:

                st.warning(
                    "⚠️ At least 2 different dates "
                    "are required for forecasting."
                )

        except ImportError:

            st.error(
                "❌ Prophet is not installed."
            )

            st.code(
                "python -m pip install prophet"
            )

        except Exception as e:

            st.error(
                f"❌ Forecasting Error: {e}"
            )

    else:

        st.warning(
            "⚠️ Date column is required "
            "for AI Sales Forecasting."
        )

            # ============================================================
# 33. AI MODEL PERFORMANCE DASHBOARD
# ============================================================

st.markdown("---")
st.header("🏆 AI Model Performance Dashboard")

if show_ml and len(df) >= 5:

    st.write(
        "Comparison of different Machine Learning models "
        "based on prediction performance."
    )

    # Model performance chart
    performance_chart = results_df[
        ["Model", "MAE", "RMSE", "R2 Score"]
    ].copy()

    st.subheader("📊 Model Performance")

    st.dataframe(
        performance_chart.round(3),
        use_container_width=True
    )

    # R2 chart
    fig_performance, ax_performance = plt.subplots(
        figsize=(10, 5)
    )

    ax_performance.bar(
        performance_chart["Model"],
        performance_chart["R2 Score"]
    )

    ax_performance.set_title(
        "Model R² Score Comparison"
    )

    ax_performance.set_xlabel(
        "Machine Learning Model"
    )

    ax_performance.set_ylabel(
        "R² Score"
    )

    ax_performance.set_ylim(
        min(-1, performance_chart["R2 Score"].min() - 0.1),
        max(1, performance_chart["R2 Score"].max() + 0.1)
    )

    plt.xticks(rotation=20)

    st.pyplot(fig_performance)

    # AI explanation
    st.subheader("🤖 AI Model Analysis")

    best_r2_row = results_df.loc[
        results_df["R2 Score"].idxmax()
    ]

    lowest_mae_row = results_df.loc[
        results_df["MAE"].idxmin()
    ]

    st.info(
        f"""
        **Highest R² model in the current test split:**
        {best_r2_row["Model"]}

        **Lowest MAE model in the current test split:**
        {lowest_mae_row["Model"]}

        R² measures how much variation in the target is
        explained by the model, while MAE measures the
        average absolute prediction error.
        """
    )    

                # --------------------------------------------
# ============================================================
# AUTOMATED BUSINESS INSIGHTS
# ============================================================

st.markdown("---")

st.header(
    "💡 AI Business Insights"
)

# Top product insight
if "Product" in df.columns:

    top_product_name = (
        df.groupby("Product")["Total_Sales"]
        .sum()
        .idxmax()
    )

    top_product_revenue = (
        df.groupby("Product")["Total_Sales"]
        .sum()
        .max()
    )

    st.info(
        f"🏆 Top performing product: "
        f"**{top_product_name}** "
        f"with revenue of "
        f"₹{top_product_revenue:,.2f}"
    )

# Best city insight
if "City" in df.columns:

    best_city_name = (
        df.groupby("City")["Total_Sales"]
        .sum()
        .idxmax()
    )

    best_city_revenue = (
        df.groupby("City")["Total_Sales"]
        .sum()
        .max()
    )

    st.info(
        f"🌆 Highest revenue city: "
        f"**{best_city_name}** "
        f"with revenue of "
        f"₹{best_city_revenue:,.2f}"
    )

# Category insight
if "Category" in df.columns:

    best_category = (
        df.groupby("Category")["Total_Sales"]
        .sum()
        .idxmax()
    )

    st.info(
        f"🗂️ Highest revenue category: "
        f"**{best_category}**"
    )

# ============================================================
# DOWNLOAD PROCESSED DATA
# ============================================================

st.markdown("---")

st.header(
    "📥 Download Processed Dataset"
)

download_df = df.copy()

csv_data = download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Processed CSV",
    data=csv_data,
    file_name="processed_sales_data.csv",
    mime="text/csv"
)

# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown("---")

st.header(
    "📌 Project Summary"
)

st.markdown(
    """
    ### 🤖 AI-Powered Sales Intelligence System

    **Technologies Used:**

    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - Streamlit
    - Prophet

    **Machine Learning:**

    - Linear Regression
    - Random Forest Regression
    - Gradient Boosting
    - Isolation Forest

    **AI Features:**

    - Sales Prediction
    - Model Comparison
    - Best Model Selection
    - Sales Anomaly Detection
    - Feature Importance
    - Sales Forecasting
    - Automated Business Insights

    **Developed as a Data Science Portfolio Project**
    """
)

st.markdown("---")

st.caption(
    "🚀 AI Sales Intelligence & Prediction System"
)
