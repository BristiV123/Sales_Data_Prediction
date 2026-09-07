# =====================================================
# SALES DATA ANALYSIS & PREDICTION SYSTEM
# STREAMLIT DASHBOARD
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sales Data Analysis",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# 2. TITLE
# =====================================================

st.title("📊 Sales Data Analysis & Prediction System")

st.write(
    "Interactive dashboard for analyzing sales data."
)


# =====================================================
# 3. LOAD DATASET
# =====================================================

try:
    data = pd.read_csv("sales_data.csv")

except FileNotFoundError:
    st.error("❌ sales_data.csv file not found!")
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()


# =====================================================
# 4. DATA CLEANING
# =====================================================

# Remove duplicate rows
data = data.drop_duplicates()

# Convert Quantity and Price to numeric
data["Quantity"] = pd.to_numeric(
    data["Quantity"],
    errors="coerce"
)

data["Price"] = pd.to_numeric(
    data["Price"],
    errors="coerce"
)

# Remove missing values
data = data.dropna(
    subset=["Quantity", "Price"]
)

# Remove invalid values
data = data[
    (data["Quantity"] > 0) &
    (data["Price"] > 0)
]

# Reset index
data = data.reset_index(drop=True)


# =====================================================
# 5. FEATURE ENGINEERING
# =====================================================

data["Total_Sales"] = (
    data["Quantity"] *
    data["Price"]
)

data["Revenue_Per_Unit"] = (
    data["Total_Sales"] /
    data["Quantity"]
)


# =====================================================
# 6. CALCULATIONS
# =====================================================

total_sales = data["Total_Sales"].sum()

average_sales = data["Total_Sales"].mean()

maximum_sale = data["Total_Sales"].max()

minimum_sale = data["Total_Sales"].min()


# Product-wise sales
product_sales = (
    data.groupby("Product")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

top_product = product_sales.idxmax()

top_product_sales = product_sales.max()


# City-wise sales
city_sales = (
    data.groupby("City")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

best_city = city_sales.idxmax()

best_city_sales = city_sales.max()


# =====================================================
# 7. SIDEBAR
# =====================================================

st.sidebar.title("📌 Dashboard Menu")

st.sidebar.write(
    "Sales Data Analysis Dashboard"
)

show_data = st.sidebar.checkbox(
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


# =====================================================
# 8. KEY PERFORMANCE INDICATORS
# =====================================================

st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Total Revenue",
        f"₹{total_sales:,.2f}"
    )

with col2:
    st.metric(
        "🛍️ Top Product",
        top_product
    )

with col3:
    st.metric(
        "🏙️ Best City",
        best_city
    )

with col4:
    st.metric(
        "📊 Average Sale",
        f"₹{average_sales:,.2f}"
    )


# =====================================================
# 9. DATASET
# =====================================================

if show_data:

    st.subheader("📋 Dataset")

    st.write(
        f"Total Records: **{len(data)}**"
    )

    st.dataframe(
        data,
       width="stretch"
    )


# =====================================================
# 10. SALES SUMMARY
# =====================================================

if show_analysis:

    st.subheader("📊 Sales Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Maximum Sale",
            f"₹{maximum_sale:,.2f}"
        )

    with col2:
        st.metric(
            "Minimum Sale",
            f"₹{minimum_sale:,.2f}"
        )

    with col3:
        st.metric(
            "Best City Sales",
            f"₹{best_city_sales:,.2f}"
        )


# =====================================================
# 11. PRODUCT-WISE ANALYSIS
# =====================================================

if show_analysis:

    st.subheader("🛍️ Product-wise Sales")

    product_df = product_sales.reset_index()

    product_df.columns = [
        "Product",
        "Total Sales"
    ]

    st.dataframe(
        product_df,
        width="stretch"
    )

    st.success(
        f"🏆 Top Selling Product: {top_product}"
    )


# =====================================================
# 12. CITY-WISE ANALYSIS
# =====================================================

if show_analysis:

    st.subheader("🏙️ City-wise Sales")

    city_df = city_sales.reset_index()

    city_df.columns = [
        "City",
        "Total Sales"
    ]

    st.dataframe(
        city_df,
        width="stretch"
    )

    st.success(
        f"🏆 Best Performing City: {best_city}"
    )


# =====================================================
# 13. CHARTS
# =====================================================

if show_charts:

    st.subheader("📊 Sales Visualizations")


    # -------------------------------------------------
    # Product-wise Sales Chart
    # -------------------------------------------------

    st.write("### 🛍️ Product-wise Total Sales")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=product_sales.index,
        y=product_sales.values,
        ax=ax1
    )

    ax1.set_title(
        "Product-wise Total Sales"
    )

    ax1.set_xlabel("Product")

    ax1.set_ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig1)

    plt.close(fig1)


    # -------------------------------------------------
    # City-wise Sales Chart
    # -------------------------------------------------

    st.write("### 🏙️ City-wise Total Sales")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=city_sales.index,
        y=city_sales.values,
        ax=ax2
    )

    ax2.set_title(
        "City-wise Total Sales"
    )

    ax2.set_xlabel("City")

    ax2.set_ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # -------------------------------------------------
    # Sales Distribution
    # -------------------------------------------------

    st.write("### 📈 Sales Distribution")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data["Total_Sales"],
        bins=10,
        kde=True,
        ax=ax3
    )

    ax3.set_title(
        "Sales Distribution"
    )

    ax3.set_xlabel("Total Sales")

    ax3.set_ylabel("Frequency")

    plt.tight_layout()

    st.pyplot(fig3)

    plt.close(fig3)


    # -------------------------------------------------
    # City-wise Sales Pie Chart
    # -------------------------------------------------

    st.write("### 🥧 City-wise Sales Distribution")

    fig4, ax4 = plt.subplots(figsize=(8, 8))

    city_sales.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax4
    )

    ax4.set_title(
        "City-wise Sales Distribution"
    )

    ax4.set_ylabel("")

    plt.tight_layout()

    st.pyplot(fig4)

    plt.close(fig4)


# =====================================================
# 14. BUSINESS INSIGHTS
# =====================================================

st.subheader("💡 Business Insights")

st.write(
    f"🔹 Total Revenue: **₹{total_sales:,.2f}**"
)

st.write(
    f"🔹 Top Selling Product: **{top_product}**"
)

st.write(
    f"🔹 Top Product Sales: **₹{top_product_sales:,.2f}**"
)

st.write(
    f"🔹 Best Performing City: **{best_city}**"
)

st.write(
    f"🔹 Best City Sales: **₹{best_city_sales:,.2f}**"
)

st.write(
    f"🔹 Average Sale: **₹{average_sales:,.2f}**"
)


# =====================================================
# 15. PROJECT STATUS
# =====================================================

st.divider()

st.success(
    "✅ Sales Data Analysis Dashboard is running successfully!"
)

st.info(
    "🚀 Machine Learning Sales Prediction will be added in the next step."
)

# ============================================================
# MACHINE LEARNING - SALES PREDICTION
# ============================================================

st.header("🤖 Sales Prediction")

st.write("Enter Quantity and Price to predict Total Sales.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Features and target
X = data[["Quantity", "Price"]]
y = data["Total_Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

st.subheader("💰 Predict Sales")

col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input(
        "Enter Quantity",
        min_value=1,
        value=1,
        step=1
    )

with col2:
    price = st.number_input(
        "Enter Price",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

if st.button("🔮 Predict Sales"):

    input_data = pd.DataFrame({
        "Quantity": [quantity],
        "Price": [price]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Total Sales: ₹{prediction:,.2f}"
    )

    # ============================================================
# ACTUAL VS PREDICTED SALES
# ============================================================

st.subheader("📊 Actual vs Predicted Sales")

comparison_df = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

st.dataframe(
    comparison_df,
    width="stretch"
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    comparison_df["Actual Sales"].values,
    label="Actual Sales",
    marker="o"
)

ax.plot(
    comparison_df["Predicted Sales"].values,
    label="Predicted Sales",
    marker="x"
)

ax.set_title("Actual vs Predicted Sales")
ax.set_xlabel("Test Data Index")
ax.set_ylabel("Total Sales")
ax.legend()

st.pyplot(fig)

# ============================================================
# MODEL SUMMARY
# ============================================================

st.subheader("📋 Machine Learning Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🤖 Model")
    st.write("Linear Regression")

with col2:
    st.info("📚 Training Data")
    st.write(f"{len(X_train)} records")

with col3:
    st.info("🧪 Testing Data")
    st.write(f"{len(X_test)} records")


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Business Insights")

st.write(
    f"🔹 Total Revenue: ₹{total_sales:,.2f}"
)

st.write(
    f"🔹 Average Sale: ₹{average_sales:,.2f}"
)

st.write(
    f"🔹 Best Performing Product: {top_product}"
)

st.write(
    f"🔹 Best Performing City: {best_city}"
)

st.write(
    "🔹 Linear Regression is used for sales prediction."
)

st.write(
    f"🔹 Model R² Score: {r2:.2f}"
)