# =====================================================
# SALES DATA ANALYSIS PROJECT
# =====================================================

# Python Libraries Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# =====================================================
# 1. LOAD DATASET
# =====================================================

try:
    data = pd.read_csv("sales_data.csv")
    print("✅ Dataset loaded successfully!")

except FileNotFoundError:
    print("❌ Error: sales_data.csv file not found.")
    exit()

except Exception as e:
    print("❌ An error occurred:", e)
    exit()


# =====================================================
# 2. DATASET INFORMATION
# =====================================================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nFirst 5 Rows:")
print(data.head())

print("\nNumber of Rows:", data.shape[0])
print("Number of Columns:", data.shape[1])

print("\nColumn Names:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())


# =====================================================
# 3. DATA CLEANING
# =====================================================

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Check duplicate rows
duplicates = data.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

# Remove duplicate rows
data = data.drop_duplicates()

# Missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Convert Quantity and Price to numeric
data["Quantity"] = pd.to_numeric(
    data["Quantity"],
    errors="coerce"
)

data["Price"] = pd.to_numeric(
    data["Price"],
    errors="coerce"
)

# Remove missing Quantity and Price
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

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

print("\nCleaned Dataset Shape:")
print(data.shape)

print("\n✅ Data Cleaning Completed Successfully!")


# =====================================================
# 4. TOTAL SALES CALCULATION
# =====================================================

data["Total_Sales"] = (
    data["Quantity"] *
    data["Price"]
)

total_sales = data["Total_Sales"].sum()

print("\n" + "=" * 50)
print("SALES CALCULATION")
print("=" * 50)

print("\nTotal Sales:", total_sales)


# =====================================================
# 5. PRODUCT-WISE SALES ANALYSIS
# =====================================================

product_sales = (
    data.groupby("Product")["Total_Sales"]
    .sum()
)

print("\n" + "=" * 50)
print("PRODUCT-WISE SALES")
print("=" * 50)

print("\nSales by Product:")
print(product_sales)

# Top-selling product
top_product = product_sales.idxmax()
top_sales = product_sales.max()

print("\nTop Selling Product:", top_product)
print("Top Product Sales:", top_sales)


# =====================================================
# 6. CITY-WISE SALES ANALYSIS
# =====================================================

city_sales = (
    data.groupby("City")["Total_Sales"]
    .sum()
)

print("\n" + "=" * 50)
print("CITY-WISE SALES")
print("=" * 50)

print("\nSales by City:")
print(city_sales)

# Best performing city
best_city = city_sales.idxmax()
best_city_sales = city_sales.max()

print("\nBest Performing City:", best_city)
print("Best City Sales:", best_city_sales)

# =====================================================
# 7. FEATURE ENGINEERING
# =====================================================

print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Total Sales Feature
data["Total_Sales"] = data["Quantity"] * data["Price"]

# Revenue per Unit
data["Revenue_Per_Unit"] = (
    data["Total_Sales"] / data["Quantity"]
)

print("\nNew Features Created:")
print("- Total_Sales")
print("- Revenue_Per_Unit")

print("\nUpdated Dataset:")
print(data.head())

print("\nUpdated Columns:")
print(data.columns.tolist())

print("\n✅ Feature Engineering Completed Successfully!")


# =====================================================
# 8. EDA - BASIC SALES STATISTICS
# =====================================================

print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Sales Statistics
print("\nSales Statistics:")
print(data["Total_Sales"].describe())

# 2. Average Sales
average_sales = data["Total_Sales"].mean()
print("\nAverage Sales:", average_sales)

# 3. Maximum Sale
maximum_sale = data["Total_Sales"].max()
print("Maximum Sale:", maximum_sale)

# 4. Minimum Sale
minimum_sale = data["Total_Sales"].min()
print("Minimum Sale:", minimum_sale)

# 5. Product-wise Sales
print("\nProduct-wise Sales:")
print(product_sales)

# 6. City-wise Sales
print("\nCity-wise Sales:")
print(city_sales)

# 7. Sales Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data["Total_Sales"],
    bins=10,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Total Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "sales_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n✅ EDA Completed Successfully!")

# =====================================================
# 9. MACHINE LEARNING - DATA PREPARATION
# =====================================================

print("\n" + "=" * 50)
print("MACHINE LEARNING DATA PREPARATION")
print("=" * 50)

# Features
X = data[["Quantity", "Price"]]

# Target
y = data["Total_Sales"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\n✅ Machine Learning Data Preparation Completed!")

# =====================================================
# 10. TRAIN-TEST SPLIT & LINEAR REGRESSION
# =====================================================

print("\n" + "=" * 50)
print("TRAIN-TEST SPLIT & LINEAR REGRESSION")
print("=" * 50)

# Features
X = data[["Quantity", "Price"]]

# Target
y = data["Total_Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("\n✅ Linear Regression Model Trained Successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nActual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(y_pred)

print("\n✅ Sales Prediction Completed!")



# =====================================================
# 11. PRODUCT-WISE SALES BAR CHART
# =====================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x=product_sales.index,
    y=product_sales.values
)

plt.title("Product-wise Total Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "product_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 12. CITY-WISE SALES BAR CHART
# =====================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x=city_sales.index,
    y=city_sales.values
)

plt.title("City-wise Total Sales")
plt.xlabel("City")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "city_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 13. SALES DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data["Total_Sales"],
    bins=10,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Total Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "sales_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 14. CITY-WISE SALES PIE CHART
# =====================================================

plt.figure(figsize=(7, 7))

city_sales.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("City-wise Sales Distribution")
plt.ylabel("")

plt.tight_layout()

plt.savefig(
    "city_sales_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 15. BUSINESS INSIGHTS
# =====================================================

print("\n" + "=" * 50)
print("BUSINESS INSIGHTS")
print("=" * 50)

print("\nTotal Revenue: ₹", total_sales)

print("Top Selling Product:", top_product)

print("Highest Product Sales: ₹", top_sales)

print("Best Performing City:", best_city)

print("Sales in Best City: ₹", best_city_sales)

print("Average Sales: ₹", average_sales)


# =====================================================
# PROJECT COMPLETED
# =====================================================

print("\n" + "=" * 50)
print("✅ SALES DATA ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 50)