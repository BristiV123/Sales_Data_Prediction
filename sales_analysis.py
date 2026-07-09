#Python LIbraries Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv("sales_data.csv")

# Display first 5 rows
print(data.head())

#Total Sales Coloumn
# Create Total Sales column
data["Total_Sales"] = data["Quantity"] * data["Price"]

# Display updated dataset
print(data.head())

#Total Sales And Top Selling Product
# Calculate overall sales
total_sales = data["Total_Sales"].sum()
print("Total Sales:", total_sales)

# Calculate sales by product
product_sales = data.groupby("Product")["Total_Sales"].sum()

print("\nSales by Product:")
print(product_sales)

# Find the top-selling product
top_product = product_sales.idxmax()
top_sales = product_sales.max()

print("\nTop Selling Product:", top_product)
print("Top Sales:", top_sales)

#Product-wise sales Bar Chart Create
# Product-wise Sales Bar Chart

product_sales.plot(kind="bar", figsize=(8,5))

plt.title("Product-wise Total Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.show()

#City Wise Sales Analysis
# Calculate total sales by city
city_sales = data.groupby("City")["Total_Sales"].sum()

print("\nCity-wise Sales:")
print(city_sales)

# City-wise Sales Bar Chart

city_sales.plot(kind="bar", figsize=(8,5))

plt.title("City-wise Total Sales")
plt.xlabel("City")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.show()

#Pie Chart And Business Insights
# City-wise Sales Pie Chart

city_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(7,7)
)

plt.title("City-wise Sales Distribution")
plt.ylabel("")
plt.show()

# Business Insights

print("\n===== Business Insights =====")

print("Total Sales:", total_sales)

print("Top Selling Product:", top_product)

print("Highest Sales:", top_sales)

print("Best Performing City:", city_sales.idxmax())

print("Sales in Best City:", city_sales.max())