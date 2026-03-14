import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/realistic_retail_pricing_data.csv")

# Remove invalid rows
df = df[(df["units_sold"] > 0) & (df["price"] > 0)]

# Log transformation
df["log_units"] = np.log(df["units_sold"])
df["log_price"] = np.log(df["price"])

# Simple regression using numpy
X = df["log_price"].values
Y = df["log_units"].values

X = np.column_stack((np.ones(len(X)), X))

beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)

intercept = beta[0]
elasticity = beta[1]

print("Price Elasticity:", elasticity)

# Predict units
pred_log_units = X @ beta
pred_units = np.exp(pred_log_units)

# Plot price vs units
plt.scatter(df["price"], df["units_sold"], alpha=0.3)
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.title("Price vs Demand")
plt.savefig("price_vs_demand.png")
plt.show()

# Revenue simulation
prices = np.linspace(df["price"].min(), df["price"].max(), 50)
revenues = []

for p in prices:
    predicted_units = np.exp(intercept + elasticity * np.log(p))
    revenue = p * predicted_units
    revenues.append(revenue)

plt.plot(prices, revenues)
plt.xlabel("Price")
plt.ylabel("Revenue")
plt.title("Revenue vs Price Simulation")
plt.savefig("revenue_simulation.png")
plt.show()

print("Analysis complete")