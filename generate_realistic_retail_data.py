import os
import numpy as np
import pandas as pd


def generate_realistic_retail_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    n_stores: int = 40,
    n_products: int = 120,
    random_state: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_state)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # ----------------------------
    # 1. Store master
    # ----------------------------
    store_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_stores + 1)]
    store_clusters = ["Urban High Income", "Township Value", "Suburban Family", "Rural Mixed"]

    store_cluster_probs = [0.22, 0.28, 0.30, 0.20]
    assigned_clusters = np.random.choice(store_clusters, size=n_stores, p=store_cluster_probs)

    store_master = pd.DataFrame({
        "store_id": store_ids,
        "store_cluster": assigned_clusters,
        "region": np.random.choice(
            ["Western Cape", "Gauteng", "KwaZulu-Natal", "Eastern Cape", "Limpopo"],
            size=n_stores
        ),
        "store_size": np.random.choice(["Small", "Medium", "Large"], size=n_stores, p=[0.25, 0.5, 0.25]),
    })

    cluster_store_multiplier = {
        "Urban High Income": 1.25,
        "Township Value": 1.10,
        "Suburban Family": 1.15,
        "Rural Mixed": 0.90,
    }

    cluster_price_sensitivity = {
        "Urban High Income": 0.75,
        "Township Value": 1.35,
        "Suburban Family": 1.00,
        "Rural Mixed": 1.15,
    }

    # ----------------------------
    # 2. Product master
    # ----------------------------
    product_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_products + 1)]
    categories = ["Beverages", "Snacks", "Household", "Personal Care", "Dairy", "Bakery"]

    category_probs = [0.18, 0.18, 0.18, 0.14, 0.18, 0.14]
    product_categories = np.random.choice(categories, size=n_products, p=category_probs)

    product_master = pd.DataFrame({
        "product_id": product_ids,
        "category": product_categories,
    })

    category_base_price = {
        "Beverages": (18, 45),
        "Snacks": (10, 35),
        "Household": (35, 120),
        "Personal Care": (30, 110),
        "Dairy": (12, 50),
        "Bakery": (8, 28),
    }

    category_base_demand = {
        "Beverages": 32,
        "Snacks": 28,
        "Household": 16,
        "Personal Care": 14,
        "Dairy": 26,
        "Bakery": 30,
    }

    category_elasticity = {
        "Beverages": -1.15,
        "Snacks": -1.30,
        "Household": -0.95,
        "Personal Care": -0.85,
        "Dairy": -1.10,
        "Bakery": -1.40,
    }

    category_seasonality = {
        "Beverages": 1.15,
        "Snacks": 1.05,
        "Household": 1.00,
        "Personal Care": 1.00,
        "Dairy": 1.08,
        "Bakery": 1.03,
    }

    base_prices = []
    base_costs = []
    premium_flags = []

    for cat in product_master["category"]:
        low, high = category_base_price[cat]
        price = np.random.uniform(low, high)
        premium = np.random.binomial(1, 0.22)
        if premium == 1:
            price *= np.random.uniform(1.15, 1.55)
        cost = price * np.random.uniform(0.58, 0.78)

        base_prices.append(round(price, 2))
        base_costs.append(round(cost, 2))
        premium_flags.append(premium)

    product_master["base_price"] = base_prices
    product_master["unit_cost"] = base_costs
    product_master["premium_flag"] = premium_flags

    # ----------------------------
    # 3. Calendar / events
    # ----------------------------
    calendar = pd.DataFrame({"date": dates})
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["month"] = calendar["date"].dt.month
    calendar["day"] = calendar["date"].dt.day
    calendar["weekend_flag"] = calendar["day_of_week"].isin([5, 6]).astype(int)

    # Simple holiday / payday logic
    holiday_dates = pd.to_datetime([
        "2024-01-01", "2024-03-21", "2024-03-29", "2024-04-01",
        "2024-04-27", "2024-05-01", "2024-06-16", "2024-08-09",
        "2024-09-24", "2024-12-16", "2024-12-25", "2024-12-26"
    ])
    calendar["holiday_flag"] = calendar["date"].isin(holiday_dates).astype(int)
    calendar["month_end_flag"] = (calendar["day"] >= 25).astype(int)
    calendar["payday_window_flag"] = calendar["day"].isin([24, 25, 26, 27, 28, 29, 30, 31, 1, 2]).astype(int)

    # December / summer uplift proxy
    calendar["season_index"] = (
        1
        + 0.10 * np.sin(2 * np.pi * calendar.index / 365.25)
        + 0.07 * (calendar["month"].isin([11, 12])).astype(int)
        + 0.04 * (calendar["month"].isin([1, 2])).astype(int)
    )

    # ----------------------------
    # 4. Build full panel
    # ----------------------------
    df = (
        calendar.assign(key=1)
        .merge(store_master.assign(key=1), on="key")
        .merge(product_master.assign(key=1), on="key")
        .drop(columns="key")
    )

    # ----------------------------
    # 5. Promotion logic
    # ----------------------------
    cluster_promo_prob = {
        "Urban High Income": 0.14,
        "Township Value": 0.26,
        "Suburban Family": 0.20,
        "Rural Mixed": 0.18,
    }

    promo_prob = (
        df["store_cluster"].map(cluster_promo_prob)
        + 0.05 * df["month_end_flag"]
        + 0.03 * df["payday_window_flag"]
    ).clip(0.05, 0.55)

    df["promo_flag"] = np.random.binomial(1, promo_prob)

    promo_discount = np.where(
        df["promo_flag"] == 1,
        np.random.uniform(0.05, 0.22, len(df)),
        0.0
    )

    # ----------------------------
    # 6. Price and competitor price
    # ----------------------------
    cluster_price_level = {
        "Urban High Income": 1.06,
        "Township Value": 0.96,
        "Suburban Family": 1.01,
        "Rural Mixed": 0.98,
    }

    regional_price_noise = np.random.normal(1.0, 0.025, len(df))
    daily_price_noise = np.random.normal(1.0, 0.03, len(df))

    df["list_price"] = (
        df["base_price"]
        * df["store_cluster"].map(cluster_price_level)
        * regional_price_noise
        * daily_price_noise
    ).clip(lower=5)

    df["price"] = (df["list_price"] * (1 - promo_discount)).round(2)

    competitor_gap = np.random.normal(1.0, 0.05, len(df))
    df["competitor_price"] = (df["list_price"] * competitor_gap).round(2)

    # ----------------------------
    # 7. Stock availability
    # ----------------------------
    category_stock_base = {
        "Beverages": 180,
        "Snacks": 160,
        "Household": 90,
        "Personal Care": 75,
        "Dairy": 120,
        "Bakery": 100,
    }

    size_multiplier = {
        "Small": 0.80,
        "Medium": 1.00,
        "Large": 1.30,
    }

    stock_base = (
        df["category"].map(category_stock_base)
        * df["store_size"].map(size_multiplier)
        * np.random.uniform(0.75, 1.25, len(df))
    )

    df["stock_available"] = np.maximum(5, stock_base.round()).astype(int)

    # ----------------------------
    # 8. Demand generation
    # ----------------------------
    df["base_demand"] = df["category"].map(category_base_demand).astype(float)
    df["store_multiplier"] = df["store_cluster"].map(cluster_store_multiplier).astype(float)
    df["cat_elasticity"] = df["category"].map(category_elasticity).astype(float)
    df["cluster_sensitivity"] = df["store_cluster"].map(cluster_price_sensitivity).astype(float)
    df["seasonality_multiplier"] = (
        df["season_index"] * df["category"].map(category_seasonality)
    )

    # Relative price vs product base
    df["price_index_vs_base"] = (df["price"] / df["base_price"]).clip(lower=0.4, upper=2.2)
    df["competitor_index"] = (df["competitor_price"] / df["price"]).clip(lower=0.6, upper=1.5)

    # Promo and timing effects
    promo_effect = 1 + (0.18 * df["promo_flag"])
    holiday_effect = 1 + (0.20 * df["holiday_flag"])
    payday_effect = 1 + (0.08 * df["payday_window_flag"])
    weekend_effect = 1 + (
        0.06 * df["weekend_flag"] * df["category"].isin(["Beverages", "Snacks", "Bakery"]).astype(int)
    )

    # Price response
    effective_elasticity = df["cat_elasticity"] * df["cluster_sensitivity"]
    price_effect = np.power(df["price_index_vs_base"], effective_elasticity)

    # Competitor price effect:
    # if competitor is more expensive than us, we gain demand
    competitor_effect = np.power(df["competitor_index"], 0.35)

    # Premium products slightly lower volume
    premium_effect = np.where(df["premium_flag"] == 1, 0.82, 1.0)

    # Noise
    random_noise = np.random.lognormal(mean=0.0, sigma=0.18, size=len(df))

    raw_demand = (
        df["base_demand"]
        * df["store_multiplier"]
        * df["seasonality_multiplier"]
        * promo_effect
        * holiday_effect
        * payday_effect
        * weekend_effect
        * price_effect
        * competitor_effect
        * premium_effect
        * random_noise
    )

    df["units_demanded"] = np.maximum(0, raw_demand.round()).astype(int)
    df["units_sold"] = np.minimum(df["units_demanded"], df["stock_available"]).astype(int)
    df["stockout_flag"] = (df["units_demanded"] > df["stock_available"]).astype(int)

    # ----------------------------
    # 9. Financial outputs
    # ----------------------------
    df["revenue"] = (df["price"] * df["units_sold"]).round(2)
    df["gross_profit"] = ((df["price"] - df["unit_cost"]) * df["units_sold"]).round(2)
    df["margin_pct"] = np.where(df["revenue"] > 0, df["gross_profit"] / df["revenue"], 0).round(4)

    # ----------------------------
    # 10. Final clean-up
    # ----------------------------
    final_cols = [
        "date",
        "store_id",
        "region",
        "store_size",
        "store_cluster",
        "product_id",
        "category",
        "premium_flag",
        "day_of_week",
        "month",
        "weekend_flag",
        "holiday_flag",
        "month_end_flag",
        "payday_window_flag",
        "promo_flag",
        "base_price",
        "list_price",
        "price",
        "competitor_price",
        "unit_cost",
        "stock_available",
        "units_demanded",
        "units_sold",
        "stockout_flag",
        "revenue",
        "gross_profit",
        "margin_pct",
    ]

    return df[final_cols].sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_realistic_retail_data()
    output_path = "data/realistic_retail_pricing_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nSample:")
    print(df.head())

    print("\nCategory summary:")
    print(
        df.groupby("category")[["units_sold", "revenue", "gross_profit"]]
        .mean()
        .round(2)
    )

    print("\nStore cluster summary:")
    print(
        df.groupby("store_cluster")[["units_sold", "revenue", "gross_profit", "margin_pct"]]
        .mean()
        .round(2)
    )