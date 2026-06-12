"""
data/generate_data.py
Generates a synthetic sales dataset with 2,000 transactions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

CATEGORIES = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden"]
VENDORS    = ["Alice", "Bob", "Carlos", "Diana", "Eduardo", "Fiona"]
REGIONS    = ["North", "South", "East", "West", "Central"]
CHANNELS   = ["Online", "Store", "Phone", "Partner"]

PRODUCTS = {
    "Electronics":     ["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor"],
    "Clothing":        ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers"],
    "Food & Beverage": ["Coffee", "Tea", "Snack Box", "Juice Pack", "Energy Bar"],
    "Home & Garden":   ["Lamp", "Plant Pot", "Rug", "Curtain", "Pillow Set"],
}

BASE_PRICES = {
    "Laptop": 1200, "Smartphone": 800, "Tablet": 500, "Headphones": 150, "Monitor": 350,
    "T-Shirt": 25,  "Jeans": 60,      "Jacket": 120,  "Dress": 90,      "Sneakers": 110,
    "Coffee": 18,   "Tea": 12,        "Snack Box": 30, "Juice Pack": 15, "Energy Bar": 8,
    "Lamp": 45,     "Plant Pot": 20,  "Rug": 80,       "Curtain": 55,    "Pillow Set": 40,
}

def generate():
    dates = pd.date_range("2023-01-01", "2024-12-31", periods=2000)
    records = []
    for date in dates:
        cat      = np.random.choice(CATEGORIES)
        product  = np.random.choice(PRODUCTS[cat])
        base     = BASE_PRICES[product]
        qty      = np.random.randint(1, 10)
        discount = round(np.random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)
        price    = round(base * (1 - discount) * np.random.uniform(0.9, 1.1), 2)
        month_factor = 1 + 0.3 * np.sin((date.month - 1) * np.pi / 6)
        revenue  = round(price * qty * month_factor, 2)
        records.append({
            "date":       date.strftime("%Y-%m-%d"),
            "category":   cat,
            "product":    product,
            "vendor":     np.random.choice(VENDORS),
            "region":     np.random.choice(REGIONS),
            "channel":    np.random.choice(CHANNELS),
            "quantity":   qty,
            "unit_price": price,
            "discount":   discount,
            "revenue":    revenue,
        })
    df = pd.DataFrame(records)
    out = Path(__file__).parent / "sales.csv"
    df.to_csv(out, index=False)
    print(f"✅ Dataset created: {out} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    generate()
