"""
Generate a realistic synthetic sales dataset with 10,000+ rows.

Columns: order_id, date, customer_id, product_category, product_name,
         quantity, unit_price, revenue, region, salesperson
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Reproducible randomness
np.random.seed(42)

# --- Configuration ---
NUM_RECORDS = 12000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data", "sales_raw.csv")

# --- Lookups ---
REGIONS = ["East", "West", "North", "South"]
REGION_PROB = [0.28, 0.30, 0.22, 0.20]

PRODUCT_CATALOG = {
    "Electronics": [
        ("Wireless Headphones", 49.99, 89.99),
        ("USB-C Charging Cable", 9.99, 19.99),
        ("Bluetooth Speaker", 29.99, 79.99),
        ("Smart Watch", 149.99, 299.99),
        ("Portable Power Bank", 19.99, 49.99),
        ("4K Monitor", 249.99, 449.99),
        ("Mechanical Keyboard", 59.99, 129.99),
        ("Wireless Mouse", 19.99, 39.99),
    ],
    "Clothing": [
        ("Cotton T-Shirt", 14.99, 34.99),
        ("Running Shoes", 49.99, 129.99),
        ("Denim Jeans", 29.99, 69.99),
        ("Winter Jacket", 59.99, 149.99),
        ("Wool Sweater", 39.99, 89.99),
        ("Baseball Cap", 9.99, 24.99),
    ],
    "Home & Garden": [
        ("LED Desk Lamp", 19.99, 49.99),
        ("Stainless Steel Cookware Set", 49.99, 149.99),
        ("Yoga Mat", 14.99, 39.99),
        ("Plant Pot Set", 9.99, 29.99),
        ("Throw Blanket", 19.99, 54.99),
        ("Coffee Maker", 39.99, 99.99),
        ("Scented Candle Pack", 12.99, 29.99),
    ],
    "Sports & Outdoors": [
        ("Camping Tent", 79.99, 249.99),
        ("Hiking Backpack", 39.99, 119.99),
        ("Resistance Bands", 9.99, 24.99),
        ("Dumbbell Set", 29.99, 99.99),
        ("Water Bottle", 9.99, 29.99),
        ("Fishing Rod", 29.99, 79.99),
    ],
    "Books & Media": [
        ("Paperback Novel", 8.99, 16.99),
        ("Cookbook", 19.99, 34.99),
        ("Educational DVD Set", 14.99, 39.99),
        ("Art Supply Kit", 24.99, 59.99),
        ("Board Game", 19.99, 49.99),
    ],
    "Office Supplies": [
        ("Printer Paper (Ream)", 5.99, 12.99),
        ("Ballpoint Pen Pack", 3.99, 9.99),
        ("Filing Cabinet", 49.99, 129.99),
        ("Whiteboard", 19.99, 59.99),
        ("Ergonomic Office Chair", 149.99, 349.99),
    ],
}

CATEGORY_PROB = [0.22, 0.18, 0.18, 0.15, 0.10, 0.17]

SALESPERSONS = [
    "Alice Chen", "Bob Martinez", "Carol Williams",
    "David Kim", "Eva Thompson", "Frank Davis",
    "Grace Liu", "Henry Foster", "Irene Okafor",
    "James Wilson"
]

REGION_SALESPERSON_MAP = {
    "East": ["Alice Chen", "David Kim", "Grace Liu"],
    "West": ["Bob Martinez", "Eva Thompson", "Henry Foster"],
    "North": ["Carol Williams", "Frank Davis"],
    "South": ["Irene Okafor", "James Wilson"],
}

NUM_CUSTOMERS = 800  # Repeated customers across the 12k rows


def _dates_range(start, end):
    """Return a random date between start and end."""
    delta_days = (end - start).days
    return start + timedelta(days=int(np.random.randint(0, delta_days)))


def generate_dataset(n=NUM_RECORDS):
    """Generate the full synthetic sales dataset."""

    # --- Dates ---
    dates = np.array([_dates_range(START_DATE, END_DATE) for _ in range(n)])

    # --- Customers and Regions ---
    customer_ids = np.random.choice(
        [f"C-{str(i).zfill(5)}" for i in range(1, NUM_CUSTOMERS + 1)],
        size=n,
    )
    regions = np.random.choice(REGIONS, size=n, p=REGION_PROB)

    # --- Product Category & Product ---
    categories = np.random.choice(
        list(PRODUCT_CATALOG.keys()), size=n, p=CATEGORY_PROB
    )
    product_names = []
    unit_prices = []
    for cat in categories:
        items = PRODUCT_CATALOG[cat]
        name, low_p, high_p = items[np.random.randint(0, len(items))]
        product_names.append(name)
        unit_prices.append(round(np.random.uniform(low_p, high_p), 2))

    # --- Quantity ---
    # Most orders 1-5 items, occasional larger ones
    _q_probs = np.array([0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01,
                         0.008, 0.005, 0.004, 0.002, 0.0005, 0.0005])
    _q_probs /= _q_probs.sum()  # normalize
    quantity = np.random.choice(
        np.arange(1, 16),
        size=n,
        p=_q_probs,
    )

    # --- Salesperson (region-dependent) ---
    salesperson = []
    for r in regions:
        salesperson.append(np.random.choice(REGION_SALESPERSON_MAP[r]))

    # --- Revenue ---
    revenue = [round(q * p, 2) for q, p in zip(quantity, unit_prices)]

    # --- Inject some missing values (~2%) to test ETL cleaning ---
    order_ids = [f"ORD-{str(i).zfill(6)}" for i in range(1, n + 1)]
    revenue = [None if np.random.random() < 0.008 else r for r in revenue]
    unit_prices = [None if np.random.random() < 0.006 else p for p in unit_prices]
    product_names = [None if np.random.random() < 0.004 else p for p in product_names]

    df = pd.DataFrame({
        "order_id": order_ids,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "customer_id": customer_ids,
        "product_category": categories,
        "product_name": product_names,
        "quantity": quantity,
        "unit_price": unit_prices,
        "revenue": revenue,
        "region": regions,
        "salesperson": salesperson,
    })

    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df)} rows -> {OUTPUT_FILE}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nSample:\n{df.head(3).to_string(index=False)}")
