"""
ETL Pipeline: Extract → Transform → Load

- Extract: reads the raw CSV produced by generate_sales_data.py
- Transform:
    • Clean missing values (median impute for price/revenue, mode for product_name)
    • Calculate revenue where missing (quantity * median unit_price per category)
    • Customer Lifetime Value (CLV)
    • Monthly cohort assignment & retention
    • Product performance metrics
    • Regional breakdowns
- Load: write transformed data & summary tables into a SQLite database
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "sales_raw.csv"
DB_PATH = DATA_DIR / "sales_analytics.db"


# ─────────────────────────────────────────────
#  1. EXTRACT
# ─────────────────────────────────────────────
def extract(csv_path: Union[str, Path] = RAW_CSV) -> pd.DataFrame:
    """Read raw CSV and return a DataFrame."""
    print(f"[EXTRACT] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[EXTRACT] {len(df)} rows loaded, columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
#  2. TRANSFORM
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and derive revenue when missing."""
    df = df.copy()

    # --- unit_price imputation (median per category) ---
    median_prices = df.groupby("product_category")["unit_price"].transform("median")
    df["unit_price"] = df["unit_price"].fillna(median_prices)

    # If still any NaNs (category unknown), fill with overall median
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

    # --- product_name imputation (mode per category) ---
    for cat in df["product_category"].dropna().unique():
        mask = df["product_category"] == cat
        mode_name = df.loc[mask, "product_name"].mode()
        if not mode_name.empty:
            df.loc[mask, "product_name"] = df.loc[mask, "product_name"].fillna(mode_name.iloc[0])

    # --- revenue imputation (quantity * unit_price) ---
    df["revenue"] = df["revenue"].fillna(df["quantity"] * df["unit_price"])

    # Round for cleanliness
    df["unit_price"] = df["unit_price"].round(2)
    df["revenue"] = df["revenue"].round(2)

    # Date conversion
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Drop any remaining completely empty rows
    df.dropna(subset=["order_id", "date"], inplace=True)
    # Convert month back to string for SQLite
    df["month"] = df["month"].astype(str)

    # Also ensure order_id is string and date will be converted when loading
    return df


def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Customer Lifetime Value = sum of revenue per customer."""
    clv = (
        df.groupby("customer_id")
        .agg(
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            avg_order_value=("revenue", "mean"),
        )
        .reset_index()
        .rename(columns={"total_revenue": "customer_lifetime_value"})
    )
    # Convert datetime columns to strings for SQLite
    clv["first_purchase"] = clv["first_purchase"].astype(str)
    clv["last_purchase"] = clv["last_purchase"].astype(str)
    clv["customer_tenure_days"] = (
        (pd.to_datetime(clv["last_purchase"]) - pd.to_datetime(clv["first_purchase"])).dt.days
    )
    return clv


def monthly_cohort_analysis(df: pd.DataFrame):
    """
    Assign cohort_month (first purchase month) to each customer,
    then compute activity retention matrix.
    """
    # First purchase month per customer
    first = df.groupby("customer_id")["month"].min().reset_index()
    first.rename(columns={"month": "cohort_month"}, inplace=True)

    df_cohort = df.merge(first, on="customer_id", how="left")
    # Convert all Period types to strings for SQLite
    df_cohort["cohort_month"] = df_cohort["cohort_month"].astype(str)
    df_cohort["activity_month"] = df_cohort["month"].astype(str)
    df_cohort = df_cohort.drop(columns=["month"])

    # Number of customers active per cohort per month
    cohort_sizes = (
        df_cohort.groupby("cohort_month")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "cohort_size"})
    )

    retention = (
        df_cohort.groupby(["cohort_month", "activity_month"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    retention_raw = (
        df_cohort.groupby(["cohort_month", "activity_month"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    # Ensure all month columns are strings for SQLite
    retention_raw.columns = [str(c).replace(" ", "_") for c in retention_raw.columns]
    retention_raw.index.name = "cohort_month"

    cohort_sizes_map = dict(zip(cohort_sizes["cohort_month"], cohort_sizes["cohort_size"]))
    retention = retention_raw.copy()
    retention["cohort_size"] = [cohort_sizes_map.get(idx, 0) for idx in retention.index]

    # Convert to rates
    retention_rate = retention_raw.div(
        pd.Series([cohort_sizes_map.get(idx, 1) for idx in retention_raw.index],
                  index=retention_raw.index),
        axis=0
    )
    retention_rate = retention_rate.round(4)

    # Flatten for SQLite storage
    retention_flat = retention.reset_index()
    retention_flat.columns = [str(c).replace(" ", "_") for c in retention_flat.columns]
    
    retention_rate_flat = retention_rate.reset_index()
    retention_rate_flat.columns = [str(c).replace(" ", "_") for c in retention_rate_flat.columns]

    return df_cohort, retention_flat, retention_rate_flat


def product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per product and per category."""
    # by product
    prod = (
        df.groupby(["product_name", "product_category"])
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
            avg_unit_price=("unit_price", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    # by category
    cat = (
        df.groupby("product_category")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
            unique_products=("product_name", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    return prod, cat


def regional_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order stats per region."""
    reg = (
        df.groupby("region")
        .agg(
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
            avg_order_value=("revenue", "mean"),
            unique_customers=("customer_id", "nunique"),
            total_units=("quantity", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    return reg


def monthly_revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue by month with MoM growth."""
    monthly = (
        df.groupby("year_month")
        .agg(total_revenue=("revenue", "sum"),
             order_count=("order_id", "nunique"),
             unique_customers=("customer_id", "nunique"))
        .reset_index()
        .sort_values("year_month")
    )
    monthly["mom_growth_pct"] = monthly["total_revenue"].pct_change() * 100
    return monthly


def salesperson_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Leaderboard."""
    sp = (
        df.groupby("salesperson")
        .agg(
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
            avg_order_value=("revenue", "mean"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    sp["rank"] = range(1, len(sp) + 1)
    return sp


def transform(df: pd.DataFrame):
    """Run all transformations and return a dict of DataFrames."""
    print("[TRANSFORM] Cleaning data ...")
    clean = clean_data(df)

    print("[TRANSFORM] Computing CLV ...")
    clv = compute_clv(clean)

    print("[TRANSFORM] Building monthly cohort analysis ...")
    df_cohort, retention, retention_rate = monthly_cohort_analysis(clean)

    print("[TRANSFORM] Computing product performance ...")
    prod_perf, cat_perf = product_performance(clean)

    print("[TRANSFORM] Regional breakdown ...")
    regional = regional_breakdown(clean)

    print("[TRANSFORM] Monthly revenue summary ...")
    monthly_rev = monthly_revenue_summary(clean)

    print("[TRANSFORM] Salesperson leaderboard ...")
    leadership = salesperson_performance(clean)

    return {
        "orders_clean": clean,
        "clv": clv,
        "df_cohort": df_cohort,
        "retention": retention,
        "retention_rate": retention_rate,
        "product_performance": prod_perf,
        "category_performance": cat_perf,
        "regional_breakdown": regional,
        "monthly_revenue": monthly_rev,
        "salesperson_perf": leadership,
    }


# ─────────────────────────────────────────────
#  3. LOAD
# ─────────────────────────────────────────────

def load(tables: Dict, db_path: Union[str, Path] = DB_PATH):
    """Write each DataFrame to a named SQLite table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] Writing tables to {db_path} ...")
    with sqlite3.connect(str(db_path)) as conn:
        for name, df in tables.items():
            # Flatten any remaining index
            df = df.reset_index(drop=True) if df.index.name or not df.index.equals(pd.RangeIndex(len(df))) else df
            # Convert datetime columns to strings for SQLite
            for col in df.select_dtypes(include=["datetime"]).columns:
                df[col] = df[col].astype(str)
            for col in df.select_dtypes(include=["float64", "int64"]):
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
            
            df.to_sql(name, conn, if_exists="replace", index=False)
            print(f"  ✓ {name}: {len(df)} rows")

    print("[LOAD] Done.")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(csv_path: Union[str, Path] = RAW_CSV, db_path: Union[str, Path] = DB_PATH):
    """End-to-end ETL pipeline."""
    raw = extract(csv_path)
    tables = transform(raw)
    load(tables, db_path)
    print("\n✅ ETL pipeline completed successfully.")
    return tables


if __name__ == "__main__":
    run_pipeline()
