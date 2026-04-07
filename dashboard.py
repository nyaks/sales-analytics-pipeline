"""
Streamlit Sales Analytics Dashboard

Connects to the SQLite database produced by etl_pipeline.py and provides:
  - Revenue trends over time
  - Top products and categories
  - Regional performance comparison
  - Customer cohort retention analysis
  - Salesperson performance leaderboard
  - Month-over-month growth rates
  - 3-month revenue forecast (exponential smoothing)
"""

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sales Analytics Dashboard")
st.markdown("Comprehensive analysis of sales data generated from synthetic pipeline.")

# ─────────────────────────────────────────────
#  Data Loading Helpers
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "sales_analytics.db"


@st.cache_data(ttl=3600)
def load_table(table_name: str) -> pd.DataFrame:
    """Load a table from the SQLite database."""
    if not DB_PATH.exists():
        st.error(
            f"Database not found at {DB_PATH}. "
            "Run `python etl_pipeline.py` first."
        )
        st.stop()
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df


orders = load_table("orders_clean")
monthly_rev = load_table("monthly_revenue")
regional = load_table("regional_breakdown")
product_perf = load_table("product_performance")
category_perf = load_table("category_performance")
salesperson_perf = load_table("salesperson_perf")
clv = load_table("clv")
retention_rate = load_table("retention_rate")

# Ensure date sorting
monthly_rev = monthly_rev.sort_values("year_month").reset_index(drop=True)

# ─────────────────────────────────────────────
#  KPI Cards Row
# ─────────────────────────────────────────────
total_revenue = orders["revenue"].sum()
total_orders = orders["order_id"].nunique()
total_customers = orders["customer_id"].nunique()
avg_order_value = orders["revenue"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Customers", f"{total_customers:,}")
col4.metric("Avg Order Value", f"${avg_order_value:.2f}")

# ─────────────────────────────────────────────
#  Forecasting Helper
# ─────────────────────────────────────────────
def forecast_linear_regression(series: pd.Series, n_periods: int = 3):
    """Forecast next n_periods using simple linear regression on index."""
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(x, y)
    future_x = np.arange(len(series), len(series) + n_periods).reshape(-1, 1)
    future_y = model.predict(future_x)
    future_dates = _generate_future_dates(series.index[-1], n_periods)
    return future_dates, future_y


def _generate_future_dates(last_ym: str, n: int) -> list[str]:
    """Generate next N YYYY-MM strings after last_ym."""
    year, month = map(int, last_ym.split("-"))
    dates = []
    for _ in range(n):
        month += 1
        if month > 12:
            month = 1
            year += 1
        dates.append(f"{year}-{month:02d}")
    return dates


def forecast_exponential_smoothing(series: pd.Series, alpha: float = 0.3, n_periods: int = 3):
    """Simple exponential smoothing forecast."""
    level = series.iloc[0]
    smoothed = [level]
    for val in series.iloc[1:]:
        level = alpha * val + (1 - alpha) * level
        smoothed.append(level)
    # Final level is the forecast for all future periods
    future_dates = _generate_future_dates(series.index[-1], n_periods)
    return future_dates, np.full(n_periods, smoothed[-1])


# ─────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────
st.sidebar.header("📑 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Revenue & Forecast", "Products", "Regional",
     "Customer Cohorts", "Salespeople"],
)

# ─────────────────────────────────────────────
#  PAGE: Overview
# ─────────────────────────────────────────────
if page == "Overview":
    st.header("Overview")
    c1, c2 = st.columns(2)

    # Top categories
    fig_cat = px.bar(
        category_perf.head(8),
        x="product_category",
        y="total_revenue",
        title="Revenue by Product Category",
        color="total_revenue",
        color_continuous_scale="Blues",
    )
    fig_cat.update_layout(xaxis_title=None)
    c1.plotly_chart(fig_cat, use_container_width=True)

    # Regional split
    fig_reg = px.pie(
        regional,
        values="total_revenue",
        names="region",
        title="Revenue Share by Region",
    )
    c2.plotly_chart(fig_reg, use_container_width=True)

    # Recent monthly trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_rev["year_month"],
        y=monthly_rev["total_revenue"],
        mode="lines+markers",
        name="Revenue",
        line=dict(color="#1f77b4", width=2),
    ))
    fig_trend.update_layout(
        title="Monthly Revenue Trend",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: Revenue & Forecast
# ─────────────────────────────────────────────
elif page == "Revenue & Forecast":
    st.header("Revenue Trends & Forecast")

    method = st.selectbox("Forecasting Method",
                          ["Linear Regression", "Exponential Smoothing"])

    # Build historical series
    hist_series = monthly_rev.set_index("year_month")["total_revenue"]

    if method == "Linear Regression":
        fut_dates, fut_vals = forecast_linear_regression(hist_series, 3)
    else:
        alpha = st.slider("Smoothing Alpha", 0.1, 0.9, 0.3, step=0.05)
        fut_dates, fut_vals = forecast_exponential_smoothing(hist_series, alpha, 3)

    # Combined chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_series.index.tolist(),
        y=hist_series.values,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates,
        y=fut_vals,
        mode="lines+markers",
        name="Forecast (3 months)",
        line=dict(color="#ff7f0e", dash="dash"),
    ))
    fig.update_layout(
        title=f"Revenue Forecast — {method}",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.subheader("Forecast Values (Next 3 Months)")
    forecast_df = pd.DataFrame({"Month": fut_dates, "Predicted Revenue": fut_vals})
    forecast_df["Predicted Revenue"] = forecast_df["Predicted Revenue"].map(
        lambda x: f"${x:,.0f}"
    )
    st.table(forecast_df)

    # MoM growth
    st.subheader("Month-over-Month Growth Rates")
    growth = monthly_rev[["year_month", "total_revenue", "mom_growth_pct"]].copy()
    growth["mom_growth_pct_label"] = growth["mom_growth_pct"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )
    st.dataframe(growth, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: Products
# ─────────────────────────────────────────────
elif page == "Products":
    st.header("Product & Category Performance")

    tab1, tab2 = st.tabs(["Top Products", "Category Summary"])

    with tab1:
        top_n = st.slider("Show top N products", 5, 30, 15)
        top_products = product_perf.head(top_n)
        fig = px.bar(
            top_products,
            y="product_name",
            x="total_revenue",
            orientation="h",
            title=f"Top {top_n} Products by Revenue",
            color="total_revenue",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis_categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_products, hide_index=True, use_container_width=True)

    with tab2:
        fig_cat = px.bar(
            category_perf,
            x="product_category",
            y="total_revenue",
            title="Revenue by Category",
            color="total_revenue",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.dataframe(category_perf, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: Regional
# ─────────────────────────────────────────────
elif page == "Regional":
    st.header("Regional Performance")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            regional,
            x="region",
            y="total_revenue",
            title="Total Revenue by Region",
            color="region",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            regional,
            x="region",
            y="avg_order_value",
            title="Average Order Value by Region",
            color="region",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown sub-tables
    st.subheader("Regional Details")
    st.dataframe(regional, hide_index=True, use_container_width=True)

    # Orders per region over time
    orders["date"] = pd.to_datetime(orders["date"])
    reg_monthly = (
        orders.groupby(["region", orders["date"].dt.to_period("M").astype(str)])
        .agg(total_revenue=("revenue", "sum"))
        .reset_index()
        .rename(columns={"date": "year_month"})
    )
    fig_ts = px.line(
        reg_monthly,
        x="year_month",
        y="total_revenue",
        color="region",
        title="Monthly Revenue by Region",
        markers=True,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: Customer Cohorts
# ─────────────────────────────────────────────
elif page == "Customer Cohorts":
    st.header("Customer Cohort Retention Analysis")

    # Cohort heatmap
    # retention_rate table has cohort_month as index and activity_months as columns
    cohorts = retention_rate.copy()
    cohort_col = [c for c in cohorts.columns if c not in
                  ("cohort_month", "cohort_size", "index")]

    if cohort_col:
        # Pivot to long form for heatmap
        heatmap_data = cohorts.melt(
            id_vars=["cohort_month", "cohort_size"],
            value_vars=cohort_col,
            var_name="activity_month",
            value_name="retention_rate",
        )

        # Calculate cohort_age_index
        cohort_map = {
            cm: i for i, cm in enumerate(sorted(cohorts["cohort_month"].unique()))
        }
        activity_map = {
            am: i for i, am in enumerate(sorted(cohort_col))
        }

        heatmap_data["cohort_idx"] = heatmap_data["cohort_month"].map(cohort_map)
        heatmap_data["activity_idx"] = heatmap_data["activity_month"].map(activity_map)
        heatmap_data["cohort_age"] = heatmap_data["activity_idx"] - heatmap_data["cohort_idx"]

        fig = px.density_heatmap(
            heatmap_data,
            x="cohort_age",
            y="cohort_month",
            z="retention_rate",
            title="Cohort Retention Heatmap (Rate)",
            color_continuous_scale="YlGnBu",
            labels={"cohort_age": "Months since first purchase", "cohort_month": "Cohort"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough cohort data for heatmap yet.")

    # CLV distribution
    st.subheader("Customer Lifetime Value Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            clv, x="customer_lifetime_value", nbins=40,
            title="CLV Distribution",
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.dataframe(
            clv.sort_values("customer_lifetime_value", ascending=False).head(20),
            hide_index=True,
            use_container_width=True,
        )

    # CLV vs tenure
    fig_scatter = px.scatter(
        clv,
        x="customer_tenure_days",
        y="customer_lifetime_value",
        title="CLV vs Customer Tenure",
        opacity=0.6,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: Salespeople
# ─────────────────────────────────────────────
elif page == "Salespeople":
    st.header("Salesperson Performance Leaderboard")

    # Leaderboard table with rank
    sp_display = salesperson_perf[
        ["rank", "salesperson", "total_revenue", "order_count",
         "avg_order_value", "unique_customers"]
    ].copy()
    sp_display["total_revenue"] = sp_display["total_revenue"].map(
        lambda x: f"${x:,.0f}"
    )
    sp_display["avg_order_value"] = sp_display["avg_order_value"].map(
        lambda x: f"${x:.2f}"
    )
    st.dataframe(sp_display, hide_index=True, use_container_width=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            salesperson_perf,
            y="salesperson",
            x="total_revenue",
            orientation="h",
            title="Revenue by Salesperson",
            color="total_revenue",
            color_continuous_scale="Sunset",
        )
        fig.update_layout(yaxis_categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(
            salesperson_perf,
            x="salesperson",
            y="order_count",
            title="Order Count by Salesperson",
            color="order_count",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, use_container_width=True)
