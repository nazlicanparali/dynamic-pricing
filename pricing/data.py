"""Loading and cleaning the Olist tables.

Everything downstream works from one item-level table (one row per item sold)
built here, so the filtering rules only live in one place.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

# orders that never turned into a sale
DROP_STATUSES = {"canceled", "unavailable"}


def _read(raw_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = Path(raw_dir) / FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Olist dataset from Kaggle and put the CSVs "
            "in data/raw/ (see data/README.md)."
        )
    return pd.read_csv(path, **kwargs)


def load_items(raw_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Item-level sales with category, customer location and purchase date."""
    orders = _read(raw_dir, "orders", parse_dates=["order_purchase_timestamp"])
    items = _read(raw_dir, "items")
    products = _read(raw_dir, "products", usecols=["product_id", "product_category_name"])
    customers = _read(raw_dir, "customers")
    cats = _read(raw_dir, "categories")

    orders = orders[~orders["order_status"].isin(DROP_STATUSES)]
    orders = orders[
        (orders["order_purchase_timestamp"] >= start_date)
        & (orders["order_purchase_timestamp"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
    ]

    df = (
        items.merge(orders, on="order_id", how="inner")
        .merge(products, on="product_id", how="left")
        .merge(cats, on="product_category_name", how="left")
        .merge(customers[["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
               on="customer_id", how="left")
    )

    df = df.rename(columns={"product_category_name_english": "category"})
    # a few Portuguese names are missing from the translation file
    df["category"] = df["category"].fillna(df["product_category_name"]).fillna("unknown")
    df["date"] = df["order_purchase_timestamp"].dt.normalize()
    df["zip3"] = df["customer_zip_code_prefix"].astype(str).str.zfill(5).str[:3]

    df = df[df["price"] > 0]
    log.info("loaded %s items from %s orders (%s to %s)",
             f"{len(df):,}", f"{df['order_id'].nunique():,}", start_date, end_date)

    keep = ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value",
            "category", "date", "customer_city", "customer_state", "zip3"]
    return df[keep].reset_index(drop=True)


def top_categories(items: pd.DataFrame, n: int) -> list[str]:
    counts = items.loc[items["category"] != "unknown", "category"].value_counts()
    return counts.head(n).index.tolist()


def daily_category_panel(items: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Complete category x day panel (days with no sales are filled with 0).

    Besides units sold it keeps a few daily series that are used as extra
    (lagged) inputs for the forecasting model.
    """
    sub = items[items["category"].isin(categories)]
    daily = sub.groupby(["category", "date"]).agg(
        units=("order_item_id", "size"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        n_sellers=("seller_id", "nunique"),
    )

    all_days = pd.date_range(items["date"].min(), items["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product([categories, all_days], names=["category", "date"])
    panel = daily.reindex(full_index).reset_index()

    panel["units"] = panel["units"].fillna(0).astype(int)
    panel["n_sellers"] = panel["n_sellers"].fillna(0).astype(int)
    # price is undefined on zero-sales days; carry the last observed value forward
    for col in ["avg_price", "avg_freight"]:
        panel[col] = panel.groupby("category")[col].transform(lambda s: s.ffill().bfill())

    # whole-platform volume (all categories, not just the top ones)
    total = items.groupby("date").size().reindex(all_days, fill_value=0)
    panel["platform_units"] = panel["date"].map(total).astype(int)
    return panel


def weekly_listing_panel(items: pd.DataFrame) -> pd.DataFrame:
    """Units and average price per listing (product x seller) per week.

    The same product can be sold by several sellers at different prices, so a
    "listing" is the right unit for looking at how one seller's price changes
    affect their own sales.
    """
    df = items.copy()
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    weekly = df.groupby(["product_id", "seller_id", "week"]).agg(
        units=("order_item_id", "size"),
        price=("price", "mean"),
        freight=("freight_value", "mean"),
        category=("category", "first"),
    ).reset_index()
    weekly["listing"] = weekly["product_id"] + "_" + weekly["seller_id"]
    weekly["log_units"] = np.log(weekly["units"])
    weekly["log_price"] = np.log(weekly["price"])
    weekly["log_freight"] = np.log1p(weekly["freight"])
    return weekly
