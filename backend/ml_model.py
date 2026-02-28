"""
PriceIQ — ML Model Pipeline
Loads the Amazon India Products dataset (Pricing_dataset.csv), cleans it,
engineers features, trains a Random Forest Regressor to predict demand
(rating_count) from price and product attributes, and exposes prediction
+ insight functions for the API layer.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ── Global state ──────────────────────────────────────────────────────────────
MODEL = None
FEATURE_COLS = None
DATA = None          # cleaned dataframe kept for analytics helpers
MRP_RANGE = {}       # per-category min/max price for sliders
CATEGORIES = []


def _data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "Pricing_dataset.csv")


# ── Helpers to parse Indian ₹ strings ─────────────────────────────────────────
def _parse_price(val):
    """Convert '₹1,099' or 'â‚¹1,099' to float."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    for ch in ["₹", "\u20b9", "â\u201a¹", ",", " "]:
        s = s.replace(ch, "")
    # remove any remaining non-numeric except dot
    s = "".join(c for c in s if c.isdigit() or c == ".")
    try:
        return float(s) if s else np.nan
    except ValueError:
        return np.nan


def _parse_pct(val):
    """Convert '64%' to 0.64."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("%", "").strip()
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def _parse_count(val):
    """Convert '24,269' to 24269.0."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# ── 1. Load & Clean ──────────────────────────────────────────────────────────
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(_data_path())

    # Parse numeric columns from strings
    df["actual_price"] = df["actual_price"].apply(_parse_price)
    df["discounted_price"] = df["discounted_price"].apply(_parse_price)
    df["discount_percentage"] = df["discount_percentage"].apply(_parse_pct)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = df["rating_count"].apply(_parse_count)

    # Drop rows with missing critical values
    df.dropna(subset=["actual_price", "discounted_price", "rating_count", "rating"], inplace=True)

    # Remove extreme outliers (price <= 0)
    df = df[(df["actual_price"] > 0) & (df["discounted_price"] > 0)].copy()

    # Extract top-level category
    df["top_category"] = df["category"].str.split("|").str[0].str.strip()

    # Compute derived features
    df["discount_amount"] = df["actual_price"] - df["discounted_price"]
    df["price_ratio"] = df["discounted_price"] / df["actual_price"]

    # Only keep categories with enough samples
    cat_counts = df["top_category"].value_counts()
    valid_cats = cat_counts[cat_counts >= 5].index
    df = df[df["top_category"].isin(valid_cats)].copy()

    df.reset_index(drop=True, inplace=True)
    return df


# ── 2. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    global MRP_RANGE, CATEGORIES

    CATEGORIES = sorted(df["top_category"].unique().tolist())

    # Store per-category price ranges for frontend sliders
    for cat in CATEGORIES:
        subset = df[df["top_category"] == cat]
        MRP_RANGE[cat] = {
            "min": float(subset["discounted_price"].min()),
            "max": float(subset["actual_price"].max()),
            "mean": float(subset["discounted_price"].mean()),
        }

    # One-hot encode top_category
    df = pd.get_dummies(df, columns=["top_category"], drop_first=False)

    # Numeric features
    numeric_cols = [
        "actual_price",
        "discounted_price",
        "discount_percentage",
        "rating",
        "discount_amount",
        "price_ratio",
    ]

    cat_onehot = [c for c in df.columns if c.startswith("top_category_")]
    keep_cols = numeric_cols + cat_onehot

    # Drop text / non-feature columns
    drop_cols = [
        "product_id", "product_name", "category", "about_product",
        "user_id", "user_name", "review_id", "review_title",
        "review_content", "img_link", "product_link",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    return df, keep_cols


# ── 3. Train ─────────────────────────────────────────────────────────────────
def train_model():
    global MODEL, FEATURE_COLS, DATA

    raw = load_and_clean()
    DATA = raw.copy()

    df, keep_cols = engineer_features(raw)

    target = "rating_count"
    FEATURE_COLS = [c for c in keep_cols if c in df.columns]

    X = df[FEATURE_COLS].fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    MODEL = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    MODEL.fit(X_train, y_train)

    preds = MODEL.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\n{'='*50}")
    print(f"  PriceIQ Model Trained")
    print(f"  Dataset : Amazon India Products — {len(df)} records")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  R² Score : {r2:.4f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"{'='*50}\n")

    model_dir = os.path.dirname(__file__)
    joblib.dump(MODEL, os.path.join(model_dir, "model.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(model_dir, "feature_cols.pkl"))


# ── 4. Prediction helpers ────────────────────────────────────────────────────
def _build_row(
    discounted_price: float,
    actual_price: float,
    discount_pct: float,
    rating: float,
    category: str,
) -> pd.DataFrame:
    """Build a single-row DataFrame aligned with FEATURE_COLS."""
    row = {col: 0 for col in FEATURE_COLS}
    row["actual_price"] = actual_price
    row["discounted_price"] = discounted_price
    row["discount_percentage"] = discount_pct
    row["rating"] = rating
    row["discount_amount"] = actual_price - discounted_price
    row["price_ratio"] = discounted_price / actual_price if actual_price > 0 else 1.0

    cat_col = f"top_category_{category}"
    if cat_col in row:
        row[cat_col] = 1

    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict_demand(
    discounted_price: float,
    actual_price: float = None,
    category: str = "Electronics",
    rating: float = 4.0,
    discount_pct: float = None,
) -> float:
    """Predict demand (rating_count proxy) for given price and attributes."""
    if actual_price is None:
        actual_price = discounted_price * 1.5
    if discount_pct is None:
        discount_pct = 1.0 - (discounted_price / actual_price) if actual_price > 0 else 0.3

    row = _build_row(discounted_price, actual_price, discount_pct, rating, category)
    pred = float(MODEL.predict(row)[0])
    return max(pred, 0)


# ── 5. Elasticity ────────────────────────────────────────────────────────────
def get_elasticity(
    category: str = "Electronics",
    base_price: float = None,
    actual_price: float = None,
    rating: float = 4.0,
) -> float:
    if base_price is None:
        base_price = MRP_RANGE.get(category, {}).get("mean", 500)
    if actual_price is None:
        actual_price = base_price * 1.5

    d1 = predict_demand(base_price, actual_price, category, rating)
    d2 = predict_demand(base_price * 1.1, actual_price, category, rating)

    pct_demand = (d2 - d1) / d1 if d1 != 0 else 0
    pct_price = 0.10
    elasticity = pct_demand / pct_price if pct_price != 0 else 0
    return round(elasticity, 4)


# ── 6. Analytics helpers ─────────────────────────────────────────────────────
def get_category_summary() -> list[dict]:
    grouped = DATA.groupby("top_category").agg(
        avg_price=("discounted_price", "mean"),
        avg_mrp=("actual_price", "mean"),
        avg_demand=("rating_count", "mean"),
        avg_rating=("rating", "mean"),
        avg_discount=("discount_percentage", "mean"),
        count=("rating_count", "count"),
        min_price=("discounted_price", "min"),
        max_price=("actual_price", "max"),
    ).reset_index()

    results = []
    for _, r in grouped.iterrows():
        results.append({
            "item_type": r["top_category"],
            "avg_mrp": round(r["avg_mrp"], 2),
            "avg_price": round(r["avg_price"], 2),
            "avg_sales": round(r["avg_demand"], 2),
            "avg_rating": round(r["avg_rating"], 2),
            "avg_discount": round(r["avg_discount"], 4),
            "count": int(r["count"]),
            "min_mrp": round(r["min_price"], 2),
            "max_mrp": round(r["max_price"], 2),
        })
    return results


def get_outlet_comparison() -> list[dict]:
    """Comparison by sub-category (2nd level of category hierarchy)."""
    df = DATA.copy()
    df["sub_category"] = df["category"].str.split("|").str[1].str.strip()
    sub_cats = df["sub_category"].value_counts().head(6).index
    subset = df[df["sub_category"].isin(sub_cats)]

    grouped = subset.groupby("sub_category").agg(
        avg_sales=("rating_count", "mean"),
        avg_mrp=("actual_price", "mean"),
        avg_discount=("discount_percentage", "mean"),
    ).reset_index()

    return [
        {
            "outlet_type": r["sub_category"],
            "avg_sales": round(r["avg_sales"], 2),
            "avg_mrp": round(r["avg_mrp"], 2),
            "avg_visibility": round(r["avg_discount"], 4),
        }
        for _, r in grouped.iterrows()
    ]


def get_price_segments() -> list[dict]:
    prices = DATA["discounted_price"]
    bins = [0,
            prices.quantile(0.2),
            prices.quantile(0.4),
            prices.quantile(0.6),
            prices.quantile(0.8),
            prices.max() + 1]
    # Remove duplicate bin edges
    bins = sorted(set(bins))
    n_labels = len(bins) - 1
    label_pool = ["Very Low", "Low", "Mid", "High", "Premium"]
    labels = label_pool[:n_labels]

    seg_col = pd.cut(DATA["discounted_price"], bins=bins, labels=labels, duplicates="drop")
    df_tmp = DATA.copy()
    df_tmp["PriceSegment"] = seg_col
    seg = df_tmp.groupby("PriceSegment", observed=False)["rating_count"].mean().reset_index()
    return [
        {"segment": str(r["PriceSegment"]), "avg_sales": round(r["rating_count"], 2)}
        for _, r in seg.iterrows()
    ]


def get_top_products(n: int = 10) -> list[dict]:
    top = DATA.nlargest(n, "rating_count")[
        ["product_id", "product_name", "discounted_price", "actual_price",
         "rating", "rating_count", "top_category"]
    ]
    return [
        {
            "id": r["product_id"],
            "name": (str(r["product_name"])[:60] + "...") if len(str(r["product_name"])) > 60 else str(r["product_name"]),
            "mrp": round(r["actual_price"], 2),
            "price": round(r["discounted_price"], 2),
            "rating": round(r["rating"], 1),
            "sales": round(r["rating_count"], 0),
            "item_type": r["top_category"],
        }
        for _, r in top.iterrows()
    ]


def get_mrp_range(category: str = None) -> dict:
    if category and category in MRP_RANGE:
        return MRP_RANGE[category]
    return {
        "min": float(DATA["discounted_price"].min()),
        "max": float(DATA["actual_price"].max()),
        "mean": float(DATA["discounted_price"].mean()),
    }
