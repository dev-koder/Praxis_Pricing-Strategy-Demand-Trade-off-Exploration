"""
PriceIQ — FastAPI Backend
Serves ML predictions, analytics, and GenAI strategy to the React frontend.
Adapted for the Amazon India Products dataset (Pricing_dataset.csv).
"""

from __future__ import annotations

import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ml_model import (
    train_model,
    predict_demand,
    get_elasticity,
    get_category_summary,
    get_outlet_comparison,
    get_price_segments,
    get_top_products,
    get_mrp_range,
    DATA,
    MRP_RANGE,
)
from genai_strategy import generate_strategy

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="PriceIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.on_event("startup")
def startup():
    train_model()


# ── Pydantic models ──────────────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    price: float
    actual_price: float = None          # MRP / list price
    rating: float = 4.0
    item_type: str = "Electronics"      # top-level category
    outlet_type: str = "Online"         # channel label for strategy text
    cost: float = None                  # cost basis for margin calc
    segment: str = "B2C"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "loaded"}


@app.get("/api/categories")
def categories():
    return get_category_summary()


@app.get("/api/outlets")
def outlets():
    return get_outlet_comparison()


@app.get("/api/price-segments")
def price_segments():
    return get_price_segments()


@app.get("/api/top-products")
def top_products():
    return get_top_products()


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    # Derive actual_price (MRP) if not provided
    actual_price = req.actual_price or req.price * 1.5
    discount_pct = 1.0 - (req.price / actual_price) if actual_price > 0 else 0.3

    # Predicted demand
    demand = predict_demand(
        discounted_price=req.price,
        actual_price=actual_price,
        category=req.item_type,
        rating=req.rating,
        discount_pct=discount_pct,
    )

    # Elasticity
    elasticity = get_elasticity(
        category=req.item_type,
        base_price=req.price,
        actual_price=actual_price,
        rating=req.rating,
    )

    # Cost: default to 40% of selling price if not given
    cost = req.cost if req.cost is not None else req.price * 0.40

    # Revenue / Profit / Margin
    revenue = req.price * demand
    profit = (req.price - cost) * demand
    margin = (req.price - cost) / req.price if req.price > 0 else 0.0

    # Optimal price — grid search over price range
    mrp = get_mrp_range(req.item_type)
    lo = max(mrp["min"], cost * 1.05) if cost > 0 else mrp["min"]
    # Cap at actual_price (MRP) — can't sell above MRP
    hi = min(actual_price * 1.2, mrp["max"])
    price_grid = np.linspace(lo, hi, 80)
    best_rev = 0
    optimal_price = req.price
    for p in price_grid:
        d = predict_demand(float(p), actual_price, req.item_type, req.rating)
        r = float(p) * d
        if r > best_rev:
            best_rev = r
            optimal_price = float(p)

    # Competitor avg price = category mean
    competitor_avg_price = mrp["mean"]

    # GenAI strategy
    strategy = generate_strategy(
        price=req.price,
        predicted_demand=demand,
        elasticity=elasticity,
        margin=margin,
        item_type=req.item_type,
        outlet_type=req.outlet_type,
        optimal_price=optimal_price,
        competitor_avg_price=competitor_avg_price,
    )

    return {
        "predicted_demand": round(demand, 2),
        "revenue": round(revenue, 2),
        "profit": round(profit, 2),
        "margin": round(margin, 4),
        "elasticity": elasticity,
        "optimal_price": round(optimal_price, 2),
        "strategy": strategy,
    }


@app.get("/api/demand-curve")
def demand_curve(
    item_type: str = "Electronics",
    rating: float = 4.0,
    cost: float = 0,
    steps: int = 60,
):
    mrp = get_mrp_range(item_type)
    actual_price = mrp["mean"] * 2  # reference MRP
    hi = min(mrp["mean"] * 3, mrp["max"])
    steps = min(max(steps, 10), 100)
    prices = np.linspace(mrp["min"], hi, steps)
    curve = []
    for p in prices:
        d = predict_demand(float(p), actual_price, item_type, rating)
        rev = float(p) * d
        prof = (float(p) - cost) * d
        curve.append({
            "price": round(float(p), 2),
            "demand": round(d, 2),
            "revenue": round(rev, 2),
            "profit": round(prof, 2),
        })
    return curve


@app.get("/api/competitors")
def competitors(item_type: str = "Electronics"):
    """Simulated competitor data derived from dataset stats."""
    summary = {c["item_type"]: c for c in get_category_summary()}
    cat = summary.get(item_type, list(summary.values())[0] if summary else {})
    avg = cat.get("avg_price", 500)

    competitors_data = [
        {"name": "ValueMart",   "price": round(avg * 0.80, 2), "market_share": 22, "strategy": "Value Leader"},
        {"name": "FlipDeals",   "price": round(avg * 0.92, 2), "market_share": 18, "strategy": "Competitive"},
        {"name": "PrimeBuy",    "price": round(avg * 1.08, 2), "market_share": 28, "strategy": "Premium"},
        {"name": "QuickShop",   "price": round(avg * 1.20, 2), "market_share": 12, "strategy": "Niche Premium"},
    ]
    return {
        "item_type": item_type,
        "category_avg_price": round(avg, 2),
        "competitors": competitors_data,
    }
