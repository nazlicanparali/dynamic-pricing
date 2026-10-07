"""Small API on top of the pipeline outputs.

    uvicorn api.app:app --reload
    open http://127.0.0.1:8000/docs

It doesn't retrain anything: it reads the elasticity table and the latest
forecast from outputs/ (or from PRICING_OUTPUTS_DIR if that's set).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pricing.config import ROOT, load_config
from pricing.optimization import recommend

app = FastAPI(title="Dynamic pricing API", version="0.1.0")


def outputs_dir() -> Path:
    return Path(os.environ.get("PRICING_OUTPUTS_DIR", ROOT / "outputs"))


@lru_cache
def artifacts() -> dict:
    out = outputs_dir()
    try:
        elasticities = pd.read_csv(out / "elasticities.csv", index_col="category")
        forecast = pd.read_csv(out / "forecast_next_7_days.csv", parse_dates=["date"])
        with open(out / "elasticity_summary.json") as f:
            prior = json.load(f)["prior"]
    except FileNotFoundError as e:
        raise RuntimeError(f"missing pipeline output ({e.filename}); run `python run_pipeline.py` first")
    return {"elasticities": elasticities, "forecast": forecast, "prior": prior}


class PriceRequest(BaseModel):
    category: str = Field(examples=["health_beauty"])
    current_price: float = Field(gt=0, examples=[89.9])
    gross_margin: float | None = Field(default=None, gt=0, lt=1,
                                       description="defaults to the value in config.yaml")
    max_change: float | None = Field(default=None, gt=0, le=0.5,
                                     description="guardrail, defaults to config.yaml")


class PriceResponse(BaseModel):
    category: str
    elasticity: float
    elasticity_se: float
    used_fallback: bool
    current_price: float
    recommended_price: float      # = current price unless the action is "change price"
    candidate_price: float        # what the optimiser would pick (the price to test, if testing)
    recommended_change: float
    expected_profit_lift: float
    prob_profit_gain: float
    expected_units_change: float
    action: str


@app.get("/health")
def health():
    try:
        a = artifacts()
    except RuntimeError as e:
        return {"status": "error", "detail": str(e)}
    return {"status": "ok", "n_categories": len(a["elasticities"]),
            "forecast_until": a["forecast"]["date"].max().date().isoformat()}


@app.get("/categories")
def categories():
    est = artifacts()["elasticities"]
    return [{"category": c, "elasticity": round(r["elasticity"], 3), "se": round(r["se"], 3)}
            for c, r in est.iterrows()]


@app.post("/price/recommend", response_model=PriceResponse)
def price_recommendation(req: PriceRequest):
    a = artifacts()
    cfg = load_config()["optimization"]
    est = a["elasticities"]

    if req.category in est.index:
        e, se, fallback = est.loc[req.category, "elasticity"], est.loc[req.category, "se"], False
    else:
        # unknown or small category: use the overall mean, with the between-category spread
        # added to the uncertainty
        prior = a["prior"]
        e, se, fallback = prior["mean"], float(np.hypot(prior["tau"], prior["mean_se"])), True

    rng = np.random.default_rng(0)  # same request -> same answer
    rec = recommend(float(e), float(se), req.gross_margin or cfg["gross_margin"],
                    req.max_change or cfg["max_price_change"], cfg["grid_step"], cfg["n_draws"], rng)
    candidate = round(req.current_price * (1 + rec["recommended_change"]), 2)
    act = rec["action"] == "change price"
    return PriceResponse(
        category=req.category, elasticity=round(float(e), 4), elasticity_se=round(float(se), 4),
        used_fallback=fallback, current_price=req.current_price,
        recommended_price=candidate if act else req.current_price,
        candidate_price=candidate,
        recommended_change=round(rec["recommended_change"], 4) if act else 0.0,
        expected_profit_lift=round(rec["expected_profit_lift"], 4),
        prob_profit_gain=round(rec["prob_profit_gain"], 4),
        expected_units_change=round(rec["expected_units_change"], 4),
        action=rec["action"],
    )


@app.get("/forecast/{category}")
def forecast(category: str):
    fc = artifacts()["forecast"]
    rows = fc[fc["category"] == category]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"no forecast for category '{category}'")
    return [{"date": r["date"].date().isoformat(), "forecast": r["forecast"],
             "lower_80": r["lower_80"], "upper_80": r["upper_80"]} for _, r in rows.iterrows()]
