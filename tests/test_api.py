import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API pointed at a tiny fake outputs folder, so the test doesn't need the real data."""
    pd.DataFrame({"category": ["toys", "pet_shop"], "elasticity": [-1.6, -0.2],
                  "se": [0.3, 0.8]}).to_csv(tmp_path / "elasticities.csv", index=False)
    pd.DataFrame({"category": ["toys"] * 2, "date": ["2018-08-23", "2018-08-24"],
                  "forecast": [6.1, 5.8], "lower_80": [3, 3], "upper_80": [9, 9]}).to_csv(
        tmp_path / "forecast_next_7_days.csv", index=False)
    with open(tmp_path / "elasticity_summary.json", "w") as f:
        json.dump({"prior": {"mean": -1.3, "mean_se": 0.3, "tau": 0.8}}, f)

    monkeypatch.setenv("PRICING_OUTPUTS_DIR", str(tmp_path))
    from api import app as api_module
    api_module.artifacts.cache_clear()
    yield TestClient(api_module.app)
    api_module.artifacts.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recommendation_within_guardrail(client):
    r = client.post("/price/recommend", json={"category": "toys", "current_price": 100})
    assert r.status_code == 200
    body = r.json()
    assert 90 <= body["recommended_price"] <= 110
    assert body["used_fallback"] is False


def test_uncertain_category_keeps_current_price(client):
    body = client.post("/price/recommend", json={"category": "pet_shop", "current_price": 50}).json()
    assert body["action"].startswith("test first")
    assert body["recommended_price"] == 50
    assert body["candidate_price"] != 50


def test_unknown_category_uses_fallback(client):
    r = client.post("/price/recommend", json={"category": "spaceships", "current_price": 50})
    assert r.status_code == 200
    assert r.json()["used_fallback"] is True


def test_bad_input_is_rejected(client):
    assert client.post("/price/recommend", json={"category": "toys", "current_price": -5}).status_code == 422


def test_forecast(client):
    assert len(client.get("/forecast/toys").json()) == 2
    assert client.get("/forecast/nothing").status_code == 404
