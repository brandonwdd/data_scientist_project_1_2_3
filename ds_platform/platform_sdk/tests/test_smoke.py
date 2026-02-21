"""Minimal smoke tests for Platform SDK"""

import pytest

def test_promotion_gate_evaluate_min():
    """PromotionGate: *_min threshold passes when value >= threshold."""
    from platform_sdk.training.promotion_gate import PromotionGate

    gate = PromotionGate(config={"auc_min": 0.70})
    result = gate.evaluate({"auc": 0.80})
    assert result["passed"] is True
    assert any(d["metric"] == "auc" and d["passed"] for d in result["details"])


def test_promotion_gate_evaluate_min_fail():
    """PromotionGate: *_min threshold fails when value < threshold."""
    from platform_sdk.training.promotion_gate import PromotionGate

    gate = PromotionGate(config={"auc_min": 0.70})
    result = gate.evaluate({"auc": 0.60})
    assert result["passed"] is False


def test_create_app():
    """App factory returns a FastAPI app."""
    from platform_sdk.serving.app_factory import create_app

    app = create_app(title="CI Test", domain="test", enable_metrics=False)
    assert app is not None
    assert app.title == "CI Test"
