"""Eval gate: promotion thresholds (faithfulness, relevancy, citation, evidence recall)."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_gate_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load promotion_gate.yaml for RAG."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "promotion_gate.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_gate(metrics: Dict[str, float], config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Evaluate metrics against gate. Returns {passed: bool, details: [...]}.
    """
    config = config or load_gate_config()
    details = []
    passed = True

    ev = config.get("eval", {})
    for k, v in ev.items():
        if not k.endswith("_min"):
            continue
        name = k.replace("_min", "")
        val = metrics.get(name)
        if val is None:
            continue
        ok = val >= v
        details.append({"metric": name, "value": val, "threshold": v, "passed": ok})
        if not ok:
            passed = False

    cit = config.get("citation", {})
    for k, v in cit.items():
        if not k.endswith("_min"):
            continue
        name = k.replace("_min", "")
        val = metrics.get(name)
        if val is None:
            continue
        ok = val >= v
        details.append({"metric": name, "value": val, "threshold": v, "passed": ok})
        if not ok:
            passed = False

    slo = config.get("slo", {})
    for k, v in slo.items():
        if not k.endswith("_max"):
            continue
        name = k.replace("_max", "")
        val = metrics.get(name)
        if val is None:
            continue
        ok = val <= v
        details.append({"metric": name, "value": val, "threshold": v, "passed": ok})
        if not ok:
            passed = False

    guard = config.get("guardrails", {})
    if guard.get("must_return_idk_when_no_evidence") and metrics.get("idk_violation_rate", 0) > 0:
        details.append({"metric": "must_return_idk", "value": metrics.get("idk_violation_rate"), "threshold": 0, "passed": False})
        passed = False
    if "hallucination_flag_rate_max" in guard:
        h = metrics.get("hallucination_flag_rate")
        if h is not None and h > guard["hallucination_flag_rate_max"]:
            details.append({"metric": "hallucination_flag_rate", "value": h, "threshold": guard["hallucination_flag_rate_max"], "passed": False})
            passed = False

    return {"passed": passed, "details": details}


def promote_if_passed(metrics: Dict[str, float], model_uri: str, registered_name: str, config: Optional[Dict] = None) -> bool:
    """Promote if gate passes; write results to registry tags."""
    res = evaluate_gate(metrics, config)
    if not res["passed"]:
        return False
    try:
        import mlflow
        from datetime import datetime
        mlflow.register_model(model_uri, registered_name)
        client = mlflow.tracking.MlflowClient()
        vers = client.get_latest_versions(registered_name, stages=["Production"])
        if vers:
            v = vers[0]
            client.set_model_version_tag(registered_name, v.version, "eval_gate_passed", "true")
            client.set_model_version_tag(registered_name, v.version, "eval_gate_ts", datetime.utcnow().isoformat())
    except Exception:
        pass
    return True
