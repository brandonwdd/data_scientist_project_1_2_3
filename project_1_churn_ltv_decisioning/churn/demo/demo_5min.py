"""5-min churn demo workflow."""

import os
import sys
import time
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def main():
    """Run 5-minute demo"""
    
    print("=" * 60)
    print("Churn + LTV Decisioning Service - 5-Min Demo")
    print("=" * 60)
    
    # Step 1: Check infrastructure
    print("\n[1/5] Checking infrastructure...")
    check_infrastructure()
    
    # Step 2: Train model (simulated)
    print("\n[2/5] Training model...")
    print("  → Running: python churn/training/train_churn.py")
    print("  → Model training would happen here")
    print("  → Check MLflow UI: http://localhost:5000")
    
    # Step 3: Check MLflow
    print("\n[3/5] Checking MLflow...")
    check_mlflow()
    
    # Step 4: Start service
    print("\n[4/5] Starting service...")
    print("  → Run: uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000")
    print("  → Waiting for service to be ready...")
    time.sleep(2)
    
    # Step 5: Test API
    print("\n[5/5] Testing API...")
    test_api()
    
    # Step 6: Check Prometheus
    print("\n[6/6] Checking Prometheus metrics...")
    check_prometheus()
    
    print("\n" + "=" * 60)
    print("Demo completed! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. View MLflow UI: http://localhost:5000")
    print("  2. View Prometheus: http://localhost:9090")
    print("  3. Test API: curl http://localhost:8000/score -d '{\"user_id\": \"123\"}'")


def check_infrastructure():
    """Check if infrastructure is running"""
    try:
        # Check MLflow
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            print("  ✓ MLflow is running")
        else:
            print("  ✗ MLflow not accessible")
    except:
        print("  ✗ MLflow not running (start with: cd infra && docker compose up -d)")
    
    try:
        # Check Prometheus
        response = requests.get("http://localhost:9090/-/healthy", timeout=2)
        if response.status_code == 200:
            print("  ✓ Prometheus is running")
        else:
            print("  ✗ Prometheus not accessible")
    except:
        print("  ✗ Prometheus not running")


def check_mlflow():
    """Check MLflow experiments"""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        experiments = mlflow.search_experiments()
        print(f"  ✓ Found {len(experiments)} experiments in MLflow")
        
        # List recent runs
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id for exp in experiments[:1]], max_results=5)
        if len(runs) > 0:
            print(f"  ✓ Found {len(runs)} recent runs")
        else:
            print("  → No runs yet (train model to create runs)")
    except Exception as e:
        print(f"  ✗ MLflow check failed: {e}")


def test_api():
    """Test scoring API"""
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("  ✓ Service is healthy")
        else:
            print("  ✗ Service not healthy")
            return
        
        # Test score endpoint
        payload = {"user_id": "demo_user_123"}
        response = requests.post(
            "http://localhost:8000/score",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Score API working")
            print(f"    - User: {result.get('user_id')}")
            print(f"    - Churn Prob: {result.get('churn_prob', 0):.4f}")
            print(f"    - LTV 90d: ${result.get('ltv_90d', 0):.2f}")
            print(f"    - Action: {result.get('action')}")
            print(f"    - Reason Codes: {result.get('reason_codes', [])}")
        else:
            print(f"  ✗ Score API failed: {response.status_code}")
            print(f"    Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("  ✗ Service not running")
        print("    Start with: uvicorn churn.serving.app:app")
    except Exception as e:
        print(f"  ✗ API test failed: {e}")


def check_prometheus():
    """Check Prometheus metrics"""
    try:
        response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=2)
        if response.status_code == 200:
            print("  ✓ Prometheus is queryable")
            print("    → View metrics: http://localhost:9090")
        else:
            print("  ✗ Prometheus not queryable")
    except Exception as e:
        print(f"  ✗ Prometheus check failed: {e}")


if __name__ == "__main__":
    main()
