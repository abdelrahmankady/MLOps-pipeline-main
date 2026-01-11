from fastapi.testclient import TestClient
from src.serve_api import app

client = TestClient(app)

def test_health_check():
    """Test if the API is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "active"}

def test_prediction_endpoint():
    """Test a sample prediction flow."""
    # Mock input data
    payload = {
        "Date": "2023-01-01",
        "BranchID": "B001",
        "InvoiceNumber": "INV-1001",
        "ItemCode": "ITEM-500",
        "QuantitySold": 1.0
    }
    
    # Send POST request
    response = client.post("/predict", json=payload)
    
    # Check response
    assert response.status_code == 200
    json_data = response.json()
    
    # We expect a prediction class (LOW, MEDIUM, or HIGH)
    assert "prediction" in json_data
    assert json_data["prediction"] in ["LOW", "MEDIUM", "HIGH"]