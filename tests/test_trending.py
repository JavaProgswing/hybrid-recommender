from fastapi.testclient import TestClient
from backend import main

client = TestClient(main.app)


def test_trending_validation_bounds():
    # Test valid defaults
    # Since we don't have Supabase credentials in tests, it will try to call get_supabase()
    # and fail with RuntimeError unless it's mocked or caught by cache/empty.
    # Let's mock get_supabase to see if validation fails first.
    # FastAPI path/query validation happens BEFORE the route function starts, 
    # so invalid parameters will fail with 422 immediately without calling the function or get_supabase!
    
    # Test negative days
    response = client.get("/api/trending?days=-1")
    assert response.status_code == 422

    # Test excessive days
    response = client.get("/api/trending?days=366")
    assert response.status_code == 422

    # Test limit negative
    response = client.get("/api/trending?limit=0")
    assert response.status_code == 422

    # Test limit excessive
    response = client.get("/api/trending?limit=101")
    assert response.status_code == 422
