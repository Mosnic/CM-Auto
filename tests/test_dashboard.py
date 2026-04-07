```python
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Mock config at module level before importing dashboard
mock_config = {
    "dashboard": {"port": 8002, "host": "0.0.0.0"},
    "paths": {"uploads": "/uploads", "data": "/data"},
    "models": {
        "vision": {"model_id": "test", "endpoint": "http://localhost:8000/v1"},
        "embedding": {"model_id": "test", "endpoint": "http://localhost:8001/v1"},
        "coding": {"model_id": "test", "endpoint": "http://localhost:11434"}
    },
    "thresholds": {"similarity_known": 0.25},
    "processing": {"frame_width": 640},
    "startup_commands": {"vision": "test"},
    "logging": {"level": "INFO"}
}

with patch('builtins.open', create=True) as mock_open:
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
    import dashboard

@pytest.fixture
def mock_db():
    with patch('dashboard.CatDatabase') as mock:
        db_instance = Mock()
        mock.return_value = db_instance
        yield db_instance

@pytest.fixture
def mock_identity_engine():
    with patch('dashboard.CatIdentityEngine') as mock:
        engine_instance = Mock()
        mock.return_value = engine_instance
        yield engine_instance

@pytest.fixture
def mock_reflective_agent():
    with patch('dashboard.ReflectiveAgent') as mock:
        agent_instance = Mock()
        mock.return_value = agent_instance
        yield agent_instance

@pytest.fixture
def dashboard_instance(mock_db, mock_identity_engine, mock_reflective_agent):
    return dashboard.Dashboard()

@pytest.fixture
def client(dashboard_instance):
    return TestClient(dashboard_instance.app)

@pytest.fixture
def mock_uploads_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('dashboard.UPLOADS_DIR', Path(tmpdir)):
            yield Path(tmpdir)

class TestDashboard:
    def test_init(self, mock_db, mock_identity_engine, mock_reflective_agent):
        with patch('dashboard.Dashboard._setup_routes'), patch('dashboard.Dashboard._setup_static_files'):
            dash = dashboard.Dashboard()
            assert dash.app is not None
            assert dash.db is not None
            assert dash.identity_engine is not None
            assert dash.reflective_agent is not None

    @patch('dashboard.StaticFiles')
    def test_setup_static_files_success(self, mock_static_files, dashboard_instance, mock_uploads_dir):
        dashboard_instance._setup_static_files()
        # Should not raise exception

    @patch('dashboard.StaticFiles', side_effect=Exception("Mount error"))
    def test_setup_static_files_error(self, mock_static_files, dashboard_instance, mock_uploads_dir):
        with patch.object(dashboard_instance.logger, 'warning') as mock_warning:
            dashboard_instance._setup_static_files()
            mock_warning.assert_called()

    def test_setup_routes(self, dashboard_instance):
        dashboard_instance._setup_routes()
        # Verify routes are set up by checking the app's router
        routes = [route.path for route in dashboard_instance.app.router.routes]
        assert "/" in routes
        assert "/cat/{cat_id}" in routes
        assert "/visits" in routes
        assert "/alerts" in routes

    def test_home_success(self, client, mock_db):
        mock_db.get_all_cats.return_value = [
            {"cat_id": "cat1", "description": "Test Cat", "status": "active", "visit_count": 5}
        ]
        mock_db.get_active_alerts.return_value = [
            {"alert_type": "test", "message": "test alert", "created_at": "2023-01-01"}
        ]
        mock_db.get_visits_by_date_range.return_value = [{"visit_id": "v1"}]
        
        response = client.get("/")
        assert response.status_code == 200
        assert "Cat Monitor Dashboard" in response.text

    def test_home_empty_data(self, client, mock_db):
        mock_db.get_all_cats.return_value = []
        mock_db.get_active_alerts.return_value = []
        mock_db.get_visits_by_date_range.return_value = []
        
        response = client.get("/")
        assert response.status_code == 200
        assert "0" in response.text  # Should show 0 for stats

    def test_home_none_data(self, client, mock_db):
        mock_db.get_all_cats.return_value = None
        mock_db.get_active_alerts.return_value = None
        mock_db.get_visits_by_date_range.return_value = None
        
        response = client.get("/")
        assert response.status_code == 200

    def test_home_database_error(self, client, mock_db):
        mock_db.get_all_cats.side_effect = Exception("DB Error")
        
        response = client.get("/")
        assert response.status_code == 500
        assert "Error" in response.text

    def test_cat_profile_success(self, client, mock_db, mock_reflective_agent):
        mock_db.get_cat_profile.return_value = {
            "cat_id": "cat1", "description": "Test Cat", "status": "active"
        }
        mock_db.get_visits_by_cat.return_value = [
            {"visit_id": "v1", "timestamp": "2023-01-01", "confidence": 0.8}
        ]
        mock_reflective_agent.analyze_cat_history.return_value = "Analysis text"
        
        response = client.get("/cat/cat1")
        assert response.status_code == 200
        assert "Test Cat" in response.text

    def test_cat_profile_not_found(self, client, mock_db):
        mock_db.get_cat_profile.return_value = None
        
        response = client.get("/cat/nonexistent")
        assert response.status_code == 404

    def test_cat_profile_empty_visits(self, client, mock_db, mock_reflective_agent):
        mock_db.get_cat_profile.return_value = {"cat_id": "cat1", "description": "Test Cat"}
        mock_db.get_visits_by_cat.return_value = []
        mock_reflective_agent.analyze_cat_history.return_value = None
        
        response = client.get("/cat/cat1")
        assert response.status_code == 200
        assert "No visits recorded" in response.text

    def test_cat_profile_analysis_error(self, client, mock_db, mock_reflective_agent):
        mock_db.get_cat_profile.return_value = {"cat_id": "cat1", "description": "Test Cat"}
        mock_db.get_visits_by_cat.return_value = []
        mock_reflective_agent.analyze_cat_history.side_effect = Exception("Analysis error")
        
        response = client.get("/cat/cat1")
        assert response.status_code == 200  # Should still render without analysis

    def test_recent_visits_success(self, client, mock_db):
        mock_db.get_visits_by_date_range.return_value = [
            {"visit_id": "v1", "timestamp": "2023-01-01", "cat_id": "cat1", "confidence": 0.8}
        ]
        
        response = client.get("/visits")
        assert response.status_code == 200
        assert "Recent Visits" in response.text

    def test_recent_visits_custom_days(self, client, mock_db):
        mock_db.get_visits_by_date_range.return_value = []
        
        response = client.get("/visits?days=30")
        assert response.status_code == 200

    def test_recent_visits_empty(self, client, mock_db):
        mock_db.get_visits_by_date_range.return_value = []
        
        response = client.get("/visits")
        assert response.status_code == 200
        assert "No visits found" in response.text

    def test_recent_visits_none(self, client, mock_db):
        mock_db.get_visits_by_date_range.return_value = None
        
        response = client.get("/visits")
        assert response.status_code == 200

    def test_recent_visits_error(self, client, mock_db):
        mock_db.get_visits_by_date_range.side_effect = Exception("DB Error")
        
        response = client.get("/visits")
        assert response.status_code == 500

    def test_alerts_page_success(self, client, mock_db):
        mock_db.get_active_alerts.return_value = [
            {"alert_id": "a1", "alert_type": "test", "message": "test alert"}
        ]
        
        response = client.get("/alerts")
        assert response.status_code == 200
        assert "Active Alerts" in response.text

    def test_alerts_page_empty(self, client, mock_db):
        mock_db.get_active_alerts.return_value = []
        
        response = client.get("/alerts")
        assert response.status_code == 200
        assert "No active alerts" in response.text

    def test_alerts_page_error(self, client, mock_db):
        mock_db.get_active_alerts.side_effect = Exception("DB Error")
        
        response = client.get("/alerts")
        assert response.status_code == 500

    def test_correct_identity_success(self, client, mock_db, mock_identity_engine):
        mock_db.get_visit.return_value = {"visit_id": "v1", "cat_id": "old_cat"}
        mock_db.get_cat_profile.return_value = {"cat_id": "new_cat"}
        
        response = client.post("/api/correct-identity", data={
            "visit_id": "v1",
            "correct_cat_id": "new_cat"
        })
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_db.update_visit_cat.assert_called_once_with("v1", "new_cat")

    def test_correct_identity_new_cat(self, client, mock_db, mock_identity_engine):
        mock_db.get_visit.return_value = {"visit_id": "v1"}
        mock_identity_engine.create_new_cat_profile.return_value = {"cat_id": "new_cat_123"}
        
        response = client.post("/api/correct-identity", data={
            "visit_id": "v1", 
            "correct_cat_id": "new"
        })
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_identity_engine.create_new_cat_profile.assert_called_once()

    def test_correct_identity_visit_not_found(self, client, mock_db):
        mock_db.get_visit.return_value = None
        
        response = client.post("/api/correct-identity", data={
            "visit_id": "nonexistent",
            "correct_cat_id": "cat1"
        })
        assert response.status_code == 404
        assert response.json()["success"] is False

    def test_correct_identity_cat_not_found(self, client, mock_db):
        mock_db.get_visit.return_value = {"visit_id": "v1"}
        mock_db.get_cat_profile.return_value = None
        
        response = client.post("/api/correct-identity", data={
            "visit_id": "v1",
            "correct_cat_id": "nonexistent"
        })
        assert response.status_code == 404
        assert response.json()["success"] is False

    def test_correct_identity_error(self, client, mock_db):
        mock_db.get_visit.side_effect = Exception("DB Error")
        
        response = client.post("/api/correct-identity", data={
            "visit_id": "v1",
            "correct_cat_id": "cat1"
        })
        assert response.status_code == 500
        assert response.json()["success"] is False

    def test_merge_cats_success(self, client, mock_db, mock_identity_engine):
        mock_db.get_cat_profile.side_effect = [
            {"cat_id": "cat1"}, {"cat_id": "cat2"}  # Both cats exist
        ]
        mock_db.get_visits_by_cat.return_value = [
            {"visit_id": "v1"}, {"visit_id": "v2"}
        ]
        mock_identity_engine.average_embeddings = Mock()
        
        response = client.post("/api/merge-cats", data={
            "cat_id_1": "cat1",
            "cat_id_2": "cat2"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["visits_reassigned"] == 2
        mock_db.delete_cat_profile.assert_called_once_with("cat2")

    def test_merge_cats_first_not_found(self, client, mock_db):
        mock_db.get_cat_profile.side_effect = [None, {"cat_id": "cat2"}]
        
        response = client.post("/api/merge-cats", data={
            "cat_id_1": "nonexistent",
            "cat_id_2": "cat2"
        })
        assert response.status_code == 404
        assert response.json()["success"] is False

    def test_merge_cats_second_not_found(self, client, mock_db):
        mock_db.get_cat_profile.side_effect = [{"cat_id": "cat1"}, None]
        
        response = client.post("/api/merge-cats", data={
            "cat_id_1": "cat1",
            "cat_id_2": "nonexistent"
        })
        assert response.status_code == 404
        assert response.json()["success"] is False

    def test_merge_cats_no_visits(self, client, mock_db, mock_identity_engine):
        mock_db.get_cat_profile.side_effect = [{"cat_id": "cat1"}, {"cat_id": "cat2"}]
        mock_db.get_visits_by_cat.return_value = None
        
        response = client.post("/api/merge-cats", data={
            "cat_id_1": "cat1",
            "cat_id_2": "cat2"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["visits_reassigned"] == 0

    def test_merge_cats_error(self, client, mock_db):
        mock_db.get_cat_profile.side_effect = Exception("DB Error")
        
        response = client.post("/api/merge-cats", data={
            "cat_id_1": "cat1",
            "cat_id_2": "cat2"
        })
        assert response.status_code == 500
        assert response.json()["success"] is False

    def test_api_stats_success(self, client, mock_db):
        mock_db.get_all_cats.return_value = [
            {"status": "active"}, {"status": "inactive"}, {"status": "active"}
        ]
        mock_db.get_visits_by_date_range.return_value = [{"visit_id": "v1"}]
        mock_db.get_active_alerts.return_value = [{"alert_id": "a1"}]
        
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_cats"] == 3
        assert data["active_cats"] == 2
        assert data["active_alerts"] == 1

    def test_api_stats_empty(self, client, mock_db):
        mock_db.get_all_cats.return_value = []
        mock_db.get_visits_by_date_range.return_value = []
        mock_db.get_active_alerts.return_value = []
        
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_cats"] == 0
        assert data["active_cats"] == 0

    def test_api_stats_none_data(self, client, mock_db):
        mock_db.get_all_cats.return_value = None
        mock_db.get_visits_by_date_range.return_value = None
        mock_db.get_active_alerts.return_value = None
        
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_api_stats_error(self, client, mock_db):
        mock_db.get_all_cats.side_effect = Exception("DB Error")
        
        response = client.get("/api/stats")
        assert response.status_code == 500

    def test_serve_image_success(self, client, mock_uploads_dir):
        image_file = mock_uploads_dir / "test.jpg"
        image_file.write_text("fake image data")
        
        response = client.get("/image/test.jpg")
        assert response.status_code == 200

    def test_serve_image_not_found(self, client, mock_uploads_dir):
        response = client.get("/image/nonexistent.jpg")
        assert response.status_code == 404

    def test_serve_image_path_traversal(self, client, mock_uploads_dir):
        response = client.get("/image/../../../etc/passwd")
        assert response.status_code == 403

    def test_serve_image_error(self, client):
        with patch('dashboard.UPLOADS_DIR.resolve', side_effect=Exception("Path error")):
            response = client.get("/image/test.jpg")
            assert response.status_code == 500

    @patch('dashboard.uvicorn.run')
    def test_run(self, mock_uvicorn_run, dashboard_instance):
        dashboard_instance.run(host="127.0.0.1", port=9000)
        mock_uvicorn_run.assert_called_once_with(dashboard_instance.app, host="127.0.0.1", port=9000)

def test_create_dashboard_app(mock_db, mock_identity_engine, mock_reflective_agent):
    with patch('dashboard.Dashboard._setup_routes'), patch('dashboard.Dashboard._setup_static_files'):
        app = dashboard.create_dashboard_app()
        assert app is not None

def test_format_timestamp_success():
    result = dashboard.format_timestamp("2023-01-01T12:30:00Z")
    assert result == "2023-01-01 12:30"

def test_format_timestamp_no_z():
    result = dashboard.format_timestamp("2023-01-01T12:30:00+00:00")
    assert result == "2023-01-01 12:30"

def test_format_timestamp_invalid():
    result = dashboard.format_timestamp("invalid")
    assert result == "invalid"

def test_format_timestamp_none():
    result = dashboard.format_timestamp(None)
    assert result is None

def test_calculate_visit_stats_success():
    visits = [
        {"confidence": 0.8, "camera": "cam1"},
        {"confidence": 0.9, "camera": "cam2"},
        {"confidence": 0.7, "camera": "cam1"}
    ]
    stats = dashboard.calculate_visit_stats(visits)
    assert stats["total"] == 3
    assert stats["avg_confidence"] == 0.8
    assert set(stats["cameras"]) == {"cam1", "cam2"}

def test_calculate_visit_stats_empty():
    stats = dashboard.calculate_visit_stats([])
    assert stats["total"] == 0
    assert stats["avg_confidence"] == 0
    assert stats["cameras"] == []

def test_calculate_visit_stats_none():
    stats = dashboard.calculate_visit_stats(None)
    assert stats["total"] == 0
    assert stats["avg_confidence"] == 0
    assert stats["cameras"] == []

def test_calculate_visit_stats_missing_fields():
    visits = [
        {"confidence": 0.8},  # Missing camera
        {"camera": "cam1"},   # Missing confidence
        {}                     # Missing both
    ]
    stats = dashboard.calculate_visit_stats(visits)
    assert stats["total"] == 3
    assert "avg_confidence" in stats
    assert "unknown" in stats["cameras"]
```