import pytest
import sqlite3
import json
import uuid
from unittest.mock import patch, mock_open
from pathlib import Path
from datetime import datetime, timedelta

import database


@pytest.fixture
def mock_config():
    config_data = {
        "paths": {
            "database": "/data/catmonitor.db"
        }
    }
    with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
        with patch("database.cfg", config_data):
            yield config_data


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    return str(db_path)


@pytest.fixture
def cat_db(temp_db, mock_config):
    return database.CatDatabase(db_path=temp_db)


class TestDictFactory:
    def test_dict_factory_basic(self):
        class MockCursor:
            description = [("id", None), ("name", None)]
        
        cursor = MockCursor()
        row = (1, "test")
        result = database.dict_factory(cursor, row)
        assert result == {"id": 1, "name": "test"}

    def test_dict_factory_empty(self):
        class MockCursor:
            description = []
        
        cursor = MockCursor()
        row = ()
        result = database.dict_factory(cursor, row)
        assert result == {}


class TestSerializeJsonField:
    def test_serialize_dict(self):
        data = {"key": "value"}
        result = database.serialize_json_field(data)
        assert result == '{"key": "value"}'

    def test_serialize_list(self):
        data = ["item1", "item2"]
        result = database.serialize_json_field(data)
        assert result == '["item1", "item2"]'

    def test_serialize_none(self):
        result = database.serialize_json_field(None)
        assert result == "[]"

    def test_serialize_empty_dict(self):
        data = {}
        result = database.serialize_json_field(data)
        assert result == "{}"


class TestDeserializeJsonField:
    def test_deserialize_dict(self):
        json_str = '{"key": "value"}'
        result = database.deserialize_json_field(json_str)
        assert result == {"key": "value"}

    def test_deserialize_list(self):
        json_str = '["item1", "item2"]'
        result = database.deserialize_json_field(json_str)
        assert result == ["item1", "item2"]

    def test_deserialize_empty_string(self):
        result = database.deserialize_json_field("")
        assert result == []

    def test_deserialize_none(self):
        result = database.deserialize_json_field(None)
        assert result == []

    def test_deserialize_invalid_json(self):
        result = database.deserialize_json_field("invalid json")
        assert result == []


class TestCatDatabaseInit:
    def test_init_default_path(self, mock_config):
        with patch.object(database.CatDatabase, '_ensure_directory'):
            with patch.object(database.CatDatabase, '_init_schema'):
                db = database.CatDatabase()
                assert db.db_path == "/data/catmonitor.db"

    def test_init_custom_path(self, mock_config):
        with patch.object(database.CatDatabase, '_ensure_directory'):
            with patch.object(database.CatDatabase, '_init_schema'):
                db = database.CatDatabase(db_path="/custom/path.db")
                assert db.db_path == "/custom/path.db"

    @patch('database.Path')
    def test_ensure_directory_success(self, mock_path, temp_db, mock_config):
        mock_path.return_value.parent.mkdir = lambda parents=True, exist_ok=True: None
        db = database.CatDatabase(db_path=temp_db)
        mock_path.assert_called_with(temp_db)

    @patch('database.Path')
    def test_ensure_directory_failure(self, mock_path, temp_db, mock_config):
        mock_path.return_value.parent.mkdir.side_effect = OSError("Permission denied")
        # Should not raise exception, just log warning
        database.CatDatabase(db_path=temp_db)

    def test_init_schema_failure(self, temp_db, mock_config):
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Connection failed")
            with pytest.raises(sqlite3.Error):
                database.CatDatabase(db_path=temp_db)


class TestCreateCatProfile:
    def test_create_cat_profile_basic(self, cat_db):
        analysis = {
            "description": "Orange tabby",
            "appearance": {
                "coat_color": "orange",
                "eye_color": "green",
                "distinctive_markings": ["white paws"]
            },
            "body_condition": "healthy"
        }
        cat_id = cat_db.create_cat_profile(analysis)
        assert isinstance(cat_id, str)
        
        profile = cat_db.get_cat_profile(cat_id)
        assert profile["description"] == "Orange tabby"
        assert profile["coat_color"] == "orange"
        assert profile["eye_color"] == "green"

    def test_create_cat_profile_with_custom_id(self, cat_db):
        custom_id = str(uuid.uuid4())
        analysis = {"description": "Test cat"}
        cat_id = cat_db.create_cat_profile(analysis, cat_id=custom_id)
        assert cat_id == custom_id

    def test_create_cat_profile_empty_analysis(self, cat_db):
        cat_id = cat_db.create_cat_profile({})
        assert isinstance(cat_id, str)
        profile = cat_db.get_cat_profile(cat_id)
        assert profile is not None

    def test_create_cat_profile_duplicate_id(self, cat_db):
        cat_id = str(uuid.uuid4())
        cat_db.create_cat_profile({}, cat_id=cat_id)
        
        with pytest.raises(RuntimeError):
            cat_db.create_cat_profile({}, cat_id=cat_id)


class TestUpdateCatProfile:
    def test_update_cat_profile_basic(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Original"})
        
        update_analysis = {
            "description": "Updated description",
            "body_condition": "good",
            "appearance": {"coat_color": "black"}
        }
        cat_db.update_cat_profile(cat_id, update_analysis)
        
        profile = cat_db.get_cat_profile(cat_id)
        assert profile["description"] == "Updated description"
        assert profile["visit_count"] == 2

    def test_update_cat_profile_nonexistent(self, cat_db):
        fake_id = str(uuid.uuid4())
        # Should not raise exception
        cat_db.update_cat_profile(fake_id, {"description": "test"})


class TestRecordVisit:
    def test_record_visit_basic(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        analysis = {
            "behavior": "eating",
            "body_condition": "healthy",
            "health_flags": ["none"],
            "camera": "front_door"
        }
        
        visit_id = cat_db.record_visit(
            cat_id=cat_id,
            clip_path="/clips/test.mp4", 
            frame_path="/frames/test.jpg",
            analysis=analysis,
            confidence=0.8,
            distance=0.2
        )
        
        assert isinstance(visit_id, str)
        visits = cat_db.get_cat_visits(cat_id)
        assert len(visits) == 1
        assert visits[0]["behavior"] == "eating"
        assert visits[0]["confidence"] == 0.8

    def test_record_visit_nonexistent_cat(self, cat_db):
        fake_cat_id = str(uuid.uuid4())
        
        visit_id = cat_db.record_visit(
            cat_id=fake_cat_id,
            clip_path="/clips/test.mp4",
            frame_path="/frames/test.jpg", 
            analysis={},
            confidence=0.5,
            distance=0.3
        )
        
        assert isinstance(visit_id, str)
        # Should create placeholder cat profile
        profile = cat_db.get_cat_profile(fake_cat_id)
        assert profile is not None


class TestGetCatProfile:
    def test_get_cat_profile_exists(self, cat_db):
        analysis = {"description": "Test cat"}
        cat_id = cat_db.create_cat_profile(analysis)
        
        profile = cat_db.get_cat_profile(cat_id)
        assert profile is not None
        assert profile["cat_id"] == cat_id
        assert profile["description"] == "Test cat"

    def test_get_cat_profile_not_exists(self, cat_db):
        fake_id = str(uuid.uuid4())
        profile = cat_db.get_cat_profile(fake_id)
        assert profile is None

    def test_get_cat_profile_db_error(self, cat_db):
        with patch.object(cat_db, 'get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("DB error")
            with pytest.raises(sqlite3.Error):
                cat_db.get_cat_profile("test_id")


class TestGetCatVisits:
    def test_get_cat_visits_basic(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        # Record multiple visits
        cat_db.record_visit(cat_id, "/clip1.mp4", "/frame1.jpg", {"behavior": "eating"}, 0.8, 0.1)
        cat_db.record_visit(cat_id, "/clip2.mp4", "/frame2.jpg", {"behavior": "sleeping"}, 0.9, 0.05)
        
        visits = cat_db.get_cat_visits(cat_id)
        assert len(visits) == 2
        # Should be ordered newest first
        assert visits[0]["behavior"] == "sleeping"
        assert visits[1]["behavior"] == "eating"

    def test_get_cat_visits_with_limit(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        for i in range(5):
            cat_db.record_visit(cat_id, f"/clip{i}.mp4", f"/frame{i}.jpg", {}, 0.8, 0.1)
        
        visits = cat_db.get_cat_visits(cat_id, limit=3)
        assert len(visits) == 3

    def test_get_cat_visits_empty(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        visits = cat_db.get_cat_visits(cat_id)
        assert len(visits) == 0

    def test_get_cat_visits_db_error(self, cat_db):
        with patch.object(cat_db, 'get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("DB error")
            with pytest.raises(sqlite3.Error):
                cat_db.get_cat_visits("test_id")


class TestGetRecentVisits:
    def test_get_recent_visits_basic(self, cat_db):
        cat_id1 = cat_db.create_cat_profile({"description": "Cat 1"})
        cat_id2 = cat_db.create_cat_profile({"description": "Cat 2"})
        
        cat_db.record_visit(cat_id1, "/clip1.mp4", "/frame1.jpg", {}, 0.8, 0.1)
        cat_db.record_visit(cat_id2, "/clip2.mp4", "/frame2.jpg", {}, 0.9, 0.05)
        
        visits = cat_db.get_recent_visits(days=7)
        assert len(visits) == 2
        
        # Should include cat profile data
        assert all("cat_description" in visit for visit in visits)

    def test_get_recent_visits_empty(self, cat_db):
        visits = cat_db.get_recent_visits(days=7)
        assert len(visits) == 0

    def test_get_recent_visits_db_error(self, cat_db):
        with patch.object(cat_db, 'get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("DB error")
            with pytest.raises(sqlite3.Error):
                cat_db.get_recent_visits()


class TestUpdateCatStatus:
    def test_update_cat_status_valid(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        cat_db.update_cat_status(cat_id, "active")
        
        profile = cat_db.get_cat_profile(cat_id)
        assert profile["status"] == "active"

    def test_update_cat_status_invalid(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        cat_db.update_cat_status(cat_id, "invalid_status")
        
        profile = cat_db.get_cat_profile(cat_id)
        assert profile["status"] == "active"  # Falls back to default

    def test_update_cat_status_db_error(self, cat_db):
        with patch.object(cat_db, 'get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("DB error")
            with pytest.raises(sqlite3.Error):
                cat_db.update_cat_status("test_id", "active")


class TestCreateAlert:
    def test_create_alert_basic(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        alert_id = cat_db.create_alert(
            cat_id=cat_id,
            alert_type="health_concern",
            message="Cat appears lethargic",
            severity="warning"
        )
        
        assert isinstance(alert_id, str)
        alerts = cat_db.get_active_alerts(cat_id)
        assert len(alerts) == 1
        assert alerts[0]["message"] == "Cat appears lethargic"
        assert alerts[0]["severity"] == "warning"

    def test_create_alert_invalid_severity(self, cat_db):
        cat_id = cat_db.create_cat_profile({"description": "Test cat"})
        
        alert_id = cat_db.create_alert(
            cat_id=cat_id,
            alert_type="health_concern", 
            message="Test",
            severity="invalid"
        )
        
        alerts = cat_db.get_active_alerts(cat_id)
        assert alerts[0]["severity"] == "info"  # Falls back to default

    def test_create_alert_nonexistent_cat(self, cat_db):
        fake_cat_id = str(uuid.uuid4())
        
        alert_id = cat_db.create_alert(
            cat_id=fake_cat_id,
            alert_type="new_cat",
            message="New cat detected"
        )
        
        assert isinstance(alert_id, str)
        # Should create placeholder cat
        profile = cat_db.get_cat_profile(fake_cat_id)
        assert profile is not None


class TestGetActiveAlerts:
    def test_get_active_alerts_all(self, cat_db):
        cat_id1 = cat_db.create_cat_profile({"description": "Cat 1"})
        cat_id2 = cat_db.create_cat_profile({"description": "Cat 2"})
        
        cat_db.create_alert(cat_id1, "health_concern", "Alert 1")
        cat_db.create_alert(cat_id2, "new_cat", "Alert 2")
        
        alerts = cat_db.get_active_alerts()
        assert len(alerts) == 2

    def test_get_active_alerts_filtered(self, cat_db):
        cat_id1 = cat_db.create_cat_profile({"description": "Cat 1"})
        cat_id2 = cat_db.create_cat_profile({"description": "Cat 2"})
        
        cat_db.create_alert(cat_id1, "health_concern", "Alert 1")
        cat_db.create_alert(cat_id2, "new_cat", "Alert 2")
        
        alerts = cat_db.get_active_alerts(cat_id=cat_id1)
        assert len(alerts) == 1
        assert alerts[0]["message"] == "Alert 1"

    def test_get_active_alerts_empty(self, cat_db):
        alerts = cat_db.get_active_alerts()
        assert len(alerts) == 0

    def test_get_active_alerts_db_error(self, cat_db):
        with patch.object(cat_db, 'get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("DB error")
            with pytest.raises(sqlite3.Error):
                cat_db.get_active_alerts()