import json
import os
import sys
import tempfile
from unittest.mock import mock_open, patch
import pytest

import config


@pytest.fixture
def temp_config():
    """Provide a temporary valid config for testing."""
    return {
        "models": {
            "vision": {
                "model_id": "test-vision-model",
                "endpoint": "http://localhost:8000/v1",
                "port": 8000
            }
        },
        "paths": {
            "uploads": "/uploads",
            "data": "/data"
        },
        "thresholds": {
            "similarity_known": 0.25
        }
    }


def test_load_config_happy_path():
    """Test successful config loading."""
    test_config = {"test": "value", "nested": {"key": "data"}}
    mock_file_content = json.dumps(test_config)
    
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        result = config.load_config()
    
    assert result == test_config
    assert isinstance(result, dict)


def test_load_config_file_not_found():
    """Test FileNotFoundError handling."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                config.load_config()
    
    mock_print.assert_called_once()
    mock_exit.assert_called_once_with(1)
    print_args = mock_print.call_args[0][0]
    assert "FATAL" in print_args
    assert "sys_config.json not found" in print_args


def test_load_config_json_decode_error():
    """Test malformed JSON handling."""
    malformed_json = '{"invalid": json,}'
    
    with patch("builtins.open", mock_open(read_data=malformed_json)):
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                config.load_config()
    
    mock_print.assert_called_once()
    mock_exit.assert_called_once_with(1)
    print_args = mock_print.call_args[0][0]
    assert "FATAL" in print_args
    assert "malformed" in print_args


def test_load_config_non_dict_content():
    """Test non-dict JSON content handling."""
    non_dict_json = '"string_value"'
    
    with patch("builtins.open", mock_open(read_data=non_dict_json)):
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                config.load_config()
    
    mock_print.assert_called_once()
    mock_exit.assert_called_once_with(1)
    print_args = mock_print.call_args[0][0]
    assert "FATAL" in print_args
    assert "must be a JSON object" in print_args


def test_load_config_empty_dict():
    """Test loading empty dict."""
    empty_config = "{}"
    
    with patch("builtins.open", mock_open(read_data=empty_config)):
        result = config.load_config()
    
    assert result == {}
    assert isinstance(result, dict)


def test_reload_cfg_happy_path():
    """Test reload_cfg function."""
    test_config = {"reloaded": "config", "value": 123}
    mock_file_content = json.dumps(test_config)
    
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        result = config.reload_cfg()
    
    assert result == test_config
    assert isinstance(result, dict)


def test_reload_cfg_file_not_found():
    """Test reload_cfg with missing file."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                config.reload_cfg()
    
    mock_print.assert_called_once()
    mock_exit.assert_called_once_with(1)


def test_reload_cfg_json_error():
    """Test reload_cfg with malformed JSON."""
    bad_json = '{"bad": json}'
    
    with patch("builtins.open", mock_open(read_data=bad_json)):
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                config.reload_cfg()
    
    mock_print.assert_called_once()
    mock_exit.assert_called_once_with(1)


def test_cfg_module_level_variable(temp_config):
    """Test that cfg is loaded at module level."""
    mock_file_content = json.dumps(temp_config)
    
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        # Reload the module to test import-time loading
        import importlib
        importlib.reload(config)
    
    assert hasattr(config, 'cfg')
    assert isinstance(config.cfg, dict)


def test_config_path_construction():
    """Test that config path is constructed correctly."""
    expected_path = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), "sys_config.json")
    assert config._CONFIG_PATH == expected_path