import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, call
from watcher import FileWatcher, setup_logging, main, VIDEO_EXTENSIONS, POLL_INTERVAL


@pytest.fixture(autouse=True)
def mock_config():
    config_data = {
        "paths": {
            "uploads": "/uploads",
            "logs": "/data/logs"
        },
        "thresholds": {
            "file_stable_seconds": 3
        },
        "logging": {
            "level": "INFO",
            "file": "/data/logs/catmonitor.log"
        }
    }
    with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
        yield


class TestFileWatcher:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path / "uploads"
    
    @pytest.fixture
    def file_watcher(self, temp_dir):
        temp_dir.mkdir(parents=True, exist_ok=True)
        return FileWatcher(str(temp_dir), stable_seconds=6)
    
    def test_init_default_stable_seconds(self, temp_dir):
        watcher = FileWatcher(str(temp_dir))
        assert watcher.upload_dir == temp_dir
        assert watcher.stable_seconds == 3
        assert watcher.known_files == set()
        assert watcher.file_sizes == {}
        assert watcher.file_stable_counts == {}
    
    def test_init_custom_stable_seconds(self, temp_dir):
        watcher = FileWatcher(str(temp_dir), stable_seconds=10)
        assert watcher.stable_seconds == 10
    
    @patch('pathlib.Path.stat')
    def test_is_file_stable_first_check(self, mock_stat, file_watcher):
        mock_stat.return_value.st_size = 1000
        filepath = file_watcher.upload_dir / "test.mp4"
        
        result = file_watcher.is_file_stable(filepath)
        
        assert result is False
        assert file_watcher.file_sizes[str(filepath)] == 1000
        assert file_watcher.file_stable_counts[str(filepath)] == 0
    
    @patch('pathlib.Path.stat')
    def test_is_file_stable_size_changed(self, mock_stat, file_watcher):
        filepath = file_watcher.upload_dir / "test.mp4"
        file_watcher.file_sizes[str(filepath)] = 1000
        file_watcher.file_stable_counts[str(filepath)] = 2
        
        mock_stat.return_value.st_size = 1500
        
        result = file_watcher.is_file_stable(filepath)
        
        assert result is False
        assert file_watcher.file_sizes[str(filepath)] == 1500
        assert file_watcher.file_stable_counts[str(filepath)] == 0
    
    @patch('pathlib.Path.stat')
    def test_is_file_stable_becomes_stable(self, mock_stat, file_watcher):
        filepath = file_watcher.upload_dir / "test.mp4"
        file_watcher.file_sizes[str(filepath)] = 1000
        file_watcher.file_stable_counts[str(filepath)] = 2
        
        mock_stat.return_value.st_size = 1000
        
        result = file_watcher.is_file_stable(filepath)
        
        assert result is True
        assert file_watcher.file_stable_counts[str(filepath)] == 3
    
    @patch('pathlib.Path.stat', side_effect=OSError("Permission denied"))
    def test_is_file_stable_stat_error(self, mock_stat, file_watcher):
        filepath = file_watcher.upload_dir / "test.mp4"
        
        with patch('watcher.logger') as mock_logger:
            result = file_watcher.is_file_stable(filepath)
            
            assert result is False
            mock_logger.warning.assert_called_once()
    
    @patch('os.open')
    @patch('os.close')
    def test_is_file_locked_not_locked(self, mock_close, mock_open, file_watcher):
        filepath = file_watcher.upload_dir / "test.mp4"
        mock_open.return_value = 3
        
        result = file_watcher.is_file_locked(filepath)
        
        assert result is False
        mock_open.assert_called_once_with(str(filepath), os.O_RDONLY | os.O_EXCL)
        mock_close.assert_called_once_with(3)
    
    @patch('os.open', side_effect=OSError("File is locked"))
    def test_is_file_locked_is_locked(self, mock_open, file_watcher):
        filepath = file_watcher.upload_dir / "test.mp4"
        
        result = file_watcher.is_file_locked(filepath)
        
        assert result is True
    
    def test_scan_for_new_files_empty_directory(self, file_watcher):
        result = file_watcher.scan_for_new_files()
        assert result == []
    
    def test_scan_for_new_files_non_video_files(self, file_watcher):
        (file_watcher.upload_dir / "test.txt").touch()
        (file_watcher.upload_dir / "image.jpg").touch()
        
        result = file_watcher.scan_for_new_files()
        assert result == []
    
    def test_scan_for_new_files_known_files(self, file_watcher):
        video_file = file_watcher.upload_dir / "test.mp4"
        video_file.touch()
        file_watcher.known_files.add(str(video_file))
        
        result = file_watcher.scan_for_new_files()
        assert result == []
    
    @patch.object(FileWatcher, 'is_file_stable', return_value=False)
    def test_scan_for_new_files_unstable_file(self, mock_stable, file_watcher):
        video_file = file_watcher.upload_dir / "test.mp4"
        video_file.touch()
        
        with patch('watcher.logger') as mock_logger:
            result = file_watcher.scan_for_new_files()
            
            assert result == []
            mock_logger.debug.assert_called_with("File not yet stable: %s", "test.mp4")
    
    @patch.object(FileWatcher, 'is_file_stable', return_value=True)
    @patch.object(FileWatcher, 'is_file_locked', return_value=True)
    def test_scan_for_new_files_locked_file(self, mock_locked, mock_stable, file_watcher):
        video_file = file_watcher.upload_dir / "test.mp4"
        video_file.touch()
        
        with patch('watcher.logger') as mock_logger:
            result = file_watcher.scan_for_new_files()
            
            assert result == []
            mock_logger.debug.assert_called_with("File still locked: %s", "test.mp4")
    
    @patch.object(FileWatcher, 'is_file_stable', return_value=True)
    @patch.object(FileWatcher, 'is_file_locked', return_value=False)
    def test_scan_for_new_files_ready_file(self, mock_locked, mock_stable, file_watcher):
        video_file = file_watcher.upload_dir / "test.mp4"
        video_file.touch()
        
        result = file_watcher.scan_for_new_files()
        
        assert len(result) == 1
        assert result[0].name == "test.mp4"
    
    @patch.object(FileWatcher, 'is_file_stable', return_value=True)
    @patch.object(FileWatcher, 'is_file_locked', return_value=False)
    def test_scan_for_new_files_multiple_video_extensions(self, mock_locked, mock_stable, file_watcher):
        for ext in VIDEO_EXTENSIONS:
            (file_watcher.upload_dir / f"test{ext}").touch()
        
        result = file_watcher.scan_for_new_files()
        
        assert len(result) == len(VIDEO_EXTENSIONS)
    
    def test_scan_for_new_files_directory_scan_error(self, file_watcher):
        with patch.object(file_watcher.upload_dir, 'iterdir', side_effect=OSError("Directory error")):
            with pytest.raises(OSError):
                file_watcher.scan_for_new_files()
    
    @patch('time.sleep')
    @patch('watcher.process_video_clip')
    @patch.object(FileWatcher, 'scan_for_new_files')
    def test_run_processes_files(self, mock_scan, mock_process, mock_sleep, file_watcher):
        video_path = file_watcher.upload_dir / "test.mp4"
        mock_scan.side_effect = [[video_path], KeyboardInterrupt]
        
        with patch('watcher.logger') as mock_logger:
            file_watcher.run()
            
            mock_process.assert_called_once_with(str(video_path))
            assert str(video_path) in file_watcher.known_files
            mock_logger.info.assert_any_call("New file detected, dispatching: %s", "test.mp4")
    
    @patch('time.sleep')
    @patch('watcher.process_video_clip', side_effect=Exception("Processing error"))
    @patch.object(FileWatcher, 'scan_for_new_files')
    def test_run_handles_processing_error(self, mock_scan, mock_process, mock_sleep, file_watcher):
        video_path = file_watcher.upload_dir / "test.mp4"
        mock_scan.side_effect = [[video_path], KeyboardInterrupt]
        
        with patch('watcher.logger') as mock_logger:
            file_watcher.run()
            
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args
            assert "Processing failed for %s" in error_call[0][0]
            assert "test.mp4" in error_call[0]
    
    @patch('time.sleep')
    @patch.object(FileWatcher, 'scan_for_new_files', side_effect=OSError("Scan error"))
    def test_run_handles_scan_error(self, mock_scan, mock_sleep, file_watcher):
        mock_scan.side_effect = [OSError("Scan error"), KeyboardInterrupt]
        
        with patch('watcher.logger') as mock_logger:
            file_watcher.run()
            
            mock_logger.error.assert_called()
            mock_sleep.assert_called()
    
    def test_run_cleans_tracking_state(self, file_watcher):
        video_path = file_watcher.upload_dir / "test.mp4"
        file_watcher.file_sizes[str(video_path)] = 1000
        file_watcher.file_stable_counts[str(video_path)] = 3
        
        with patch.object(file_watcher, 'scan_for_new_files') as mock_scan:
            with patch('watcher.process_video_clip'):
                with patch('time.sleep'):
                    mock_scan.side_effect = [[video_path], KeyboardInterrupt]
                    
                    file_watcher.run()
                    
                    assert str(video_path) not in file_watcher.file_sizes
                    assert str(video_path) not in file_watcher.file_stable_counts


@patch('pathlib.Path.mkdir')
@patch('logging.basicConfig')
def test_setup_logging(mock_basicconfig, mock_mkdir):
    setup_logging()
    
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_basicconfig.assert_called_once()
    
    # Check that logging configuration includes both file and console handlers
    args, kwargs = mock_basicconfig.call_args
    assert 'handlers' in kwargs
    assert len(kwargs['handlers']) == 2


@patch('watcher.setup_logging')
@patch('watcher.FileWatcher')
def test_main_success(mock_filewatcher_class, mock_setup_logging):
    mock_watcher = Mock()
    mock_filewatcher_class.return_value = mock_watcher
    mock_watcher.run.side_effect = KeyboardInterrupt
    
    main()
    
    mock_setup_logging.assert_called_once()
    mock_filewatcher_class.assert_called_once_with("/uploads", 3)
    mock_watcher.run.assert_called_once()


@patch('watcher.setup_logging')
@patch('watcher.FileWatcher')
def test_main_keyboard_interrupt(mock_filewatcher_class, mock_setup_logging):
    mock_watcher = Mock()
    mock_filewatcher_class.return_value = mock_watcher
    mock_watcher.run.side_effect = KeyboardInterrupt
    
    with patch('watcher.logging.getLogger') as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        main()
        
        mock_logger.info.assert_any_call("Watcher stopped by user")


@patch('watcher.setup_logging')
@patch('watcher.FileWatcher')
def test_main_exception(mock_filewatcher_class, mock_setup_logging):
    mock_watcher = Mock()
    mock_filewatcher_class.return_value = mock_watcher
    mock_watcher.run.side_effect = Exception("Test error")
    
    with patch('watcher.logging.getLogger') as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with pytest.raises(Exception, match="Test error"):
            main()
        
        mock_logger.error.assert_called()