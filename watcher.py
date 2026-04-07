import os
import time
import logging
from pathlib import Path
from typing import Set, Optional
from config import cfg
from agent.loop import process_video_clip

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
POLL_INTERVAL = 2  # seconds between directory scans

logger = logging.getLogger(__name__)


class _MutablePath(type(Path())):
    """Concrete Path subclass whose instances have __dict__, allowing patch.object."""
    pass


class FileWatcher:
    """Monitors upload directory for completed file transfers."""

    def __init__(self, upload_dir: str, stable_seconds: int = 3):
        """Initialize watcher with upload directory and stability timeout.

        Args:
            upload_dir: Path to monitor for new files
            stable_seconds: Seconds file size must remain unchanged to consider complete
        """
        self.upload_dir = _MutablePath(upload_dir)
        self.stable_seconds = stable_seconds
        self.known_files: Set[str] = set()
        self.file_sizes: dict = {}         # filepath -> last observed size
        self.file_stable_counts: dict = {} # filepath -> consecutive stable polls

    def is_file_stable(self, filepath: Path) -> bool:
        """Check if file size has been stable for required duration.

        We track how many consecutive POLL_INTERVAL-second cycles the size has
        not changed.  stable_seconds / POLL_INTERVAL gives the required count.

        Args:
            filepath: Path to file to check

        Returns:
            bool: True if file is stable and ready for processing
        """
        try:
            current_size = filepath.stat().st_size
        except OSError as exc:
            logger.warning("Cannot stat %s: %s", filepath, exc)
            return False

        key = str(filepath)
        last_size = self.file_sizes.get(key)

        if last_size is None or last_size != current_size:
            # Size changed (or first observation) — reset counter
            self.file_sizes[key] = current_size
            self.file_stable_counts[key] = 0
            return False

        # Size unchanged — increment stable counter
        self.file_stable_counts[key] = self.file_stable_counts.get(key, 0) + 1

        # Required stable polls = ceil(stable_seconds / POLL_INTERVAL)
        required_polls = max(1, int(self.stable_seconds / POLL_INTERVAL))
        return self.file_stable_counts[key] >= required_polls

    def is_file_locked(self, filepath: Path) -> bool:
        """Check if file is currently locked by another process.

        Uses a portable exclusive-open via os.O_RDONLY | os.O_EXCL so we avoid
        any dependency on external tools like lsof.

        Args:
            filepath: Path to file to check

        Returns:
            bool: True if file appears to be locked/in-use
        """
        try:
            fd = os.open(str(filepath), os.O_RDONLY | os.O_EXCL)
            os.close(fd)
            return False  # opened exclusively — no other writer holds it
        except OSError:
            # EBUSY / EACCES / EWOULDBLOCK all indicate the file is in use
            return True

    def scan_for_new_files(self) -> list[Path]:
        """Scan directory for new video files that are ready for processing.

        Returns:
            list[Path]: List of stable, unlocked video files ready for processing
        """
        ready: list[Path] = []

        try:
            entries = list(self.upload_dir.iterdir())
        except OSError as exc:
            logger.error("Failed to scan upload directory %s: %s", self.upload_dir, exc)
            raise

        for entry in entries:
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if str(entry) in self.known_files:
                continue  # already dispatched for processing

            if not self.is_file_stable(entry):
                logger.debug("File not yet stable: %s", entry.name)
                continue

            if self.is_file_locked(entry):
                logger.debug("File still locked: %s", entry.name)
                continue

            ready.append(entry)

        return ready

    def run(self) -> None:
        """Main watcher loop - runs indefinitely monitoring for new files."""
        logger.info(
            "FileWatcher started. upload_dir=%s stable_seconds=%s poll_interval=%ss",
            self.upload_dir, self.stable_seconds, POLL_INTERVAL,
        )

        while True:
            try:
                ready_files = self.scan_for_new_files()
            except KeyboardInterrupt:
                break
            except OSError as exc:
                # Directory-level error is fatal for this iteration; log and retry
                logger.error("Directory scan error: %s", exc, exc_info=True)
                time.sleep(POLL_INTERVAL)
                continue

            for filepath in ready_files:
                logger.info("New file detected, dispatching: %s", filepath.name)
                self.known_files.add(str(filepath))
                # Clean up tracking state — no longer needed after dispatch
                self.file_sizes.pop(str(filepath), None)
                self.file_stable_counts.pop(str(filepath), None)

                try:
                    process_video_clip(str(filepath))
                except Exception as exc:
                    logger.error(
                        "Processing failed for %s: %s", filepath.name, exc, exc_info=True
                    )
                    # Continue watching — don't let one bad file crash the watcher

            time.sleep(POLL_INTERVAL)


def setup_logging() -> None:
    """Configure logging to file with rotation."""
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(cfg["logging"]["file"]),
            logging.StreamHandler()
        ]
    )


def main() -> None:
    """Entry point for watcher service."""
    setup_logging()
    log = logging.getLogger(__name__)

    watcher = FileWatcher(
        cfg["paths"]["uploads"],
        cfg["thresholds"]["file_stable_seconds"]
    )

    try:
        log.info("Starting file watcher on %s", cfg["paths"]["uploads"])
        watcher.run()
    except KeyboardInterrupt:
        log.info("Watcher stopped by user")
    except Exception as exc:
        log.error("Watcher failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
