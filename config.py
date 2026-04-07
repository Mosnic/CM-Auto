import json
import os
import sys
from typing import Any, Dict

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sys_config.json")


def load_config() -> Dict[str, Any]:
    """Load configuration from sys_config.json in the same directory as this file.

    Returns:
        Dict[str, Any]: Complete configuration dictionary

    Raises:
        FileNotFoundError: If sys_config.json is missing
        json.JSONDecodeError: If sys_config.json is malformed
    """
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(
                f"FATAL: sys_config.json must be a JSON object, got {type(data).__name__}.",
                file=sys.stderr,
            )
            sys.exit(1)
        return data
    except FileNotFoundError:
        print(
            f"FATAL: sys_config.json not found at '{_CONFIG_PATH}'. "
            "Ensure the config file is present in the application directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(
            f"FATAL: sys_config.json is malformed — {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}, char {exc.pos}).",
            file=sys.stderr,
        )
        sys.exit(1)


def reload_cfg() -> Dict[str, Any]:
    """Re-read sys_config.json from disk and return the updated config dict.

    Useful for tests that patch the file between calls. Does NOT update the
    module-level ``cfg`` reference; callers must rebind if they need that.

    Returns:
        Dict[str, Any]: Freshly loaded configuration dictionary
    """
    return load_config()


# Module-level singleton loaded once at import time.
# All other modules should do: from config import cfg
cfg: Dict[str, Any] = load_config()
