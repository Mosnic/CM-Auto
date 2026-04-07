import sqlite3
import logging
import uuid
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager

from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL: Dict[str, str] = {
    "cats": """
        CREATE TABLE IF NOT EXISTS cats (
            cat_id TEXT PRIMARY KEY,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            visit_count INTEGER DEFAULT 1,
            description TEXT,
            status TEXT DEFAULT 'new',
            coat_color TEXT,
            eye_color TEXT,
            last_body_condition TEXT,
            distinctive_markings TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "visits": """
        CREATE TABLE IF NOT EXISTS visits (
            visit_id TEXT PRIMARY KEY,
            cat_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            camera TEXT,
            clip_path TEXT,
            best_frame_path TEXT,
            behavior TEXT,
            body_condition TEXT,
            health_flags TEXT DEFAULT '[]',
            confidence REAL,
            embedding_distance REAL,
            notes TEXT,
            FOREIGN KEY (cat_id) REFERENCES cats (cat_id)
        )
    """,
    "alerts": """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            cat_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP NULL,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (cat_id) REFERENCES cats (cat_id)
        )
    """,
    "links": """
        CREATE TABLE IF NOT EXISTS links (
            link_id TEXT PRIMARY KEY,
            cat_id_1 TEXT NOT NULL,
            cat_id_2 TEXT NOT NULL,
            link_type TEXT NOT NULL,
            confidence REAL,
            evidence TEXT DEFAULT '[]',
            status TEXT DEFAULT 'provisional',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cat_id_1) REFERENCES cats (cat_id),
            FOREIGN KEY (cat_id_2) REFERENCES cats (cat_id)
        )
    """,
}

CAT_STATUSES = ["new", "active", "absent", "returning", "provisional_link"]
ALERT_SEVERITIES = ["info", "warning", "error", "critical"]
ALERT_TYPES = ["health_concern", "absence_pattern", "new_cat", "uncertain_match", "system_error"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def dict_factory(cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
    """Row factory to return rows as dictionaries."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def serialize_json_field(data: Union[List, Dict]) -> str:
    """Serialize a list or dict for JSON storage in SQLite."""
    return json.dumps(data) if data is not None else "[]"


def deserialize_json_field(json_str: str) -> Union[List, Dict]:
    """Deserialize a JSON field retrieved from SQLite."""
    try:
        return json.loads(json_str) if json_str else []
    except json.JSONDecodeError:
        logger.warning("Failed to deserialize JSON field; returning empty list")
        return []


def _safe_uuid(value: Optional[str]) -> str:
    """Validate a UUID string; generate a fresh one if invalid or None."""
    if value:
        try:
            uuid.UUID(value)
            return value
        except (ValueError, AttributeError):
            logger.warning("Invalid UUID %r — generating a new one", value)
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CatDatabase:
    """SQLite database interface for cat monitoring data."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize database, creating schema if necessary.

        Args:
            db_path: Optional path override for the database file.
        """
        self.db_path = db_path or cfg["paths"]["database"]
        self.logger = logging.getLogger(__name__)
        self._ensure_directory()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_directory(self) -> None:
        """Create the parent directory of the database file if absent."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create database directory %s: %s", Path(self.db_path).parent, exc)

    def _init_schema(self) -> None:
        """Create all tables that do not yet exist."""
        try:
            with self.get_connection() as conn:
                # Enable WAL for better concurrent read performance
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                for table_name, ddl in CREATE_TABLES_SQL.items():
                    conn.execute(ddl)
                    logger.debug("Ensured table: %s", table_name)
        except sqlite3.Error as exc:
            logger.error("Schema initialisation failed: %s", exc)
            raise

    @contextmanager
    def get_connection(self):
        """Context manager yielding a SQLite connection with auto commit/rollback."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error("Database error — rolling back: %s", exc)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Cat profiles
    # ------------------------------------------------------------------

    def create_cat_profile(
        self,
        analysis: Dict[str, Any],
        cat_id: Optional[str] = None,
    ) -> str:
        """Insert a new cat profile derived from a vision analysis dict.

        Args:
            analysis: Vision analysis results (from VisionTool).
            cat_id:   Optional specific UUID to use for the new profile.

        Returns:
            The UUID string of the created profile.
        """
        cat_id = _safe_uuid(cat_id)
        now = datetime.utcnow().isoformat()

        # Pull normalised fields from the analysis payload gracefully
        appearance = analysis.get("appearance", {})
        description = analysis.get("description") or analysis.get("summary", "")
        coat_color = appearance.get("coat_color") or appearance.get("color", "")
        eye_color = appearance.get("eye_color", "")
        body_condition = analysis.get("body_condition", "")
        markings = appearance.get("distinctive_markings") or []

        sql = """
            INSERT INTO cats (
                cat_id, first_seen, last_seen, visit_count,
                description, status,
                coat_color, eye_color, last_body_condition,
                distinctive_markings, created_at, updated_at
            ) VALUES (
                :cat_id, :now, :now, 1,
                :description, 'new',
                :coat_color, :eye_color, :body_condition,
                :markings, :now, :now
            )
        """
        params = {
            "cat_id": cat_id,
            "now": now,
            "description": description,
            "coat_color": coat_color,
            "eye_color": eye_color,
            "body_condition": body_condition,
            "markings": serialize_json_field(markings),
        }

        try:
            with self.get_connection() as conn:
                conn.execute(sql, params)
            logger.info("Created cat profile: %s", cat_id)
            return cat_id
        except sqlite3.IntegrityError as exc:
            logger.error("Constraint violation creating cat %s: %s", cat_id, exc)
            raise RuntimeError(f"Could not create cat profile {cat_id}") from exc

    def update_cat_profile(self, cat_id: str, analysis: Dict[str, Any]) -> None:
        """Refresh an existing cat profile with data from a new visit.

        Args:
            cat_id:   UUID of the cat to update.
            analysis: Latest analysis data.
        """
        cat_id = _safe_uuid(cat_id)
        now = datetime.utcnow().isoformat()
        appearance = analysis.get("appearance", {})
        body_condition = analysis.get("body_condition", "")
        markings = appearance.get("distinctive_markings") or []
        description = analysis.get("description") or analysis.get("summary", "")
        coat_color = appearance.get("coat_color") or appearance.get("color", "")
        eye_color = appearance.get("eye_color", "")

        sql = """
            UPDATE cats SET
                last_seen = :now,
                visit_count = visit_count + 1,
                description = COALESCE(NULLIF(:description, ''), description),
                coat_color = COALESCE(NULLIF(:coat_color, ''), coat_color),
                eye_color = COALESCE(NULLIF(:eye_color, ''), eye_color),
                last_body_condition = COALESCE(NULLIF(:body_condition, ''), last_body_condition),
                distinctive_markings = CASE
                    WHEN :markings != '[]' THEN :markings
                    ELSE distinctive_markings
                END,
                updated_at = :now
            WHERE cat_id = :cat_id
        """
        params = {
            "cat_id": cat_id,
            "now": now,
            "description": description,
            "coat_color": coat_color,
            "eye_color": eye_color,
            "body_condition": body_condition,
            "markings": serialize_json_field(markings),
        }

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(sql, params)
            logger.debug("Updated cat profile: %s", cat_id)
        except sqlite3.Error as exc:
            logger.error("Failed to update cat %s: %s", cat_id, exc)
            raise

    # ------------------------------------------------------------------
    # Visits
    # ------------------------------------------------------------------

    def record_visit(
        self,
        cat_id: str,
        clip_path: str,
        frame_path: str,
        analysis: Dict[str, Any],
        confidence: float,
        distance: float,
    ) -> str:
        """Persist a single visit event.

        Args:
            cat_id:      UUID of the identified cat.
            clip_path:   Path to the source video clip.
            frame_path:  Path to the best representative frame.
            analysis:    Full vision analysis payload.
            confidence:  Identity match confidence (0–1).
            distance:    Embedding distance score.

        Returns:
            UUID string of the newly created visit record.
        """
        visit_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        behavior = analysis.get("behavior", "")
        body_condition = analysis.get("body_condition", "")
        health_flags = analysis.get("health_flags") or []
        camera = analysis.get("camera", "")
        notes = analysis.get("notes", "")

        sql = """
            INSERT INTO visits (
                visit_id, cat_id, timestamp, camera,
                clip_path, best_frame_path,
                behavior, body_condition, health_flags,
                confidence, embedding_distance, notes
            ) VALUES (
                :visit_id, :cat_id, :now, :camera,
                :clip_path, :frame_path,
                :behavior, :body_condition, :health_flags,
                :confidence, :distance, :notes
            )
        """
        params = {
            "visit_id": visit_id,
            "cat_id": cat_id,
            "now": now,
            "camera": camera,
            "clip_path": clip_path,
            "frame_path": frame_path,
            "behavior": behavior,
            "body_condition": body_condition,
            "health_flags": serialize_json_field(health_flags),
            "confidence": confidence,
            "distance": distance,
            "notes": notes,
        }

        # Guard: ensure the cat profile exists before inserting the FK
        if not self.get_cat_profile(cat_id):
            logger.warning(
                "Cat %s not found when recording visit — creating placeholder", cat_id
            )
            self.create_cat_profile({}, cat_id=cat_id)

        try:
            with self.get_connection() as conn:
                conn.execute(sql, params)
            logger.info("Recorded visit %s for cat %s", visit_id, cat_id)
            return visit_id
        except sqlite3.IntegrityError as exc:
            logger.error("Constraint violation recording visit for %s: %s", cat_id, exc)
            raise RuntimeError(f"Could not record visit for cat {cat_id}") from exc

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_cat_profile(self, cat_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a cat profile row by UUID.

        Returns:
            Profile dict or None if not found.
        """
        try:
            with self.get_connection() as conn:
                raw = conn.execute(
                    "SELECT * FROM cats WHERE cat_id = ?", (cat_id,)
                ).fetchone()
            if raw is None:
                return None
            row = dict(raw)
            row["distinctive_markings"] = deserialize_json_field(
                row.get("distinctive_markings", "[]")
            )
            return row
        except sqlite3.Error as exc:
            logger.error("Failed to fetch cat profile %s: %s", cat_id, exc)
            raise

    def get_cat_visits(
        self,
        cat_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return visits for a specific cat, newest first.

        Args:
            cat_id: UUID of the cat.
            limit:  If given, cap results at this count.

        Returns:
            List of visit dicts.
        """
        sql = "SELECT * FROM visits WHERE cat_id = ? ORDER BY timestamp DESC"
        params: list = [cat_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        try:
            with self.get_connection() as conn:
                raws = conn.execute(sql, params).fetchall()
            rows = [dict(r) for r in raws]
            for row in rows:
                row["health_flags"] = deserialize_json_field(row.get("health_flags", "[]"))
            return rows
        except sqlite3.Error as exc:
            logger.error("Failed to fetch visits for cat %s: %s", cat_id, exc)
            raise

    def get_recent_visits(self, days: int = 7) -> List[Dict[str, Any]]:
        """Return visits across all cats within the past *days* days.

        Joins cat profile columns so callers get a single enriched dict.

        Args:
            days: Look-back window in days.

        Returns:
            List of visit+cat dicts ordered by visit timestamp descending.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        sql = """
            SELECT v.*, c.coat_color, c.eye_color, c.status AS cat_status,
                   c.description AS cat_description
            FROM visits v
            JOIN cats c ON c.cat_id = v.cat_id
            WHERE v.timestamp >= ?
            ORDER BY v.timestamp DESC
        """
        try:
            with self.get_connection() as conn:
                raws = conn.execute(sql, (cutoff,)).fetchall()
            rows = [dict(r) for r in raws]
            for row in rows:
                row["health_flags"] = deserialize_json_field(row.get("health_flags", "[]"))
            return rows
        except sqlite3.Error as exc:
            logger.error("Failed to fetch recent visits: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def update_cat_status(self, cat_id: str, status: str) -> None:
        """Set the status column on a cat profile.

        Args:
            cat_id: UUID of the cat.
            status: One of CAT_STATUSES (validated here; falls back to 'active').
        """
        if status not in CAT_STATUSES:
            logger.warning(
                "Unknown status %r for cat %s — defaulting to 'active'", status, cat_id
            )
            status = "active"

        now = datetime.utcnow().isoformat()
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE cats SET status = ?, updated_at = ? WHERE cat_id = ?",
                    (status, now, cat_id),
                )
            logger.debug("Cat %s status → %s", cat_id, status)
        except sqlite3.Error as exc:
            logger.error("Failed to update status for cat %s: %s", cat_id, exc)
            raise

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def create_alert(
        self,
        cat_id: str,
        alert_type: str,
        message: str,
        severity: str = "info",
    ) -> str:
        """Insert a new alert record.

        Args:
            cat_id:     UUID of the associated cat.
            alert_type: Category from ALERT_TYPES.
            message:    Human-readable description.
            severity:   One of ALERT_SEVERITIES.

        Returns:
            UUID string of the created alert.
        """
        if severity not in ALERT_SEVERITIES:
            logger.warning("Unknown severity %r — using 'info'", severity)
            severity = "info"
        if alert_type not in ALERT_TYPES:
            logger.warning("Unrecognised alert_type %r", alert_type)

        alert_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Ensure cat exists; create placeholder if not.
        # Wrap in try/except so a DB error during the check doesn't abort the
        # insert — the FK constraint will surface any real violation.
        try:
            if not self.get_cat_profile(cat_id):
                logger.warning(
                    "Cat %s not found when creating alert — creating placeholder", cat_id
                )
                self.create_cat_profile({}, cat_id=cat_id)
        except Exception:
            pass

        sql = """
            INSERT INTO alerts (alert_id, cat_id, alert_type, message, severity, created_at, status)
            VALUES (:alert_id, :cat_id, :alert_type, :message, :severity, :now, 'active')
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    sql,
                    {
                        "alert_id": alert_id,
                        "cat_id": cat_id,
                        "alert_type": alert_type,
                        "message": message,
                        "severity": severity,
                        "now": now,
                    },
                )
            logger.info(
                "Created %s alert %s for cat %s", severity, alert_id, cat_id
            )
            return alert_id
        except sqlite3.IntegrityError as exc:
            logger.error("Constraint violation creating alert for %s: %s", cat_id, exc)
            raise RuntimeError(f"Could not create alert for cat {cat_id}") from exc

    def get_active_alerts(
        self, cat_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all active (unresolved) alerts.

        Args:
            cat_id: Optional filter; if given, return only alerts for that cat.

        Returns:
            List of alert dicts ordered by created_at descending.
        """
        if cat_id:
            sql = """
                SELECT * FROM alerts
                WHERE status = 'active' AND cat_id = ?
                ORDER BY created_at DESC
            """
            params: tuple = (cat_id,)
        else:
            sql = """
                SELECT * FROM alerts
                WHERE status = 'active'
                ORDER BY created_at DESC
            """
            params = ()

        try:
            with self.get_connection() as conn:
                raws = conn.execute(sql, params).fetchall()
            return [dict(r) for r in raws]
        except sqlite3.Error as exc:
            logger.error("Failed to fetch active alerts: %s", exc)
            raise
