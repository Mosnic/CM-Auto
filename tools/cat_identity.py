import uuid
import sqlite3
import logging
import base64
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import chromadb
from chromadb.config import Settings
import requests
from config import cfg

# ---------------------------------------------------------------------------
# Module-level constants (resolved once at import time from live config)
# ---------------------------------------------------------------------------
KNOWN_THRESHOLD = cfg["thresholds"]["similarity_known"]
UNCERTAIN_THRESHOLD = cfg["thresholds"]["similarity_uncertain"]
CAT_STATUSES = ["active", "absent", "new", "returning", "provisional_link"]

CREATE_CATS_TABLE = """
CREATE TABLE IF NOT EXISTS cats (
    cat_id TEXT PRIMARY KEY,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    visit_count INTEGER DEFAULT 1,
    description TEXT,
    status TEXT DEFAULT 'new',
    coat_color TEXT,
    eye_color TEXT,
    last_body_condition TEXT
)
"""

CREATE_VISITS_TABLE = """
CREATE TABLE IF NOT EXISTS visits (
    visit_id TEXT PRIMARY KEY,
    cat_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera TEXT,
    clip_path TEXT,
    behavior TEXT,
    body_condition TEXT,
    health_flags TEXT,
    confidence REAL,
    best_frame_path TEXT,
    embedding_distance REAL,
    FOREIGN KEY (cat_id) REFERENCES cats (cat_id)
)
"""


# ---------------------------------------------------------------------------
# Utility functions (module-level, importable without class instantiation)
# ---------------------------------------------------------------------------

def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute average of embedding vectors.

    Args:
        embeddings: List of embedding vectors

    Returns:
        np.ndarray: Averaged embedding vector
    """
    return np.mean(embeddings, axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        float: Cosine distance (0=identical, 2=opposite)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0  # treat zero vectors as maximally distant
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def should_apply_hard_filter(coat_color: str, eye_color: str) -> bool:
    """Determine if hard metadata filtering should be applied.

    Args:
        coat_color: Detected coat color
        eye_color: Detected eye color

    Returns:
        bool: True if both colors are known and filtering should apply
    """
    return coat_color != "unknown" and eye_color != "unknown"


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class CatIdentityEngine:
    """Core engine for cat identification using embeddings and metadata filtering."""

    def __init__(self):
        """Initialize identity engine with database connections."""
        self.db_path = cfg["paths"]["database"]
        self.chroma_path = cfg["paths"]["chroma"]
        self.embedding_endpoint = cfg["models"]["embedding"]["endpoint"]
        self.logger = logging.getLogger(__name__)
        self.chroma_client = None
        self.collection = None
        self._init_databases()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_databases(self) -> None:
        """Initialize SQLite and ChromaDB connections."""
        self._init_sqlite()
        self._init_chromadb()

    def _init_sqlite(self) -> None:
        """Create SQLite tables if they don't exist."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            con = sqlite3.connect(str(db_path))
            with con:
                con.execute(CREATE_CATS_TABLE)
                con.execute(CREATE_VISITS_TABLE)
                con.execute("PRAGMA journal_mode=WAL;")  # safer for concurrent access
            con.close()
            self.logger.info("SQLite initialised at %s", db_path)
        except sqlite3.Error as exc:
            self.logger.exception("Failed to initialise SQLite database: %s", exc)
            raise

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB collection for embeddings."""
        chroma_path = Path(self.chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False),
            )
            # Get or create a dedicated collection for cat face embeddings
            self.collection = self.chroma_client.get_or_create_collection(
                name="cat_embeddings",
                metadata={"hnsw:space": "cosine"},  # use cosine distance natively
            )
            self.logger.info(
                "ChromaDB initialised at %s (collection has %d entries)",
                chroma_path,
                self.collection.count(),
            )
        except Exception as exc:
            self.logger.exception("Failed to initialise ChromaDB: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Generate embedding for image using embedding model.

        Args:
            image_path: Path to image file

        Returns:
            np.ndarray: Image embedding vector, or None on failure

        Raises:
            requests.RequestException: If embedding API fails
        """
        try:
            image_data = Path(image_path).read_bytes()
        except OSError as exc:
            self.logger.error("Cannot read image for embedding %s: %s", image_path, exc)
            return None

        # Encode image as base64 data URL (OpenAI-compatible vision embedding format)
        b64_image = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_image}"

        model_id = cfg["models"]["embedding"]["model_id"]
        payload = {
            "model": model_id,
            "input": [{"type": "image_url", "image_url": {"url": data_url}}],
        }

        try:
            response = requests.post(
                f"{self.embedding_endpoint}/embeddings",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            vector = np.array(data["data"][0]["embedding"], dtype=np.float32)
            self.logger.debug(
                "Embedding generated for %s — dim=%d", image_path.name, len(vector)
            )
            return vector
        except requests.RequestException as exc:
            self.logger.error("Embedding API request failed for %s: %s", image_path, exc)
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            self.logger.error(
                "Unexpected embedding API response for %s: %s", image_path, exc
            )
            return None

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar_cats(
        self,
        embedding: np.ndarray,
        coat_color: str,
        eye_color: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find cats with similar embeddings and matching metadata.

        Args:
            embedding: Query embedding vector
            coat_color: Coat color for metadata filtering
            eye_color: Eye color for metadata filtering
            top_k: Number of results to return

        Returns:
            List[Dict[str, Any]]: Similar cats with distances and metadata
        """
        if self.collection is None:
            self.logger.error("ChromaDB collection not initialised")
            return []

        if self.collection.count() == 0:
            self.logger.debug("ChromaDB collection is empty — no similar cats to find")
            return []

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding.tolist()],
            "n_results": min(top_k, self.collection.count()),
            "include": ["distances", "metadatas", "documents"],
        }

        # Apply hard metadata filter only when both attributes are reliably known
        if should_apply_hard_filter(coat_color, eye_color):
            query_kwargs["where"] = {
                "$and": [
                    {"coat_color": {"$eq": coat_color}},
                    {"eye_color": {"$eq": eye_color}},
                ]
            }

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as exc:
            self.logger.warning(
                "ChromaDB query failed (coat=%s, eye=%s): %s — retrying without filter",
                coat_color,
                eye_color,
                exc,
            )
            # Fall back to unfiltered query
            try:
                query_kwargs.pop("where", None)
                results = self.collection.query(**query_kwargs)
            except Exception as exc2:
                self.logger.error("ChromaDB unfiltered query also failed: %s", exc2)
                return []

        # Flatten the nested lists returned by ChromaDB
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        similar: List[Dict[str, Any]] = []
        for cat_id, dist, meta in zip(ids, distances, metadatas):
            similar.append(
                {
                    "cat_id": cat_id,
                    "distance": float(dist),
                    "metadata": meta or {},
                }
            )

        self.logger.debug(
            "find_similar_cats returned %d candidates (filter=%s)",
            len(similar),
            should_apply_hard_filter(coat_color, eye_color),
        )
        return similar

    # ------------------------------------------------------------------
    # Match classification
    # ------------------------------------------------------------------

    def classify_match(self, distance: float) -> Tuple[str, float]:
        """Classify match type based on embedding distance.

        Args:
            distance: Embedding distance value

        Returns:
            Tuple[str, float]: Match type and confidence score
        """
        known_threshold = cfg["thresholds"]["similarity_known"]
        uncertain_threshold = cfg["thresholds"]["similarity_uncertain"]

        if distance < known_threshold:
            return "known", 1.0 - distance
        elif distance < uncertain_threshold:
            return "uncertain", 0.5
        else:
            return "new", 0.1

    # ------------------------------------------------------------------
    # Profile creation / update
    # ------------------------------------------------------------------

    def create_new_cat(
        self, embedding: np.ndarray, analysis: Dict[str, Any]
    ) -> str:
        """Create new cat profile with unique ID.

        Args:
            embedding: Cat's image embedding
            analysis: Vision analysis data

        Returns:
            str: New cat UUID
        """
        cat_id = str(uuid.uuid4())
        coat_color = analysis.get("coat_color", "unknown")
        eye_color = analysis.get("eye_color", "unknown")
        description = analysis.get("description", "")
        body_condition = analysis.get("body_condition", "")

        # --- SQLite insert ---
        try:
            con = sqlite3.connect(str(self.db_path))
            with con:
                con.execute(
                    """
                    INSERT INTO cats
                        (cat_id, description, status, coat_color, eye_color,
                         last_body_condition)
                    VALUES (?, ?, 'new', ?, ?, ?)
                    """,
                    (cat_id, description, coat_color, eye_color, body_condition),
                )
            con.close()
        except sqlite3.Error as exc:
            self.logger.error(
                "SQLite insert failed for new cat %s: %s", cat_id, exc
            )
            raise

        # --- ChromaDB upsert ---
        try:
            self.collection.upsert(
                ids=[cat_id],
                embeddings=[embedding.tolist()],
                metadatas=[
                    {
                        "coat_color": coat_color,
                        "eye_color": eye_color,
                        "status": "new",
                    }
                ],
                documents=[description],
            )
        except Exception as exc:
            self.logger.error(
                "ChromaDB upsert failed for new cat %s: %s", cat_id, exc
            )
            # Do not raise — SQLite record is the source of truth; Chroma is index only

        self.logger.info("Created new cat profile: %s (coat=%s, eye=%s)", cat_id, coat_color, eye_color)
        return cat_id

    def update_cat_profile(
        self,
        cat_id: str,
        embedding: np.ndarray,
        analysis: Dict[str, Any],
    ) -> None:
        """Update existing cat profile with new visit data.

        Args:
            cat_id: Existing cat's UUID
            embedding: New image embedding to average in
            analysis: Latest vision analysis data
        """
        coat_color = analysis.get("coat_color", "unknown")
        eye_color = analysis.get("eye_color", "unknown")
        description = analysis.get("description", "")
        body_condition = analysis.get("body_condition", "")

        # --- SQLite update ---
        try:
            con = sqlite3.connect(str(self.db_path))
            with con:
                con.execute(
                    """
                    UPDATE cats
                    SET last_seen = CURRENT_TIMESTAMP,
                        visit_count = visit_count + 1,
                        last_body_condition = ?,
                        status = 'returning'
                    WHERE cat_id = ?
                    """,
                    (body_condition, cat_id),
                )
            con.close()
        except sqlite3.Error as exc:
            self.logger.error(
                "SQLite update failed for cat %s: %s", cat_id, exc
            )
            # Roll back is automatic via context manager; log and continue
            return

        # --- Blend embedding with existing ChromaDB vector ---
        try:
            existing = self.collection.get(ids=[cat_id], include=["embeddings"])
            existing_embeddings = existing.get("embeddings") or []
            if existing_embeddings:
                old_vec = np.array(existing_embeddings[0], dtype=np.float32)
                blended = average_embeddings([old_vec, embedding])
            else:
                blended = embedding

            self.collection.upsert(
                ids=[cat_id],
                embeddings=[blended.tolist()],
                metadatas=[
                    {
                        "coat_color": coat_color,
                        "eye_color": eye_color,
                        "status": "returning",
                    }
                ],
                documents=[description],
            )
        except Exception as exc:
            self.logger.warning(
                "ChromaDB update failed for cat %s: %s", cat_id, exc
            )

        self.logger.info("Updated cat profile: %s", cat_id)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def process_visit(
        self, image_path: Path, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main method to process a cat visit and determine identity.

        Args:
            image_path: Path to cat image
            analysis: Vision analysis results

        Returns:
            Dict[str, Any]: Visit processing results with identity decision
        """
        result: Dict[str, Any] = {
            "cat_id": None,
            "match_type": "new",
            "confidence": 0.1,
            "embedding_distance": None,
            "visit_id": str(uuid.uuid4()),
            "error": None,
        }

        # Step 1: Generate embedding
        embedding = self.get_embedding(image_path)
        if embedding is None:
            self.logger.warning(
                "Skipping identity resolution for %s — embedding unavailable",
                image_path,
            )
            result["error"] = "embedding_failed"
            # Still record a provisional visit with a new cat ID so no data is lost
            try:
                cat_id = self._create_cat_no_embedding(analysis)
                result["cat_id"] = cat_id
                result["match_type"] = "new"
            except Exception as exc:
                self.logger.error("Fallback cat creation also failed: %s", exc)
                result["error"] = "embedding_and_creation_failed"
            self._record_visit(result, analysis, image_path)
            return result

        coat_color = analysis.get("coat_color", "unknown")
        eye_color = analysis.get("eye_color", "unknown")

        # Step 2: Search for similar cats
        candidates = self.find_similar_cats(embedding, coat_color, eye_color)

        if candidates:
            best = candidates[0]
            distance = best["distance"]
            match_type, confidence = self.classify_match(distance)

            result["embedding_distance"] = distance
            result["match_type"] = match_type
            result["confidence"] = confidence

            if match_type == "known":
                cat_id = best["cat_id"]
                self.update_cat_profile(cat_id, embedding, analysis)
                result["cat_id"] = cat_id
                self.logger.info(
                    "Identified known cat %s (dist=%.4f, conf=%.2f)",
                    cat_id, distance, confidence,
                )
            elif match_type == "uncertain":
                # Provisional link — record but flag for review
                cat_id = best["cat_id"]
                result["cat_id"] = cat_id
                result["match_type"] = "provisional_link"
                self.logger.info(
                    "Uncertain match for cat %s (dist=%.4f) — provisional link",
                    cat_id, distance,
                )
            else:
                # New cat
                cat_id = self.create_new_cat(embedding, analysis)
                result["cat_id"] = cat_id
                self.logger.info("New cat created: %s (dist=%.4f)", cat_id, distance)
        else:
            # No candidates in DB — definitely a new cat
            cat_id = self.create_new_cat(embedding, analysis)
            result["cat_id"] = cat_id
            result["match_type"] = "new"
            self.logger.info("First cat in database: %s", cat_id)

        # Step 3: Record the visit
        self._record_visit(result, analysis, image_path)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_cat_no_embedding(self, analysis: Dict[str, Any]) -> str:
        """Create a cat record when embedding is unavailable (no ChromaDB entry)."""
        cat_id = str(uuid.uuid4())
        coat_color = analysis.get("coat_color", "unknown")
        eye_color = analysis.get("eye_color", "unknown")
        description = analysis.get("description", "")
        body_condition = analysis.get("body_condition", "")

        con = sqlite3.connect(str(self.db_path))
        try:
            with con:
                con.execute(
                    """
                    INSERT INTO cats
                        (cat_id, description, status, coat_color, eye_color,
                         last_body_condition)
                    VALUES (?, ?, 'new', ?, ?, ?)
                    """,
                    (cat_id, description, coat_color, eye_color, body_condition),
                )
        finally:
            con.close()
        return cat_id

    def _record_visit(
        self,
        result: Dict[str, Any],
        analysis: Dict[str, Any],
        image_path: Path,
    ) -> None:
        """Persist a visit row to SQLite."""
        cat_id = result.get("cat_id")
        if cat_id is None:
            self.logger.warning("Cannot record visit — no cat_id resolved")
            return

        visit_id = result.get("visit_id", str(uuid.uuid4()))
        health_flags = json.dumps(analysis.get("health_flags", []))

        try:
            con = sqlite3.connect(str(self.db_path))
            with con:
                con.execute(
                    """
                    INSERT INTO visits
                        (visit_id, cat_id, camera, clip_path, behavior,
                         body_condition, health_flags, confidence,
                         best_frame_path, embedding_distance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        visit_id,
                        cat_id,
                        analysis.get("camera", ""),
                        analysis.get("clip_path", ""),
                        analysis.get("behavior", ""),
                        analysis.get("body_condition", ""),
                        health_flags,
                        result.get("confidence", 0.0),
                        str(image_path),
                        result.get("embedding_distance"),
                    ),
                )
            con.close()
            self.logger.debug("Visit %s recorded for cat %s", visit_id, cat_id)
        except sqlite3.Error as exc:
            self.logger.error(
                "Failed to record visit %s for cat %s: %s", visit_id, cat_id, exc
            )
