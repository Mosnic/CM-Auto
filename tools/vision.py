import json
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

import requests

from config import cfg

# ---------------------------------------------------------------------------
# Constants / controlled vocabulary
# ---------------------------------------------------------------------------

ANALYSIS_SCHEMA = {
    "cat_present": bool,
    "coat_color": str,       # Controlled vocabulary
    "eye_color": str,        # Controlled vocabulary
    "body_condition": str,   # poor|fair|good|excellent
    "health_flags": list,    # Array of health concerns
    "distinctive_markings": list,  # Array of notable features
    "confidence": str,       # low|medium|high
    "camera": str,           # Extracted from image overlay
}

COAT_COLORS = ["orange", "black", "tabby", "white", "calico", "gray", "unknown"]
EYE_COLORS = ["green", "blue", "yellow", "amber", "unknown"]
BODY_CONDITIONS = ["poor", "fair", "good", "excellent"]
CONFIDENCE_LEVELS = ["low", "medium", "high"]

# Default returned when analysis cannot be completed
_FAILED_ANALYSIS: Dict[str, Any] = {
    "cat_present": False,
    "coat_color": "unknown",
    "eye_color": "unknown",
    "body_condition": "fair",
    "health_flags": [],
    "distinctive_markings": [],
    "confidence": "low",
    "camera": "unknown",
}

# Prompt template for standard cat analysis
_CAT_ANALYSIS_PROMPT = (
    "Analyze this image and return ONLY a JSON object with the following fields.\n"
    "Do not include any prose or markdown fences -- raw JSON only.\n\n"
    "{\n"
    '  "cat_present": <true|false>,\n'
    '  "coat_color": "<one of: orange, black, tabby, white, calico, gray, unknown>",\n'
    '  "eye_color": "<one of: green, blue, yellow, amber, unknown>",\n'
    '  "body_condition": "<one of: poor, fair, good, excellent>",\n'
    '  "health_flags": ["<list any visible health concerns, empty list if none>"],\n'
    '  "distinctive_markings": ["<list notable physical features, empty list if none>"],\n'
    '  "confidence": "<one of: low, medium, high>",\n'
    '  "camera": "<camera label or location text visible in image overlay, or \'unknown\'>"\n'
    "}\n\n"
    "Rules:\n"
    "- Use ONLY the allowed vocabulary values listed above.\n"
    "- Set cat_present=false and all other fields to their default/unknown values if no cat is visible.\n"
    "- Estimate body_condition from visible body shape and coat quality.\n"
    "- List any wounds, discharge, unusual posture, or weight extremes in health_flags.\n"
)

# Prompt template for frame quality scoring (returns a single float string)
_QUALITY_PROMPT = (
    "Score this image for cat visibility and overall image quality.\n"
    "Return ONLY a JSON object with a single key:\n"
    '{"score": <number from 1 to 10>}\n\n'
    "Scoring guide:\n"
    "  10 -- cat fills frame, sharp focus, good lighting, clear ID features visible\n"
    "   7 -- cat clearly visible but partially obscured or slightly blurred\n"
    "   4 -- cat barely visible, heavily blurred, or very dark\n"
    "   1 -- no cat, or image is unusable\n"
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def validate_analysis_schema(data: Dict[str, Any]) -> bool:
    """Validate that analysis data matches expected schema.

    Args:
        data: Analysis data dictionary to validate

    Returns:
        bool: True if all required fields are present, False otherwise
    """
    required_fields = {
        "cat_present", "coat_color", "eye_color", "body_condition",
        "health_flags", "distinctive_markings", "confidence", "camera",
    }
    return all(field in data for field in required_fields)


def normalize_vocabulary_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize values to controlled vocabulary terms.

    Args:
        data: Raw analysis data

    Returns:
        Dict[str, Any]: Data with normalized vocabulary values
    """
    coat_mapping: Dict[str, list] = {
        "orange": ["orange", "ginger", "red", "tabby orange"],
        "black":  ["black", "dark", "solid black"],
        "tabby":  ["tabby", "striped", "gray tabby", "brown tabby"],
        "white":  ["white", "light", "cream"],
        "calico": ["calico", "tri-color", "tortoiseshell"],
        "gray":   ["gray", "grey", "silver"],
    }

    # Build reverse lookup: raw term -> canonical term
    coat_reverse: Dict[str, str] = {}
    for canonical, aliases in coat_mapping.items():
        for alias in aliases:
            coat_reverse[alias.lower()] = canonical

    normalized = data.copy()

    # Normalize coat_color
    raw_coat = str(normalized.get("coat_color", "")).lower().strip()
    normalized["coat_color"] = coat_reverse.get(raw_coat, raw_coat if raw_coat in COAT_COLORS else "unknown")

    # Ensure eye_color is within vocabulary
    raw_eye = str(normalized.get("eye_color", "")).lower().strip()
    normalized["eye_color"] = raw_eye if raw_eye in EYE_COLORS else "unknown"

    # Ensure body_condition is within vocabulary
    raw_bc = str(normalized.get("body_condition", "")).lower().strip()
    normalized["body_condition"] = raw_bc if raw_bc in BODY_CONDITIONS else "fair"

    # Ensure confidence is within vocabulary
    raw_conf = str(normalized.get("confidence", "")).lower().strip()
    normalized["confidence"] = raw_conf if raw_conf in CONFIDENCE_LEVELS else "low"

    # Ensure list fields are actually lists
    for list_field in ("health_flags", "distinctive_markings"):
        val = normalized.get(list_field, [])
        if not isinstance(val, list):
            normalized[list_field] = [str(val)] if val else []

    return normalized


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VisionTool:
    """Interface to vision model for structured cat image analysis."""

    def __init__(self) -> None:
        """Initialize vision tool with model endpoint from runtime config."""
        self.endpoint = cfg["models"]["vision"]["endpoint"]
        self.model_id = cfg["models"]["vision"]["model_id"]
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def encode_image(self, image_path: Path) -> str:
        """Encode image file as base64 string.

        Args:
            image_path: Path to image file

        Returns:
            str: Base64 encoded image data

        Raises:
            OSError: If the file cannot be read
        """
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def create_analysis_prompt(self, task_type: str = "cat_analysis") -> str:
        """Create structured prompt for cat image analysis.

        Args:
            task_type: Type of analysis to perform
                       ("cat_analysis" | "frame_quality")

        Returns:
            str: Formatted prompt for the vision model
        """
        if task_type == "frame_quality":
            return _QUALITY_PROMPT
        # Default to full cat analysis
        return _CAT_ANALYSIS_PROMPT

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate vision model response JSON.

        Args:
            response_text: Raw response text from vision model

        Returns:
            Dict[str, Any]: Parsed and validated analysis results

        Raises:
            ValueError: If response cannot be parsed or required fields missing
        """
        # Strip common markdown fences the model may emit despite the prompt
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last fence lines
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Vision model returned non-JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        return data

    def _call_api(
        self,
        prompt: str,
        b64_image: str,
        timeout: int = 60,
    ) -> str:
        """Send a single vision request; retry once on network error.

        Returns:
            str: Raw response content text

        Raises:
            requests.RequestException: After one retry
        """
        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }

        for attempt in (1, 2):
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                if attempt == 1:
                    self.logger.warning(
                        "Vision API call failed (attempt %d), retrying: %s", attempt, exc
                    )
                    time.sleep(1)
                else:
                    raise

        # Should never reach here, but keeps the type checker happy
        raise RuntimeError("Unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_path: Union[str, Path],
        task_type: str = "cat_analysis",
    ) -> Dict[str, Any]:
        """Analyze cat image using vision model.

        Args:
            image_path: Path to image file
            task_type: Type of analysis to perform

        Returns:
            Dict[str, Any]: Structured analysis results.
                            On any unrecoverable error returns a safe
                            default dict with confidence="low".

        Raises:
            requests.RequestException: If API call fails after retry
            ValueError: If response parsing fails
        """
        image_path = Path(image_path)

        # --- Encode image ---
        try:
            b64 = self.encode_image(image_path)
        except OSError as exc:
            self.logger.error("Cannot read image file %s: %s", image_path, exc)
            return _FAILED_ANALYSIS.copy()

        prompt = self.create_analysis_prompt(task_type)

        # --- Call API (raises on persistent failure) ---
        try:
            raw_text = self._call_api(prompt, b64)
        except requests.RequestException as exc:
            self.logger.error(
                "Vision API unreachable for %s: %s", image_path.name, exc
            )
            return _FAILED_ANALYSIS.copy()

        # --- Parse response ---
        try:
            data = self.parse_response(raw_text)
        except ValueError as exc:
            self.logger.error(
                "Cannot parse vision response for %s: %s | raw=%r",
                image_path.name,
                exc,
                raw_text[:200],
            )
            return _FAILED_ANALYSIS.copy()

        # --- Validate schema ---
        if not validate_analysis_schema(data):
            self.logger.warning(
                "Vision response missing required fields for %s; got keys: %s",
                image_path.name,
                list(data.keys()),
            )
            # Merge into defaults so callers always get a complete dict
            merged = _FAILED_ANALYSIS.copy()
            merged.update(data)
            data = merged

        # --- Normalize vocabulary ---
        data = normalize_vocabulary_values(data)

        self.logger.debug(
            "Analyzed %s -> cat_present=%s confidence=%s",
            image_path.name,
            data.get("cat_present"),
            data.get("confidence"),
        )
        return data

    def score_frame_quality(self, image_path: Union[str, Path]) -> float:
        """Score frame for cat visibility and image quality (1-10 scale).

        Args:
            image_path: Path to image file

        Returns:
            float: Quality score from 1.0 to 10.0.
                   Returns 1.0 on any failure so the frame is deprioritised
                   rather than crashing the pipeline.
        """
        image_path = Path(image_path)

        try:
            b64 = self.encode_image(image_path)
        except OSError as exc:
            self.logger.error(
                "Cannot read image for quality scoring %s: %s", image_path, exc
            )
            return 1.0

        try:
            raw_text = self._call_api(self.create_analysis_prompt("frame_quality"), b64)
        except requests.RequestException as exc:
            self.logger.error(
                "Vision API error during quality scoring for %s: %s",
                image_path.name,
                exc,
            )
            return 1.0

        try:
            data = self.parse_response(raw_text)
            score = float(data["score"])
            # Clamp to [1.0, 10.0]
            return max(1.0, min(10.0, score))
        except (ValueError, KeyError, TypeError) as exc:
            self.logger.warning(
                "Cannot parse quality score for %s: %s | raw=%r",
                image_path.name,
                exc,
                raw_text[:100],
            )
            return 1.0
