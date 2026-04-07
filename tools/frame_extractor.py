import asyncio
import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import aiofiles

from config import cfg

# ---------------------------------------------------------------------------
# Module-level constants derived from config — evaluated once at import time
# ---------------------------------------------------------------------------
MIN_SCORE_THRESHOLD: float = cfg["processing"]["min_score_threshold"]
TOP_FRAMES_COUNT: int = cfg["processing"]["top_frames_count"]
MIN_FRAME_SPACING: float = cfg["thresholds"]["frame_spacing_seconds"]
FRAME_WIDTH: int = cfg["processing"]["frame_width"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def parse_timestamp_from_filename(frame_path: Path) -> float:
    """Extract timestamp from FFmpeg frame filename.

    FFmpeg outputs frames as frame_0001.jpg, frame_0002.jpg, etc.
    At 1 fps extraction the frame number equals the elapsed seconds.

    Args:
        frame_path: Path to frame file (stem format: frame_NNNN)

    Returns:
        float: Timestamp in seconds, or 0.0 if the pattern is not found.
    """
    match = re.search(r'frame_(\d+)', frame_path.stem)
    return float(match.group(1)) if match else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FrameExtractor:
    """Extracts and scores frames from video clips to find the best cat-visibility frames."""

    def __init__(self) -> None:
        """Initialise the extractor with a shared VisionTool instance."""
        from tools.vision import VisionTool  # deferred to avoid circular/missing-module errors at import time
        self.vision_tool = VisionTool()
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    async def extract_best_frames(self, video_path: Path) -> List[Tuple[Path, float, dict]]:
        """Main entry point: extract frames, score each one, return the best.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of (frame_path, score, analysis) tuples for the best frames,
            sorted descending by score.  Returns an empty list if no scorable
            frames are found.
        """
        # Use a temporary directory so all frames are cleaned up automatically,
        # even if scoring raises an exception.
        with tempfile.TemporaryDirectory(prefix="catmonitor_frames_") as tmp_dir:
            output_dir = Path(tmp_dir)
            try:
                frames = await self.extract_frames_ffmpeg(video_path, output_dir)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                self.logger.error(
                    "FFmpeg extraction failed for %s: %s", video_path, exc
                )
                raise

            if not frames:
                self.logger.warning("No frames extracted from %s", video_path)
                return []

            # Score every frame concurrently for throughput.
            score_tasks = [self._safe_score_frame(f) for f in frames]
            scored_raw: List[Optional[Tuple[Path, float, dict]]] = await asyncio.gather(
                *score_tasks
            )

            # Filter out frames that completely failed scoring.
            frame_scores: List[Tuple[Path, float, dict]] = [
                r for r in scored_raw if r is not None
            ]

            if not frame_scores:
                self.logger.warning("All frames failed scoring for %s", video_path)
                return []

            best = await self.select_best_frames(frame_scores)
            self.logger.info(
                "Selected %d best frame(s) from %s (pool size: %d)",
                len(best), video_path, len(frame_scores),
            )
            return best
        # tempfile.TemporaryDirectory.__exit__ deletes tmp_dir here.
        # Callers that need persistent copies should move / copy the files
        # before this coroutine returns — or persist inside VisionTool.

    # ------------------------------------------------------------------
    # Step 1 — extraction
    # ------------------------------------------------------------------

    async def extract_frames_ffmpeg(
        self, video_path: Path, output_dir: Path
    ) -> List[Path]:
        """Extract frames from *video_path* using FFmpeg at 1 fps, resized to
        *FRAME_WIDTH* pixels wide.

        Runs FFmpeg in a subprocess so it does not block the event loop.

        Args:
            video_path: Path to the input video file.
            output_dir:  Directory in which to write extracted JPEG frames.

        Returns:
            Sorted list of Paths to extracted frame files.

        Raises:
            FileNotFoundError: If FFmpeg is not installed.
            subprocess.CalledProcessError: If FFmpeg exits with a non-zero status.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(output_dir / "frame_%04d.jpg")

        # -vf scale sets width to FRAME_WIDTH, height auto-scaled (-1) to
        # preserve aspect ratio.  fps=1 keeps one frame per second.
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps=1,scale={FRAME_WIDTH}:-1",
            "-q:v", "2",         # JPEG quality 2 = high quality, low artefacts
            "-y",                # overwrite without prompting
            output_pattern,
        ]

        self.logger.debug("Running FFmpeg: %s", " ".join(cmd))

        loop = asyncio.get_event_loop()
        try:
            # Run the blocking subprocess in a thread pool so we don't stall
            # other coroutines during potentially long video processing.
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            )
        except subprocess.CalledProcessError as exc:
            self.logger.error(
                "FFmpeg failed (exit %d) for %s.\nstderr: %s",
                exc.returncode, video_path, exc.stderr,
            )
            raise
        except FileNotFoundError:
            self.logger.critical(
                "FFmpeg not found — ensure it is installed and on PATH."
            )
            raise

        frames = sorted(output_dir.glob("frame_*.jpg"))
        self.logger.debug("Extracted %d frame(s) from %s", len(frames), video_path)
        return frames

    # ------------------------------------------------------------------
    # Step 2 — scoring
    # ------------------------------------------------------------------

    async def score_frame(self, frame_path: Path) -> Tuple[float, dict]:
        """Score a single frame for cat visibility, sharpness, angle, and lighting.

        Delegates to *VisionTool.analyze_frame* and extracts / derives a
        numeric score from the returned analysis dict.

        Args:
            frame_path: Path to the JPEG frame to evaluate.

        Returns:
            Tuple of (score, analysis) where score is in [0, 10] and
            analysis is the raw dict returned by the vision model.
        """
        analysis: dict = await self.vision_tool.analyze_frame(frame_path)

        # Prefer an explicit numeric score if the model returned one.
        if "score" in analysis and isinstance(analysis["score"], (int, float)):
            score = float(analysis["score"])
        else:
            # Derive a heuristic score from qualitative fields so we always
            # have a sortable numeric value even when the model omits a score.
            score = self._derive_score(analysis)

        # Clamp to the valid 0-10 range.
        score = max(0.0, min(10.0, score))
        self.logger.debug("Frame %s scored %.2f", frame_path.name, score)
        return score, analysis

    # ------------------------------------------------------------------
    # Step 3 — selection with temporal spacing
    # ------------------------------------------------------------------

    async def select_best_frames(
        self, frame_scores: List[Tuple[Path, float, dict]]
    ) -> List[Tuple[Path, float, dict]]:
        """Select up to TOP_FRAMES_COUNT frames that exceed MIN_SCORE_THRESHOLD
        and are spaced at least MIN_FRAME_SPACING seconds apart.

        The greedy algorithm picks the highest-scoring frame first, then
        iterates through remaining candidates in score order, accepting each
        only if it is far enough in time from every already-accepted frame.

        Args:
            frame_scores: List of (path, score, analysis) tuples (any order).

        Returns:
            Selected frames in descending score order.
        """
        # Apply the minimum quality gate.
        candidates = [
            (p, s, a) for p, s, a in frame_scores if s >= MIN_SCORE_THRESHOLD
        ]

        if not candidates:
            self.logger.info(
                "No frames met MIN_SCORE_THRESHOLD (%.1f); returning empty list.",
                MIN_SCORE_THRESHOLD,
            )
            return []

        # Sort highest score first.
        candidates.sort(key=lambda t: t[1], reverse=True)

        selected: List[Tuple[Path, float, dict]] = []
        for frame_path, score, analysis in candidates:
            if len(selected) >= TOP_FRAMES_COUNT:
                break

            ts = parse_timestamp_from_filename(frame_path)
            # Reject if too close to an already-accepted frame.
            too_close = any(
                abs(ts - parse_timestamp_from_filename(sp)) < MIN_FRAME_SPACING
                for sp, _, _ in selected
            )
            if not too_close:
                selected.append((frame_path, score, analysis))
                self.logger.debug(
                    "Selected frame %s (score=%.2f, t=%.1fs)", frame_path.name, score, ts
                )

        return selected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _safe_score_frame(
        self, frame_path: Path
    ) -> Optional[Tuple[Path, float, dict]]:
        """Wrap score_frame so a failure on a single frame is non-fatal.

        Returns None if an exception is raised, so gather() can collect all
        results without short-circuiting.
        """
        try:
            score, analysis = await self.score_frame(frame_path)
            return frame_path, score, analysis
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Scoring failed for frame %s, assigning score 0: %s",
                frame_path.name, exc,
            )
            return frame_path, 0.0, {}

    @staticmethod
    def _derive_score(analysis: dict) -> float:
        """Heuristically map qualitative vision-model fields to a numeric score.

        Used when the model does not return an explicit 'score' key.

        Scoring rubric (max 10):
          - cat_present=True   → +4
          - confidence         → low=+1  medium=+2  high=+3
          - body_condition     → poor=+0 fair=+1 good=+2 excellent=+3
        """
        score = 0.0

        if analysis.get("cat_present") is True:
            score += 4.0

        confidence_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
        score += confidence_map.get(str(analysis.get("confidence", "")).lower(), 0.0)

        condition_map = {"poor": 0.0, "fair": 1.0, "good": 2.0, "excellent": 3.0}
        score += condition_map.get(
            str(analysis.get("body_condition", "")).lower(), 0.0
        )

        return score


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

async def extract_frames_from_video(video_path: str) -> List[dict]:
    """Extract and score frames from *video_path*, returning the best ones.

    This is the primary entry point for the agent loop.

    Args:
        video_path: Absolute (or resolvable) path to the video file.

    Returns:
        List of dicts, each containing:
          - "frame_path"  (str)
          - "score"       (float, 0–10)
          - "analysis"    (dict, raw vision-model output)
    """
    extractor = FrameExtractor()
    results = await extractor.extract_best_frames(Path(video_path))

    return [
        {
            "frame_path": str(frame_path),
            "score": score,
            "analysis": analysis,
        }
        for frame_path, score, analysis in results
    ]


# ---------------------------------------------------------------------------
# API_CONTRACT
# Exports:
# - FrameExtractor: Class for extracting and scoring video frames
# - extract_frames_from_video: Async convenience function
# - parse_timestamp_from_filename: Utility for timestamp parsing
# ---------------------------------------------------------------------------
