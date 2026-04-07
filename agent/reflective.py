import json
import logging
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import requests
from config import cfg

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HEALTH_CONCERN_KEYWORDS = [
    'limping', 'injury', 'eye_discharge', 'respiratory_issue',
    'weight_loss', 'lethargy', 'poor_grooming'
]

ANALYSIS_PROMPT_TEMPLATE = """
You are analyzing cat visit data. Generate Python code to perform the following analysis:

{analysis_request}

The data will be provided as a dictionary with these keys:
- visits: List of visit records with timestamp, cat_id, body_condition, health_flags
- cats: List of cat profiles with cat_id, first_seen, last_seen, visit_count

Your code should:
1. Import only standard library modules (json, statistics, datetime)
2. Define a function called 'analyze(data)' that returns a dictionary
3. Handle missing or malformed data gracefully
4. Return meaningful insights as a structured dictionary

Write clean, documented Python code:
"""

# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def calculate_visit_frequency(visits: List[Dict[str, Any]]) -> float:
    """Calculate average visits per day for a cat.

    Args:
        visits: List of visit records

    Returns:
        float: Average visits per day
    """
    if not visits:
        return 0.0

    dates = [
        datetime.fromisoformat(v['timestamp'].replace('Z', '+00:00'))
        for v in visits
    ]
    date_range = (max(dates) - min(dates)).days

    return len(visits) / max(date_range, 1)


def detect_body_condition_trend(visits: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Analyze body condition changes over time.

    Args:
        visits: Chronologically ordered visit records

    Returns:
        Tuple[str, float]: Trend direction and confidence
    """
    conditions = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
    scores = []

    for visit in visits[-10:]:  # Last 10 visits
        if visit.get('body_condition') in conditions:
            scores.append(conditions[visit['body_condition']])

    if len(scores) < 3:
        return "insufficient_data", 0.0

    # Simple trend detection using recent vs earlier window averages
    recent_avg = statistics.mean(scores[-3:])
    earlier_avg = statistics.mean(scores[:-3])

    if recent_avg > earlier_avg + 0.5:
        return "improving", abs(recent_avg - earlier_avg) / 2.0
    elif recent_avg < earlier_avg - 0.5:
        return "declining", abs(recent_avg - earlier_avg) / 2.0
    else:
        return "stable", 0.8


def days_since_last_visit(last_seen: str) -> int:
    """Calculate days since last visit.

    Args:
        last_seen: Timestamp string of last visit

    Returns:
        int: Days since last visit
    """
    last_date = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
    # Use UTC-naive comparison; strip tzinfo for safety if present
    last_naive = last_date.replace(tzinfo=None)
    return (datetime.now() - last_naive).days


# ---------------------------------------------------------------------------
# ReflectiveAgent
# ---------------------------------------------------------------------------

class ReflectiveAgent:
    """Agent for analyzing historical patterns and trends in cat visit data."""

    def __init__(self):
        """Initialize reflective agent with database and coding model access."""
        self.db_path = cfg["paths"]["database"]
        self.coding_endpoint = cfg["models"]["coding"]["endpoint"]
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return a read-only SQLite connection with row_factory set."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _rows_to_dicts(self, rows) -> List[Dict[str, Any]]:
        """Convert sqlite3.Row objects to plain dicts."""
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_recent_visits(self, days: int = 30) -> List[Dict[str, Any]]:
        """Retrieve recent visit data from database.

        Args:
            days: Number of days of history to retrieve

        Returns:
            List[Dict[str, Any]]: Visit records with cat information
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query = """
            SELECT
                v.id,
                v.cat_id,
                v.timestamp,
                v.body_condition,
                v.health_flags,
                v.confidence,
                c.name AS cat_name,
                c.first_seen,
                c.last_seen,
                c.visit_count
            FROM visits v
            LEFT JOIN cats c ON c.cat_id = v.cat_id
            WHERE v.timestamp >= ?
            ORDER BY v.timestamp ASC
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (cutoff,)).fetchall()
            records = self._rows_to_dicts(rows)
            # Deserialize health_flags JSON string if stored that way
            for rec in records:
                if isinstance(rec.get('health_flags'), str):
                    try:
                        rec['health_flags'] = json.loads(rec['health_flags'])
                    except (json.JSONDecodeError, TypeError):
                        rec['health_flags'] = []
            self.logger.debug("Retrieved %d visit records for last %d days", len(records), days)
            return records
        except sqlite3.Error as exc:
            self.logger.error("Database error in get_recent_visits: %s", exc)
            return []

    def _get_cat_visits(self, cat_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Fetch visit records for a single cat."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query = """
            SELECT id, cat_id, timestamp, body_condition, health_flags, confidence
            FROM visits
            WHERE cat_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (cat_id, cutoff)).fetchall()
            records = self._rows_to_dicts(rows)
            for rec in records:
                if isinstance(rec.get('health_flags'), str):
                    try:
                        rec['health_flags'] = json.loads(rec['health_flags'])
                    except (json.JSONDecodeError, TypeError):
                        rec['health_flags'] = []
            return records
        except sqlite3.Error as exc:
            self.logger.error("Database error fetching visits for cat %s: %s", cat_id, exc)
            return []

    def _get_all_cat_profiles(self) -> List[Dict[str, Any]]:
        """Return all cat profile rows."""
        query = "SELECT cat_id, name, first_seen, last_seen, visit_count FROM cats"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query).fetchall()
            return self._rows_to_dicts(rows)
        except sqlite3.Error as exc:
            self.logger.error("Database error fetching cat profiles: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def detect_health_trends(self, cat_id: str) -> Dict[str, Any]:
        """Analyze health trends for a specific cat.

        Args:
            cat_id: Cat UUID to analyze

        Returns:
            Dict[str, Any]: Health trend analysis results
        """
        visits = self._get_cat_visits(cat_id)
        if not visits:
            self.logger.info("No visit data available for cat %s", cat_id)
            return {"cat_id": cat_id, "status": "no_data"}

        trend_direction, trend_confidence = detect_body_condition_trend(visits)

        # Aggregate health concern flags across all visits
        concern_counts: Dict[str, int] = {}
        for visit in visits:
            flags = visit.get('health_flags') or []
            if isinstance(flags, list):
                for flag in flags:
                    if flag in HEALTH_CONCERN_KEYWORDS:
                        concern_counts[flag] = concern_counts.get(flag, 0) + 1

        # Flag concerns that appear in more than 20% of visits
        persistent_concerns = [
            k for k, v in concern_counts.items()
            if v / len(visits) >= 0.2
        ]

        visit_freq = calculate_visit_frequency(visits)

        result = {
            "cat_id": cat_id,
            "visit_count_analyzed": len(visits),
            "body_condition_trend": trend_direction,
            "trend_confidence": round(trend_confidence, 3),
            "visit_frequency_per_day": round(visit_freq, 3),
            "health_concern_counts": concern_counts,
            "persistent_concerns": persistent_concerns,
            "requires_attention": bool(persistent_concerns) or trend_direction == "declining",
        }
        self.logger.info(
            "Health trend for cat %s: %s (conf=%.2f), concerns=%s",
            cat_id, trend_direction, trend_confidence, persistent_concerns
        )
        return result

    def detect_absence_patterns(self) -> List[Dict[str, Any]]:
        """Identify cats that have stopped visiting recently.

        Returns:
            List[Dict[str, Any]]: Cats with unusual absence patterns
        """
        cats = self._get_all_cat_profiles()
        if not cats:
            return []

        # Build per-cat median inter-visit gap (last 30 days) for comparison
        all_recent = self.get_recent_visits(days=30)
        visits_by_cat: Dict[str, List[str]] = {}
        for v in all_recent:
            visits_by_cat.setdefault(v['cat_id'], []).append(v['timestamp'])

        absent_cats = []
        for cat in cats:
            cat_id = cat['cat_id']
            last_seen = cat.get('last_seen')
            if not last_seen:
                continue

            absence_days = days_since_last_visit(last_seen)
            cat_visits = visits_by_cat.get(cat_id, [])

            # Compute expected gap from recent visit history
            if len(cat_visits) >= 2:
                sorted_ts = sorted(
                    datetime.fromisoformat(t.replace('Z', '+00:00'))
                    for t in cat_visits
                )
                gaps = [
                    (sorted_ts[i + 1] - sorted_ts[i]).days
                    for i in range(len(sorted_ts) - 1)
                ]
                median_gap = statistics.median(gaps) if gaps else 7
                # Flag if absence is more than 3x the median gap (min threshold 7 days)
                threshold = max(median_gap * 3, 7)
            else:
                # Default: flag cats unseen for > 14 days with any prior history
                threshold = 14

            if absence_days > threshold and cat.get('visit_count', 0) > 0:
                absent_cats.append({
                    "cat_id": cat_id,
                    "cat_name": cat.get('name', 'unknown'),
                    "last_seen": last_seen,
                    "absence_days": absence_days,
                    "expected_gap_days": round(threshold, 1),
                    "visit_count": cat.get('visit_count', 0),
                })
                self.logger.warning(
                    "Absence anomaly: cat %s (%s) last seen %d days ago (threshold %.1f)",
                    cat_id, cat.get('name', 'unknown'), absence_days, threshold
                )

        return absent_cats

    def generate_analysis_code(self, prompt: str) -> str:
        """Generate Python analysis code using coding model.

        Args:
            prompt: Description of analysis needed

        Returns:
            str: Generated Python code
        """
        full_prompt = ANALYSIS_PROMPT_TEMPLATE.format(analysis_request=prompt)

        # Ollama-compatible generate endpoint
        url = f"{self.coding_endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": cfg["models"]["coding"]["model_id"],
            "prompt": full_prompt,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            code = data.get("response", "")
            # Strip markdown fences if the model wrapped the code
            if "```python" in code:
                code = code.split("```python", 1)[1].rsplit("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].rsplit("```", 1)[0]
            return code.strip()
        except requests.RequestException as exc:
            self.logger.error("Coding model request failed: %s", exc)
            return ""
        except (KeyError, ValueError) as exc:
            self.logger.error("Unexpected coding model response format: %s", exc)
            return ""

    def execute_analysis_code(self, code: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute generated analysis code.

        The generated code must define a function called ``analyze(data)``
        that accepts a dict and returns a dict.  Execution uses a highly
        restricted namespace -- only ``json``, ``statistics``, and
        ``datetime`` are injected to limit risk.

        Args:
            code: Python code to execute
            data: Input data for analysis

        Returns:
            Dict[str, Any]: Analysis results
        """
        if not code:
            return {"error": "empty_code"}

        import datetime as _dt  # local alias for sandbox namespace

        # Restricted builtins -- prevents file I/O, network, imports, etc.
        safe_builtins = {
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter,
            "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "round": round, "int": int, "float": float,
            "str": str, "bool": bool, "list": list, "dict": dict,
            "tuple": tuple, "set": set,
            "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
            "print": lambda *a, **kw: None,  # swallow prints silently
            "None": None, "True": True, "False": False,
        }
        sandbox_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "json": json,
            "statistics": statistics,
            "datetime": _dt,
        }

        try:
            exec(compile(code, "<generated>", "exec"), sandbox_globals)  # noqa: S102
        except SyntaxError as exc:
            self.logger.error("Syntax error in generated code: %s", exc)
            return {"error": f"syntax_error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error during code compilation/exec: %s", exc)
            return {"error": f"exec_error: {exc}"}

        analyze_fn = sandbox_globals.get("analyze")
        if not callable(analyze_fn):
            self.logger.error("Generated code does not define callable 'analyze'")
            return {"error": "missing_analyze_function"}

        try:
            result = analyze_fn(data)
            if not isinstance(result, dict):
                self.logger.warning("analyze() returned non-dict type %s", type(result))
                return {"result": result}
            return result
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Runtime error in generated analyze(): %s", exc, exc_info=True)
            return {"error": f"runtime_error: {exc}"}

    def analyze_population_dynamics(self) -> Dict[str, Any]:
        """Analyze overall population changes and dynamics.

        Returns:
            Dict[str, Any]: Population analysis results
        """
        cats = self._get_all_cat_profiles()
        if not cats:
            return {"status": "no_data"}

        now = datetime.now()

        # Split cats into cohorts by first-seen recency
        new_cats = []      # first seen in last 30 days
        regular_cats = []  # seen within last 30 days but older
        lost_cats = []     # not seen in > 30 days

        for cat in cats:
            first_seen_str = cat.get('first_seen') or ''
            last_seen_str = cat.get('last_seen') or ''
            if not last_seen_str:
                continue

            absence = days_since_last_visit(last_seen_str)

            if first_seen_str:
                first_seen = datetime.fromisoformat(
                    first_seen_str.replace('Z', '+00:00')
                ).replace(tzinfo=None)
                age_days = (now - first_seen).days
            else:
                age_days = 999

            if age_days <= 30 and absence <= 30:
                new_cats.append(cat)
            elif absence <= 30:
                regular_cats.append(cat)
            else:
                lost_cats.append(cat)

        # Compute total visit activity over last 7 / 30 days
        recent_7 = self.get_recent_visits(days=7)
        recent_30 = self.get_recent_visits(days=30)

        unique_7 = len({v['cat_id'] for v in recent_7})
        unique_30 = len({v['cat_id'] for v in recent_30})

        result = {
            "total_known_cats": len(cats),
            "new_cats_last_30_days": len(new_cats),
            "regular_active_cats": len(regular_cats),
            "inactive_cats": len(lost_cats),
            "unique_cats_last_7_days": unique_7,
            "unique_cats_last_30_days": unique_30,
            "total_visits_last_7_days": len(recent_7),
            "total_visits_last_30_days": len(recent_30),
            "avg_visits_per_active_cat_30d": (
                round(len(recent_30) / unique_30, 2) if unique_30 else 0.0
            ),
        }
        self.logger.info("Population dynamics: %s", result)
        return result

    def detect_statistical_anomalies(self) -> List[Dict[str, Any]]:
        """Find statistical anomalies in visit patterns.

        Uses z-score on per-cat visit frequency and body condition scores
        to surface outliers worth attention.

        Returns:
            List[Dict[str, Any]]: Detected anomalies with details
        """
        recent = self.get_recent_visits(days=30)
        if not recent:
            return []

        # Group visits by cat
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for v in recent:
            by_cat.setdefault(v['cat_id'], []).append(v)

        if len(by_cat) < 2:
            # Need at least 2 cats for meaningful comparison
            return []

        freqs = {
            cat_id: calculate_visit_frequency(visits)
            for cat_id, visits in by_cat.items()
        }

        freq_values = list(freqs.values())
        try:
            mean_freq = statistics.mean(freq_values)
            stdev_freq = statistics.stdev(freq_values) if len(freq_values) > 1 else 0.0
        except statistics.StatisticsError:
            mean_freq, stdev_freq = 0.0, 0.0

        anomalies = []
        for cat_id, freq in freqs.items():
            z = (freq - mean_freq) / stdev_freq if stdev_freq > 0 else 0.0
            visits = by_cat[cat_id]

            issues = []

            # Frequency anomaly (z > 2 or z < -2)
            if abs(z) >= 2.0:
                issues.append({
                    "type": "frequency_outlier",
                    "z_score": round(z, 2),
                    "value": round(freq, 3),
                    "population_mean": round(mean_freq, 3),
                })

            # Body condition decline spike
            trend, conf = detect_body_condition_trend(visits)
            if trend == "declining" and conf > 0.5:
                issues.append({
                    "type": "declining_body_condition",
                    "confidence": round(conf, 2),
                })

            # Persistent health flags
            flag_counts: Dict[str, int] = {}
            for v in visits:
                for flag in (v.get('health_flags') or []):
                    if flag in HEALTH_CONCERN_KEYWORDS:
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1
            concerning = {k: cnt for k, cnt in flag_counts.items() if cnt / len(visits) >= 0.3}
            if concerning:
                issues.append({
                    "type": "persistent_health_flags",
                    "flags": concerning,
                })

            if issues:
                cat_name = visits[0].get('cat_name', 'unknown') if visits else 'unknown'
                anomalies.append({
                    "cat_id": cat_id,
                    "cat_name": cat_name,
                    "issues": issues,
                    "visit_count_30d": len(visits),
                })
                self.logger.warning(
                    "Anomaly detected for cat %s (%s): %s",
                    cat_id, cat_name, [i['type'] for i in issues]
                )

        return anomalies

    async def run_reflection_cycle(self) -> Dict[str, Any]:
        """Run complete reflective analysis cycle.

        Orchestrates all analysis methods and optionally uses the coding
        model to generate supplementary statistical code where useful.

        Returns:
            Dict[str, Any]: Complete reflection results
        """
        self.logger.info("Starting reflection cycle")
        results: Dict[str, Any] = {
            "cycle_timestamp": datetime.now().isoformat(),
            "population_dynamics": {},
            "absence_alerts": [],
            "health_trends": {},
            "statistical_anomalies": [],
            "llm_analysis": {},
            "errors": [],
        }

        # 1. Population overview
        try:
            results["population_dynamics"] = self.analyze_population_dynamics()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Population dynamics failed: %s", exc, exc_info=True)
            results["errors"].append(f"population_dynamics: {exc}")

        # 2. Absence detection
        try:
            results["absence_alerts"] = self.detect_absence_patterns()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Absence pattern detection failed: %s", exc, exc_info=True)
            results["errors"].append(f"absence_patterns: {exc}")

        # 3. Health trends per known cat
        cats = self._get_all_cat_profiles()
        for cat in cats:
            cat_id = cat['cat_id']
            try:
                results["health_trends"][cat_id] = self.detect_health_trends(cat_id)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Health trend failed for cat %s: %s", cat_id, exc)
                results["errors"].append(f"health_trend_{cat_id}: {exc}")

        # 4. Statistical anomalies
        try:
            results["statistical_anomalies"] = self.detect_statistical_anomalies()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Statistical anomaly detection failed: %s", exc, exc_info=True)
            results["errors"].append(f"statistical_anomalies: {exc}")

        # 5. Optional LLM-generated supplementary analysis
        recent_visits = self.get_recent_visits(days=30)
        if recent_visits and cats:
            analysis_data = {
                "visits": recent_visits[:200],  # cap to avoid huge payloads
                "cats": cats,
            }
            analysis_prompt = (
                "Identify any notable patterns in visit timing (time-of-day clustering, "
                "weekly rhythms) and summarize key findings as a dict with keys: "
                "'timing_patterns', 'weekly_rhythm', 'summary'."
            )
            try:
                code = self.generate_analysis_code(analysis_prompt)
                if code:
                    llm_result = self.execute_analysis_code(code, analysis_data)
                    results["llm_analysis"] = llm_result
                else:
                    self.logger.info("Coding model returned empty code; skipping LLM analysis")
                    results["llm_analysis"] = {"status": "skipped_empty_code"}
            except Exception as exc:  # noqa: BLE001
                self.logger.error("LLM analysis step failed: %s", exc, exc_info=True)
                results["llm_analysis"] = {"status": "error", "detail": str(exc)}
                results["errors"].append(f"llm_analysis: {exc}")

        self.logger.info(
            "Reflection cycle complete -- %d absence alerts, %d anomalies, %d errors",
            len(results["absence_alerts"]),
            len(results["statistical_anomalies"]),
            len(results["errors"]),
        )
        return results
