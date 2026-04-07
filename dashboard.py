import json
import logging
import jinja2
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from database import CatDatabase
from tools.cat_identity import CatIdentityEngine
from agent.reflective import ReflectiveAgent
from config import cfg

DASHBOARD_PORT = cfg["dashboard"]["port"]
DASHBOARD_HOST = cfg["dashboard"]["host"]
UPLOADS_DIR = Path(cfg["paths"]["uploads"])
DATA_DIR = Path(cfg["paths"]["data"])
HTML_TEMPLATE_DIR = "templates"
STATIC_FILES_DIR = "static"

HOME_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Cat Monitor Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .cat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .cat-card { border-left: 4px solid #4CAF50; }
        .alert-card { border-left: 4px solid #f44336; }
        .stats { display: flex; justify-content: space-around; text-align: center; }
        .stat { margin: 10px; }
        .stat-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        img { max-width: 200px; border-radius: 4px; }
        button { padding: 8px 16px; margin: 4px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #2196F3; color: white; }
        .btn-danger { background: #f44336; color: white; }
    </style>
</head>
<body>
    <h1>🐱 Cat Monitor Dashboard</h1>

    <div class="card">
        <h2>System Statistics</h2>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{{ stats.total_cats }}</div>
                <div>Total Cats</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ stats.active_cats }}</div>
                <div>Active Cats</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ stats.recent_visits }}</div>
                <div>Recent Visits</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ stats.active_alerts }}</div>
                <div>Active Alerts</div>
            </div>
        </div>
    </div>

    {% if alerts %}
    <div class="card alert-card">
        <h2>🚨 Active Alerts</h2>
        {% for alert in alerts %}
        <div style="margin: 10px 0; padding: 10px; background: #fff3cd; border-radius: 4px;">
            <strong>{{ alert.alert_type }}:</strong> {{ alert.message }}
            <small style="float: right;">{{ alert.created_at }}</small>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="card">
        <h2>Cat Profiles</h2>
        <div class="cat-grid">
            {% for cat in cats %}
            <div class="card cat-card">
                <h3>{{ cat.description or "Unknown Cat" }}</h3>
                <p><strong>Status:</strong> {{ cat.status }}</p>
                <p><strong>Visits:</strong> {{ cat.visit_count }}</p>
                <p><strong>Last Seen:</strong> {{ cat.last_seen }}</p>
                <p><strong>Coat:</strong> {{ cat.coat_color }}</p>
                <a href="/cat/{{ cat.cat_id }}">View Details</a>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>"""

CAT_PROFILE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Cat Profile - {{ cat.description or cat.cat_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: left; }
        th { background: #f0f0f0; }
        img { max-width: 150px; border-radius: 4px; }
        button { padding: 8px 16px; margin: 4px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #2196F3; color: white; }
        .btn-danger { background: #f44336; color: white; }
        input, select { padding: 6px; margin: 4px; border: 1px solid #ddd; border-radius: 4px; }
        .nav-link { display: inline-block; margin-bottom: 20px; color: #2196F3; text-decoration: none; }
    </style>
</head>
<body>
    <a class="nav-link" href="/">&larr; Back to Dashboard</a>
    <h1>Cat Profile: {{ cat.description or "Unknown Cat" }}</h1>

    <div class="card">
        <h2>Profile Details</h2>
        <table>
            <tr><th>Cat ID</th><td>{{ cat.cat_id }}</td></tr>
            <tr><th>Description</th><td>{{ cat.description or "N/A" }}</td></tr>
            <tr><th>Status</th><td>{{ cat.status or "unknown" }}</td></tr>
            <tr><th>Coat Color</th><td>{{ cat.coat_color or "N/A" }}</td></tr>
            <tr><th>Visit Count</th><td>{{ cat.visit_count or 0 }}</td></tr>
            <tr><th>Last Seen</th><td>{{ cat.last_seen or "N/A" }}</td></tr>
            <tr><th>First Seen</th><td>{{ cat.first_seen or "N/A" }}</td></tr>
        </table>
    </div>

    {% if analysis %}
    <div class="card">
        <h2>Historical Analysis</h2>
        <pre style="white-space: pre-wrap; background: #f9f9f9; padding: 10px; border-radius: 4px;">{{ analysis }}</pre>
    </div>
    {% endif %}

    <div class="card">
        <h2>Visit History</h2>
        {% if visits %}
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Confidence</th>
                <th>Camera</th>
                <th>Thumbnail</th>
                <th>Correct Identity</th>
            </tr>
            {% for visit in visits %}
            <tr>
                <td>{{ visit.timestamp or "N/A" }}</td>
                <td>{{ "%.2f"|format(visit.confidence|float) if visit.confidence is defined and visit.confidence is not none else "N/A" }}</td>
                <td>{{ visit.camera or "unknown" }}</td>
                <td>
                    {% if visit.thumbnail %}
                    <img src="/image/{{ visit.thumbnail }}" alt="Visit thumbnail" />
                    {% else %}
                    No image
                    {% endif %}
                </td>
                <td>
                    <form method="post" action="/api/correct-identity" style="display:inline;">
                        <input type="hidden" name="visit_id" value="{{ visit.visit_id }}" />
                        <input type="text" name="correct_cat_id" placeholder="Cat ID or 'new'" />
                        <button type="submit" class="btn-primary">Correct</button>
                    </form>
                </td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No visits recorded for this cat.</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>Merge Cat Profiles</h2>
        <p>Merge another cat's visits into this profile ({{ cat.cat_id }}) and remove the other profile.</p>
        <form method="post" action="/api/merge-cats">
            <input type="hidden" name="cat_id_1" value="{{ cat.cat_id }}" />
            <label>Cat ID to merge from (will be removed):</label>
            <input type="text" name="cat_id_2" placeholder="Other Cat ID" required />
            <button type="submit" class="btn-danger">Merge</button>
        </form>
    </div>
</body>
</html>"""

VISITS_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Recent Visits - Cat Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: left; }
        th { background: #f0f0f0; }
        img { max-width: 100px; border-radius: 4px; }
        select, button { padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #2196F3; color: white; border: none; cursor: pointer; }
        .nav-link { display: inline-block; margin-bottom: 20px; color: #2196F3; text-decoration: none; }
    </style>
</head>
<body>
    <a class="nav-link" href="/">&larr; Back to Dashboard</a>
    <h1>Recent Visits</h1>

    <div class="card">
        <form method="get" action="/visits" style="margin-bottom: 10px;">
            <label>Show visits from last:
                <select name="days">
                    <option value="7" {% if days == 7 %}selected{% endif %}>7 days</option>
                    <option value="14" {% if days == 14 %}selected{% endif %}>14 days</option>
                    <option value="30" {% if days == 30 %}selected{% endif %}>30 days</option>
                </select>
            </label>
            <button type="submit">Filter</button>
        </form>

        {% if visits %}
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Cat</th>
                <th>Confidence</th>
                <th>Camera</th>
                <th>Thumbnail</th>
            </tr>
            {% for visit in visits %}
            <tr>
                <td>{{ visit.timestamp or "N/A" }}</td>
                <td>
                    {% if visit.cat_id %}
                    <a href="/cat/{{ visit.cat_id }}">{{ visit.cat_description or visit.cat_id }}</a>
                    {% else %}
                    Unknown
                    {% endif %}
                </td>
                <td>{{ "%.2f"|format(visit.confidence|float) if visit.confidence is defined and visit.confidence is not none else "N/A" }}</td>
                <td>{{ visit.camera or "unknown" }}</td>
                <td>
                    {% if visit.thumbnail %}
                    <img src="/image/{{ visit.thumbnail }}" alt="Visit thumbnail" />
                    {% else %}
                    No image
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No visits found for the selected time range.</p>
        {% endif %}
    </div>
</body>
</html>"""

ALERTS_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Alerts - Cat Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .alert-item { margin: 10px 0; padding: 15px; background: #fff3cd; border-radius: 4px; border-left: 4px solid #f44336; }
        .alert-item.resolved { background: #e8f5e9; border-left-color: #4CAF50; }
        button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; background: #f44336; color: white; }
        .nav-link { display: inline-block; margin-bottom: 20px; color: #2196F3; text-decoration: none; }
        small { color: #666; }
    </style>
</head>
<body>
    <a class="nav-link" href="/">&larr; Back to Dashboard</a>
    <h1>Active Alerts</h1>

    <div class="card">
        {% if alerts %}
        {% for alert in alerts %}
        <div class="alert-item">
            <strong>{{ alert.alert_type or "Alert" }}</strong>: {{ alert.message or "No message" }}
            <br/>
            {% if alert.cat_id %}
            <small>Cat: <a href="/cat/{{ alert.cat_id }}">{{ alert.cat_id }}</a></small><br/>
            {% endif %}
            <small>Created: {{ alert.created_at or "N/A" }}</small>
            <div style="margin-top: 8px;">
                <span style="color: #888; font-size: 0.85em;">Alert ID: {{ alert.alert_id or "N/A" }}</span>
            </div>
        </div>
        {% endfor %}
        {% else %}
        <p>No active alerts.</p>
        {% endif %}
    </div>
</body>
</html>"""


class Dashboard:
    def __init__(self):
        self.app = FastAPI(title="Cat Monitor Dashboard")
        self.db = CatDatabase()
        self.identity_engine = CatIdentityEngine()
        self.reflective_agent = ReflectiveAgent()
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
        self._setup_static_files()

    def _setup_static_files(self) -> None:
        if UPLOADS_DIR.exists():
            try:
                self.app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
            except Exception as e:
                self.logger.warning(f"Could not mount uploads dir {UPLOADS_DIR}: {e}")
        else:
            self.logger.warning(f"Uploads directory does not exist: {UPLOADS_DIR}")

        if DATA_DIR.exists():
            try:
                self.app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
            except Exception as e:
                self.logger.warning(f"Could not mount data dir {DATA_DIR}: {e}")
        else:
            self.logger.warning(f"Data directory does not exist: {DATA_DIR}")

        static_path = Path(STATIC_FILES_DIR)
        if static_path.exists():
            try:
                self.app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
            except Exception as e:
                self.logger.warning(f"Could not mount static dir {STATIC_FILES_DIR}: {e}")

    def _setup_routes(self) -> None:
        self.app.add_api_route("/", self.home, methods=["GET"])
        self.app.add_api_route("/cat/{cat_id}", self.cat_profile, methods=["GET"])
        self.app.add_api_route("/visits", self.recent_visits, methods=["GET"])
        self.app.add_api_route("/alerts", self.alerts_page, methods=["GET"])
        self.app.add_api_route("/api/correct-identity", self.correct_identity, methods=["POST"])
        self.app.add_api_route("/api/merge-cats", self.merge_cats, methods=["POST"])
        self.app.add_api_route("/api/stats", self.api_stats, methods=["GET"])
        self.app.add_api_route("/image/{image_path:path}", self.serve_image, methods=["GET"])

    async def home(self, request: Request) -> HTMLResponse:
        try:
            cats = self.db.get_all_cats() or []
            alerts = self.db.get_active_alerts() or []
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            end_date = datetime.utcnow().isoformat()
            recent_visits = self.db.get_visits_by_date_range(start_date, end_date) or []

            active_cats = sum(1 for c in cats if c.get("status") == "active")
            stats = {
                "total_cats": len(cats),
                "active_cats": active_cats,
                "recent_visits": len(recent_visits),
                "active_alerts": len(alerts),
            }

            env = jinja2.Environment(autoescape=True)
            template = env.from_string(HOME_TEMPLATE)
            html = template.render(stats=stats, alerts=alerts, cats=cats)
            return HTMLResponse(content=html)
        except Exception as e:
            self.logger.error(f"Error rendering home page: {e}")
            return HTMLResponse(content=f"<h1>Error</h1><p>{e}</p>", status_code=500)

    async def cat_profile(self, request: Request, cat_id: str) -> HTMLResponse:
        try:
            cat = self.db.get_cat_profile(cat_id)
            if not cat:
                raise HTTPException(status_code=404, detail=f"Cat not found: {cat_id}")

            visits = self.db.get_visits_by_cat(cat_id) or []

            analysis = None
            try:
                analysis = self.reflective_agent.analyze_cat_history(cat_id)
            except Exception as ae:
                self.logger.warning(f"Could not get analysis for cat {cat_id}: {ae}")

            env = jinja2.Environment(autoescape=True)
            template = env.from_string(CAT_PROFILE_TEMPLATE)
            html = template.render(cat=cat, visits=visits, analysis=analysis)
            return HTMLResponse(content=html)
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error rendering cat profile for {cat_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def recent_visits(self, request: Request, days: int = 7) -> HTMLResponse:
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            end_date = datetime.utcnow()
            visits = self.db.get_visits_by_date_range(start_date.isoformat(), end_date.isoformat()) or []

            env = jinja2.Environment(autoescape=True)
            template = env.from_string(VISITS_TEMPLATE)
            html = template.render(visits=visits, days=days)
            return HTMLResponse(content=html)
        except Exception as e:
            self.logger.error(f"Error rendering recent visits page: {e}")
            return HTMLResponse(content=f"<h1>Error</h1><p>{e}</p>", status_code=500)

    async def alerts_page(self, request: Request) -> HTMLResponse:
        try:
            alerts = self.db.get_active_alerts() or []

            env = jinja2.Environment(autoescape=True)
            template = env.from_string(ALERTS_TEMPLATE)
            html = template.render(alerts=alerts)
            return HTMLResponse(content=html)
        except Exception as e:
            self.logger.error(f"Error rendering alerts page: {e}")
            return HTMLResponse(content=f"<h1>Error</h1><p>{e}</p>", status_code=500)

    async def correct_identity(
        self,
        visit_id: str = Form(...),
        correct_cat_id: str = Form(...),
    ) -> JSONResponse:
        try:
            self.logger.info(f"Identity correction attempt: visit_id={visit_id}, correct_cat_id={correct_cat_id}")

            visit = self.db.get_visit(visit_id)
            if not visit:
                return JSONResponse({"success": False, "error": "Visit not found"}, status_code=404)

            if correct_cat_id == "new":
                new_cat = self.identity_engine.create_new_cat_profile(description="Human-corrected new cat")
                correct_cat_id = new_cat.get("cat_id", correct_cat_id)
            else:
                existing = self.db.get_cat_profile(correct_cat_id)
                if not existing:
                    return JSONResponse(
                        {"success": False, "error": f"Cat not found: {correct_cat_id}"},
                        status_code=404,
                    )

            self.db.update_visit_cat(visit_id, correct_cat_id)
            self.logger.info(f"Successfully corrected visit {visit_id} to cat {correct_cat_id}")
            return JSONResponse({"success": True, "visit_id": visit_id, "new_cat_id": correct_cat_id})
        except Exception as e:
            self.logger.error(f"Error correcting identity for visit {visit_id}: {e}")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def merge_cats(
        self,
        cat_id_1: str = Form(...),
        cat_id_2: str = Form(...),
    ) -> JSONResponse:
        try:
            self.logger.info(f"Merge attempt: cat_id_1={cat_id_1}, cat_id_2={cat_id_2}")

            cat1 = self.db.get_cat_profile(cat_id_1)
            if not cat1:
                return JSONResponse({"success": False, "error": f"Cat not found: {cat_id_1}"}, status_code=404)

            cat2 = self.db.get_cat_profile(cat_id_2)
            if not cat2:
                return JSONResponse({"success": False, "error": f"Cat not found: {cat_id_2}"}, status_code=404)

            visits = self.db.get_visits_by_cat(cat_id_2) or []
            count = 0
            for visit in visits:
                visit_id = visit.get("visit_id")
                if visit_id:
                    self.db.update_visit_cat(visit_id, cat_id_1)
                    count += 1

            if hasattr(self.identity_engine, "average_embeddings"):
                try:
                    self.identity_engine.average_embeddings(cat_id_1, cat_id_2)
                except Exception as ee:
                    self.logger.warning(f"Could not average embeddings for merge: {ee}")

            self.db.delete_cat_profile(cat_id_2)
            self.logger.info(
                f"Successfully merged cat {cat_id_2} into {cat_id_1}, reassigned {count} visits"
            )
            return JSONResponse(
                {
                    "success": True,
                    "merged_into": cat_id_1,
                    "merged_from": cat_id_2,
                    "visits_reassigned": count,
                }
            )
        except Exception as e:
            self.logger.error(f"Error merging cats {cat_id_1} and {cat_id_2}: {e}")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def api_stats(self) -> JSONResponse:
        try:
            cats = self.db.get_all_cats() or []
            active_cats = sum(1 for c in cats if c.get("status") == "active")

            all_visits = self.db.get_visits_by_date_range(
                "1970-01-01T00:00:00", datetime.utcnow().isoformat()
            ) or []

            start_7d = (datetime.utcnow() - timedelta(days=7)).isoformat()
            start_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            now_iso = datetime.utcnow().isoformat()

            visits_7d = self.db.get_visits_by_date_range(start_7d, now_iso) or []
            visits_24h = self.db.get_visits_by_date_range(start_24h, now_iso) or []
            alerts = self.db.get_active_alerts() or []

            stats = {
                "total_cats": len(cats),
                "active_cats": active_cats,
                "total_visits": len(all_visits),
                "recent_visits_7d": len(visits_7d),
                "recent_visits_24h": len(visits_24h),
                "active_alerts": len(alerts),
                "last_updated": datetime.utcnow().isoformat(),
            }
            return JSONResponse(stats)
        except Exception as e:
            self.logger.error(f"Error fetching api stats: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def serve_image(self, image_path: str) -> FileResponse:
        try:
            uploads_resolved = UPLOADS_DIR.resolve()
            full_path = (UPLOADS_DIR / image_path).resolve()

            try:
                full_path.relative_to(uploads_resolved)
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied")

            if full_path.exists():
                return FileResponse(str(full_path))

            data_resolved = DATA_DIR.resolve()
            data_path = (DATA_DIR / image_path).resolve()
            try:
                data_path.relative_to(data_resolved)
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied")

            if data_path.exists():
                return FileResponse(str(data_path))

            raise HTTPException(status_code=404, detail="Image not found")
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error serving image {image_path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 8002) -> None:
        uvicorn.run(self.app, host=host, port=port)


def create_dashboard_app() -> FastAPI:
    dashboard = Dashboard()
    return dashboard.app


def format_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return timestamp


def calculate_visit_stats(visits: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not visits:
        return {"total": 0, "avg_confidence": 0, "cameras": []}
    total = len(visits)
    avg_confidence = sum(v.get('confidence', 0) for v in visits) / total
    cameras = list(set(v.get('camera', 'unknown') for v in visits))
    return {
        "total": total,
        "avg_confidence": round(avg_confidence, 2),
        "cameras": cameras,
        "date_range": {
            "first": visits[-1].get('timestamp', ''),
            "last": visits[0].get('timestamp', '')
        }
    }


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT)
