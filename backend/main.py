"""
Maritime Sentinel: Oil Spill Detection & AIS Vessel Correlation System.
FastAPI Main Application and REST API Endpoints.
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List

from backend.models import ScenarioData, CulpritMatch
from backend.scenarios_data import SCENARIOS, build_scenario_alpha, build_scenario_beta, build_scenario_gamma
from backend.services.report_generator import generate_markdown_dossier
from backend.services.drift_engine import DriftEngine
from backend.services.correlation_engine import AISCorrelationEngine
from ml_engine.sar_detector import SAROilSpillDetector

app = FastAPI(
    title="Maritime Sentinel API",
    description="Oil Spill Detection via Satellite Imagery & AIS Vessel Correlation (SIH Problem #143)",
    version="1.0.0"
)

# Enable CORS for local/remote development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

drift_engine = DriftEngine()
correlation_engine = AISCorrelationEngine()
sar_detector = SAROilSpillDetector()

@app.get("/api/health")
def health_check():
    return {"status": "operational", "system": "Maritime Sentinel C2 Engine"}

@app.get("/api/stats")
def get_dashboard_stats():
    return {
        "active_satellites": ["Sentinel-1A", "Sentinel-1B", "TerraSAR-X", "RADARSAT-2"],
        "total_area_scanned_sqkm": 24580.0,
        "active_incidents_count": 2,
        "lookalikes_rejected_count": 14,
        "vessels_tracked_24h": 418,
        "peak_attribution_confidence_pct": 96.8,
        "timestamp_utc": "2026-09-01T06:00:00Z"
    }

@app.get("/api/scenarios")
def list_scenarios():
    summary_list = []
    for sc_id, sc in SCENARIOS.items():
        summary_list.append({
            "id": sc.id,
            "title": sc.title,
            "description": sc.description,
            "region_name": sc.region_name,
            "acquisition_time": sc.sar_image.acquisition_time,
            "slick_count": len(sc.slicks),
            "vessel_count": len(sc.vessels),
            "primary_verdict": sc.culprits[0].verdict if sc.culprits else "None",
            "primary_culprit": sc.culprits[0].vessel_name if sc.culprits else "None",
            "confidence": sc.culprits[0].composite_score if sc.culprits else 0.0,
            "is_lookalike": sc.slicks[0].is_lookalike if sc.slicks else False
        })
    return summary_list

@app.get("/api/scenario/{scenario_id}", response_model=ScenarioData)
def get_scenario(scenario_id: str):
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    return SCENARIOS[scenario_id]

@app.get("/api/dossier/{scenario_id}/markdown")
def get_dossier_markdown(scenario_id: str):
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    scenario = SCENARIOS[scenario_id]
    md_content = generate_markdown_dossier(scenario)
    return PlainTextResponse(content=md_content, media_type="text/markdown")

@app.get("/api/drift_simulation/{scenario_id}")
def get_drift_simulation_steps(scenario_id: str):
    """
    Returns step-by-step coordinates of the slick drift from discharge origin to detection time.
    """
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    sc = SCENARIOS[scenario_id]
    origin_lat = sc.drift_origin_cone["properties"]["origin_lat"]
    origin_lng = sc.drift_origin_cone["properties"]["origin_lng"]
    elapsed_hours = sc.drift_origin_cone["properties"]["elapsed_hours"]

    points = drift_engine.generate_drift_trajectory_points(
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        elapsed_hours=elapsed_hours,
        steps=10,
        wind_speed_ms=sc.environmental.wind_speed_ms,
        wind_direction_from_deg=sc.environmental.wind_direction_deg,
        current_speed_ms=sc.environmental.current_speed_ms,
        current_direction_to_deg=sc.environmental.current_direction_deg
    )
    return {
        "scenario_id": scenario_id,
        "elapsed_hours": elapsed_hours,
        "drift_trajectory": points
    }

# Mount static frontend files if directory exists
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_dashboard():
    index_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Maritime Sentinel API Running</h1><p>Visit /api/scenarios or create frontend/index.html</p>")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
