"""Serves the cached pipeline results + the demo UI. Fully offline.
Run from the project root:  uvicorn api.main:app --port 8000
Open http://localhost:8000
"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="SIH26143 Oil Spill Attribution")


@app.get("/api/scenes")
def scenes():
    out = []
    for sid in json.load(open(ROOT / "data/cache/scenes.json")):
        meta = json.load(open(ROOT / f"data/scenes/{sid}.json"))
        out.append({"id": sid, "title": meta["title"], "acq_time": meta["acq_time"]})
    return out


@app.get("/api/analyze/{scene_id}")
def analyze(scene_id: str):
    p = ROOT / f"data/cache/{scene_id}_result.json"
    if not p.exists():
        raise HTTPException(404, "unknown scene")
    return json.load(open(p))


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


app.mount("/scenes", StaticFiles(directory=ROOT / "data/scenes"), name="scenes")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
