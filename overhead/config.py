"""Camera configuration loader.

config.json is GITIGNORED on purpose — it holds your camera's real
location. Fresh clones copy config.example.json -> config.json and set
their own coordinates.
"""
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CFG = _REPO / "config.json"

if not _CFG.exists():
    raise SystemExit(
        "config.json not found. Copy config.example.json to config.json "
        "and set your camera's lat/lon/alt.")

_d = json.loads(_CFG.read_text())
CAM = _d["camera"]                     # lat, lon, alt_m
CAM.setdefault("alt_m", 180.0)
CAM.setdefault("hfov", 60.0)
CAM.setdefault("vfov", 35.0)
