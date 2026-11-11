"""Weather context for volume prediction — Open-Meteo + DFW METAR.

Both sources are public and keyless. The runway-flow heuristic is exactly
that (a heuristic): DFW lands into the wind, and flow direction shifts the
approach/departure corridors over a given neighborhood. The predictor lets
the data speak; the heuristic only annotates.
"""
import json
import re
import urllib.request

UA = {"User-Agent": "overhead/0.1 (personal sky-camera project)"}


def open_meteo(lat, lon):
    """Current conditions + next 4h hourly wind/visibility/cloud (UTC)."""
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&current=temperature_2m,wind_speed_10m,wind_direction_10m,"
           "cloud_cover,visibility,precipitation"
           "&forecast_hours=4&hourly=wind_speed_10m,wind_direction_10m,"
           "cloud_cover,visibility&timezone=UTC")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode())
    cur = d["current"]
    hr = d["hourly"]
    return {
        "temp_c": cur["temperature_2m"],
        "wind_kt": round(cur["wind_speed_10m"] * 1.94384, 1),
        "wind_dir": cur["wind_direction_10m"],
        "cloud_pct": cur["cloud_cover"],
        "vis_m": cur["visibility"],
        "precip_mm": cur["precipitation"],
        "forecast_hours": [
            {"time": t, "wind_kt": round(ws * 1.94384, 1), "wind_dir": wd,
             "cloud_pct": cc, "vis_m": v}
            for t, ws, wd, cc, v in zip(hr["time"], hr["wind_speed_10m"],
                                        hr["wind_direction_10m"], hr["cloud_cover"],
                                        hr["visibility"])
        ],
    }


def dfw_metar():
    """Latest DFW METAR parsed from the raw observation string."""
    url = "https://aviationweather.gov/api/data/metar?ids=KDFW&format=json"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode())[0]
    raw = d.get("rawOb", "")
    wind = re.search(r"(\d{3}|VRB)(\d{2,3})(G\d{2,3})?KT", raw)
    vis = re.search(r"(\d+)SM", raw)
    return {
        "raw": raw,
        "wind_dir": int(wind.group(1)) if wind and wind.group(1) != "VRB" else None,
        "wind_kt": int(wind.group(2)) if wind else None,
        "vis_sm": int(vis.group(1)) if vis else None,
        "obs_time": d.get("obsTime"),
    }


def flow_heuristic(wind_dir):
    """Crude DFW flow annotation: land into the wind. North flow (wind
    330-060) vs south flow (150-240); else transitions/east-west ops."""
    if wind_dir is None:
        return "unknown"
    if 330 <= wind_dir or wind_dir <= 60:
        return "north flow"
    if 150 <= wind_dir <= 240:
        return "south flow"
    return "transition/crosswind"
