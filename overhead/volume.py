"""Local aircraft-volume prediction — v0: funnel physics, no training data.

Instead of learning traffic patterns from weeks of history (that starts
accumulating the moment the logger runs), v0 predicts overflights from
kinematics: every live aircraft's position + velocity vector (ADS-B ground
speed and track) is dead-reckoned along a straight line; the time and
distance of closest approach to the camera tell us whether it will pass
overhead and when. Weather (wind/visibility/METAR flow) is recorded as
context — weather modulates DFW's acceptance rate and shifts corridors,
and once enough logged history exists those features feed a learned model.

Usage:
  python -m overhead.volume --nowcast    # current prediction table
  python -m overhead.volume --log        # append one observation to the log
  python -m overhead.volume --verify PRED  # score a saved nowcast later
"""
import argparse
import json
import math
import time
from pathlib import Path

from .adsb import AdsbPoller
from .geo import bearing_deg, haversine_m, project
from .weather import open_meteo, dfw_metar, flow_heuristic

CAM = {"lat": 32.8373, "lon": -97.0827, "alt_m": 180.0}
DFW = (32.8970, -97.0380)          # airport reference point
OVERFLIGHT_RADIUS_M = 4000.0       # "overhead" = within 4 km
MAX_HORIZON_S = 3600.0             # predict out to 60 minutes
DATA = Path(__file__).resolve().parent.parent / "data"


def _velocity_nmf(ac):
    """Ground velocity as (north, east) m/s from track/ground speed."""
    if not ac.get("gs_kt") or ac.get("track_deg") is None:
        return None
    v = ac["gs_kt"] * 0.514444
    brg = math.radians(ac["track_deg"])
    return v * math.cos(brg), v * math.sin(brg)


def closest_approach(ac, cam=CAM):
    """Straight-line kinematics: (eta_s, miss_m) of closest approach to camera."""
    vel = _velocity_nmf(ac)
    if vel is None:
        return None
    vn, ve = vel
    # meters north/east of camera
    dn = (ac["lat"] - cam["lat"]) * 111_320.0
    de = (ac["lon"] - cam["lon"]) * 111_320.0 * math.cos(math.radians(cam["lat"]))
    vv = vn * vn + ve * ve
    if vv < 1e-6:
        return None
    eta = -(dn * vn + de * ve) / vv          # time to closest approach
    miss = math.sqrt(max(0.0, dn * dn + de * de - (dn * vn + de * ve) ** 2 / vv))
    return eta, miss


def nowcast(poller, cam=CAM):
    """Predicted overflights in the next hour from live funnel state."""
    preds = []
    for ac in poller.live():
        if ac["on_ground"] or ac["alt_ft"] is None:
            continue
        # OpenSky never marks on_ground; low+near-airport = taxi traffic
        if ac["alt_ft"] < 500:
            continue
        if haversine_m(CAM["lat"], CAM["lon"], DFW[0], DFW[1]) and \
                haversine_m(ac["lat"], ac["lon"], DFW[0], DFW[1]) < 5000:
            continue
        ca = closest_approach(ac, cam)
        if ca is None:
            continue
        eta, miss = ca
        if 0 <= eta <= MAX_HORIZON_S and miss <= OVERFLIGHT_RADIUS_M:
            preds.append({
                "icao24": ac["icao24"], "callsign": ac["callsign"],
                "alt_ft": ac["alt_ft"], "gs_kt": ac["gs_kt"],
                "eta_min": round(eta / 60.0, 1),
                "miss_km": round(miss / 1000.0, 2),
                "dist_km": round(haversine_m(cam["lat"], cam["lon"],
                                             ac["lat"], ac["lon"]) / 1000.0, 1),
                "bearing": round(bearing_deg(cam["lat"], cam["lon"],
                                             ac["lat"], ac["lon"]), 1),
                "heading": ac["track_deg"],
            })
    preds.sort(key=lambda p: p["eta_min"])
    return preds


def horizons(preds):
    return {f"next_{m}min": sum(1 for p in preds if p["eta_min"] <= m)
            for m in (15, 30, 60)}


def snapshot(log=False):
    poller = AdsbPoller(CAM["lat"], CAM["lon"], 25.0, cadence_s=10.0).start()
    t0 = time.time()
    while not poller.aircraft and time.time() - t0 < 30:
        time.sleep(1.0)
    preds = nowcast(poller)
    try:
        om = open_meteo(CAM["lat"], CAM["lon"])
        met = dfw_metar()
        flow = flow_heuristic(met["wind_dir"])
        wx = {"open_meteo": om, "metar": met, "dfw_flow": flow}
    except Exception as e:
        wx = {"error": str(e)}
    rec = {
        "ts": time.time(),
        "in_range": len(poller.aircraft),
        "horizons": horizons(preds),
        "predicted": preds,
        "weather": wx,
        "source": poller.source,
    }
    if log:
        DATA.mkdir(exist_ok=True)
        with open(DATA / "volume_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def verify(pred_path):
    """Score a saved nowcast: of the aircraft predicted, where are they now?"""
    pred = json.loads(Path(pred_path).read_text())
    dt = time.time() - pred["ts"]
    poller = AdsbPoller(CAM["lat"], CAM["lon"], 25.0, cadence_s=10.0).start()
    t0 = time.time()
    while not poller.aircraft and time.time() - t0 < 30:
        time.sleep(1.0)
    now_by_icao = {ac["icao24"]: ac for ac in poller.live()}
    scored = []
    for p in pred["predicted"]:
        eta_s = p["eta_min"] * 60.0
        # only score predictions whose ETA has arrived
        if eta_s > dt:
            p["status"] = "not_due_yet"
            scored.append(p)
            continue
        ac = now_by_icao.get(p["icao24"])
        if ac is None:
            p["status"] = "left_feed"
        else:
            d = haversine_m(CAM["lat"], CAM["lon"], ac["lat"], ac["lon"])
            p["status"] = "overhead_now" if d <= OVERFLIGHT_RADIUS_M else f"{d/1000:.1f}km_away"
        scored.append(p)
    due = [p for p in scored if p["status"] not in ("not_due_yet", "left_feed")]
    hit = sum(1 for p in due if p["status"] == "overhead_now")
    return {"elapsed_min": round(dt / 60.0, 1), "due": len(due),
            "overhead_now": hit, "scored": scored}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nowcast", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--verify")
    ap.add_argument("--save")
    args = ap.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return
    rec = snapshot(log=args.log)
    print(f"[volume] {time.strftime('%H:%M:%S UTC', time.gmtime(rec['ts']))} "
          f"in_range={rec['in_range']} src={rec['source']}")
    print(f"[volume] horizons: {rec['horizons']}")
    wx = rec["weather"]
    if "metar" in wx:
        print(f"[volume] wx: {wx['open_meteo']['wind_kt']}kt@{wx['open_meteo']['wind_dir']} "
              f"vis={wx['open_meteo']['vis_m']/1000:.0f}km cloud={wx['open_meteo']['cloud_pct']}% "
              f"| KDFW {wx['dfw_flow']} ({wx['metar']['raw'][:40]}...)")
    for p in rec["predicted"][:12]:
        print(f"  {p['callsign'] or p['icao24']:8s} eta={p['eta_min']:5.1f}min "
              f"miss={p['miss_km']:4.2f}km alt={p['alt_ft']:5.0f}ft "
              f"from={p['bearing']:5.1f}deg hdg={p['heading']}")
    if args.save:
        Path(args.save).write_text(json.dumps(rec, indent=2))
        print(f"[volume] saved -> {args.save}")


if __name__ == "__main__":
    main()
