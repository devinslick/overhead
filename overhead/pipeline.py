#!/usr/bin/env python3
"""overhead pipeline — demo (synthetic replay) mode.

Frames (synthetic sky flying a REAL aircraft's ADS-B projection) -> YOLO
airplane detections -> IoU tracker -> live-ADS-B correlator -> overlay ->
ZMQ events -> flight log. The pipeline is blind to ground truth; success =
it independently recovers the correct callsign.

Usage: python -m overhead.pipeline --seconds 75 [--json-only]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import zmq

from .adsb import AdsbPoller, dead_reckon
from .config import CAM
from .correlator import Correlator
from .geo import project
from .sim import SyntheticReplay
from .tracker import IoUTracker

from ultralytics import YOLO

CAM = dict(CAM)                        # az auto-locks per run
W, H = 1280, 720
MODEL = sorted((Path.home() / "Work/poc-ai-stack/runs").glob("**/best.pt"))[0]


def pick_target(aircraft, cam=CAM):
    """A real arrival: close, low-but-airborne, approach-speed, high in the sky."""
    from .geo import project as _project
    best, best_el = None, -1.0
    for ac in aircraft:
        if not ac["callsign"] or ac["alt_ft"] is None or ac["on_ground"]:
            continue
        if not (800 < ac["alt_ft"] < 12000):
            continue
        if not (100 < (ac["gs_kt"] or 0) < 220):
            continue
        brg, el, dist = _project(cam["lat"], cam["lon"], cam["alt_m"],
                                 ac["lat"], ac["lon"], ac["alt_ft"] * 0.3048)
        if dist > 20000 or el < 1.5:
            continue
        if el > best_el:
            best, best_el = ac, el
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    poller = AdsbPoller(CAM["lat"], CAM["lon"], radius_nm=25.0, cadence_s=10.0).start()
    print("[pipeline] waiting for first ADS-B snapshot...")
    for _ in range(30):
        if poller.aircraft:
            break
        time.sleep(1.0)
    target = pick_target(poller.live())
    assert target, "no suitable target aircraft in range"
    CAM["az_deg"] = project(CAM["lat"], CAM["lon"], CAM["alt_m"],
                            target["lat"], target["lon"],
                            (target["alt_ft"] or 0) * 0.3048)[0]
    print(f"[pipeline] target={target['callsign']} ({target['icao24']}) "
          f"alt={target['alt_ft']}ft camera az locked={CAM['az_deg']:.1f}")

    replay = SyntheticReplay(CAM, poller, target["icao24"], W, H)
    model = YOLO(str(MODEL))
    tracker = IoUTracker(iou_gate=0.02, max_misses=90)
    corr = Correlator(CAM, W, H, gate_deg=6.0, votes_needed=4)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("ipc:///tmp/overhead_events.ipc")
    time.sleep(0.3)

    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    log_f = open(out_dir / "flight_log.jsonl", "a")
    vw = None
    if not args.json_only:
        vw = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))

    summary = {"truth": target["callsign"], "frames": 0, "detected": 0,
               "confirmed_matches": 0, "correct_callsign_frames": 0,
               "first_match_s": None, "correlated_callsigns": set()}
    t_start = time.time()
    last_event_key = None
    last_truth_callsign = target["callsign"]

    try:
        while time.time() - t_start < args.seconds:
            got = replay.next_frame()
            if got is None:
                print("[pipeline] target left FOV/feed — ending run")
                break
            frame, truth = got
            summary["frames"] += 1

            r = model.predict(frame, imgsz=640, device=0, verbose=False,
                              conf=0.4, classes=[4])[0]
            boxes = [[float(v) for v in b.xyxy[0].tolist()] for b in r.boxes]
            summary["detected"] += bool(boxes)

            live = poller.live()
            for t in tracker.step(boxes):
                pass
            # only tracks MEASURED this frame may claim a callsign —
            # coasted ghosts otherwise correlate to neighbor aircraft
            fresh = [t for t in tracker.tracks
                     if t.confirmed and t.misses == 0]
            matches = corr.step(fresh, live)

            if not args.json_only:
                # draw
                for m in matches:
                    t = next(t for t in tracker.tracks if t.id == m["track_id"])
                    x1, y1, x2, y2 = [int(v) for v in t.box]
                    color = (0, 220, 0) if m["confirmed"] else (0, 180, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = (f"{m['callsign']} {int(m['alt_ft'])}ft "
                             f"{m['dist_km']}km c={m['conf']:.2f}") if m["confirmed"] \
                        else f"TRACK-{t.id} correlating..."
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                ts = time.strftime("%H:%M:%S", time.gmtime())
                cv2.putText(frame, f"overhead demo  UTC {ts}  src={poller.source} "
                             f"aircraft_in_range={len(live)}", (10, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
                vw.write(frame)
                if summary["frames"] in (90, 900, 1800):
                    cv2.imwrite(str(out_dir / f"frame_{summary['frames']:04d}.png"), frame)

            for m in matches:
                if m["confirmed"]:
                    summary["confirmed_matches"] += 1
                    summary["correlated_callsigns"].add(m["callsign"])
                    if m["callsign"] == truth["callsign"]:
                        summary["correct_callsign_frames"] += 1
                    key = (m["track_id"], m["icao24"])
                    if key != last_event_key:
                        last_event_key = key
                        if summary["first_match_s"] is None:
                            summary["first_match_s"] = round(time.time() - t_start, 2)
                        ev = {"ts": time.time(), "event": "match_confirmed", **m}
                        log_f.write(json.dumps(ev) + "\n")
                        pub.send_json(ev)
                        print(f"[pipeline] MATCH track={m['track_id']} -> "
                              f"{m['callsign']} (truth={truth['callsign']}) "
                              f"ang_err={m['angular_err_deg']}deg")
            if truth["callsign"] != last_truth_callsign:
                last_truth_callsign = truth["callsign"]
    finally:
        if vw:
            vw.release()
        log_f.close()
        pub.close(linger=0)
        ctx.term()

    ok = last_truth_callsign in summary["correlated_callsigns"]
    summary["correlated_callsigns"] = sorted(summary["correlated_callsigns"])
    summary["validated"] = ok
    summary["match_rate"] = round(summary["confirmed_matches"] / max(1, summary["frames"]), 3)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary))




if __name__ == "__main__":
    main()
