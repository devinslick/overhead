# overhead

A camera, a detector, and the public ADS-B network — answering *"what plane
is that?"* in real time.

Point a camera at the sky. YOLO detects aircraft, a tracker assigns stable
track IDs, and each track is correlated against live public ADS-B
transponder data (OpenSky / airplanes.live / adsb.lol) to overlay the
callsign, altitude, speed, and range of the flight you're looking at.
Every match is published as a ZeroMQ event and appended to a flight log.

Built and validated under the DFW arrival corridor — one of the densest
patches of civilian sky anywhere, with 100+ aircraft transmitting within
25 nm at any moment.

## How it works

    frames (camera / synthetic replay)
        -> YOLOv8 airplane detections (PyTorch, ~10 ms/frame on RTX 2080)
        -> IoU tracker (constant-velocity coasting over dropouts)
        -> correlator: track pixel -> (bearing, elevation) via camera FOV
                       vs live ADS-B positions -> (bearing, elevation)
                       greedy nearest-match, angular gate, vote + sticky
        -> overlay (callsign / altitude / range) + ZMQ events + flight log

Geodesy: haversine + great-circle bearing + elevation from camera
(lat/lon/alt). Dead-reckoning advances aircraft between polls along
ground track at ground speed.

## Validation (synthetic-replay mode)

The demo source flies a sprite along a **real live aircraft's** projected
path — ground truth comes from the live feed, but the pipeline never sees
it. It must independently detect, track, and correlate. If the overlay
recovers the correct callsign, the whole loop is proven.

Results (60 s runs, live DFW traffic, ~100-117 aircraft in range):

| Run | Truth | Detection | Correct callsign | First lock | Angular err |
|---|---|---|---|---|---|
| 2026-11-11 | JSX210 | 99.1% | 3882/3882 (100%) | 1.8 s | 0.95° |
| 2026-11-11 | SWA1665 | 94.2% | 2919/2919 (100%) | 1.8 s | 1.06° |

Earlier runs (before correlation tuning) showed the real problem: label
flip-flops between aircraft 2-3° apart in bearing — in this sky that's
common. Fixes that got to 100%: tighter angular gate (4°), elevation
weighted equally with bearing, 6-vote confirmation, sticky incumbency
(a confirmed match must be beaten by 2° to change hands), and
fresh-measurement gating (coasted ghost tracks may not claim callsigns).

Honest open problems (see roadmap): aircraft that turn hard between ADS-B
polls (straight-line dead reckoning diverges); multiple aircraft within
~1-2° of bearing; apparent-size-vs-range gating is unimplemented.

## Repo layout

    overhead/geo.py         haversine, bearing, elevation, pixel<->angle
    overhead/adsb.py        live ADS-B poller (3 public sources, fallback)
    overhead/tracker.py     IoU tracker w/ velocity coasting
    overhead/correlator.py  angular matching, votes, stickiness
    overhead/sim.py         synthetic-replay validation source
    overhead/pipeline.py    end-to-end demo (this is what you run)
    tests/                  geodesy + tracker unit tests
    out/                    demo.mp4, overlay frames, flight_log.jsonl

## Run

    # deps: torch+cu, ultralytics, opencv, pyzmq (see requirements.txt)
    python -m overhead.pipeline --seconds 60

The run picks the best live target (close, approach-speed, high in the
sky), auto-locks camera azimuth, and validates against it. Outputs land
in out/: an overlay video, sample frames, a JSONL flight log, and
summary.json (validated: true/false + stats).

## Real deployment

Swap the synthetic source for an RTSP camera, then calibrate once:
camera lat/lon/alt (GPS of the mount), azimuth (compass), and horizontal/
vertical FOV (lens spec or a two-reference-point calibration). Everything
downstream is unchanged. TODO: rtsp source module, camera calibration
helper, always-on service + daily flight-log summaries.

## Data & legality

ADS-B is public by design (that's the point of the system). No account,
key, or special access required for the sources used. Detector weights
fine-tuned on coco128 (public COCO subset). Nothing here touches
restricted data.

## License / status

Personal project by Devin Slick. v0.1 validated end-to-end in
synthetic-replay mode on 2026-11-11; camera-mode bring-up next.
