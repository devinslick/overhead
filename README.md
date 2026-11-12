# overhead

A camera, a detector, and the public ADS-B network — answering *"what plane
is that?"* in real time.

Repo: https://github.com/devinslick/opensky — named for the OpenSky Network,
whose public feed makes projects like this possible. This project is
independent and unaffiliated with The OpenSky Network; ADS-B data credited
to OpenSky and its contributors.

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
    cp config.example.json config.json   # then set YOUR camera lat/lon/alt
    python -m overhead.pipeline --seconds 60

(config.json is gitignored — your location stays yours.)

The run picks the best live target (close, approach-speed, high in the
sky), auto-locks camera azimuth, and validates against it. Outputs land
in out/: an overlay video, sample frames, a JSONL flight log, and
summary.json (validated: true/false + stats).

## Volume prediction (v0 — funnel physics)

`python -m overhead.volume --nowcast` predicts overflights for the next
hour from live kinematics: each aircraft's position + velocity vector is
dead-reckoned straight-line; time/distance of closest approach to the
camera yields an ETA table. Weather context comes from Open-Meteo (keyless)
and the DFW METAR (wind -> crude north/south flow annotation, which shifts
which corridors cross a given neighborhood).

Live sample (2026-11-11 22:50 UTC, 105 aircraft in range): 5 predicted
overflights within 15 min, 7 within 30, 9 within 60 — callsigns, ETAs,
miss distances.

Closed-loop verification (7.5 min later): 0/2 due predictions overhead at
sample time — one left the feed (coverage gap), one turned to land
(terminal-area maneuvering defeats straight-line kinematics), and the
verifier sampled after ETA rather than at it. Honest v0 limits, known and
documented:

- terminal-area turns break straight-line dead reckoning (biggest error source)
- single-sample verification misses flyovers between samples
- coverage gaps make "left feed" ambiguous

Fix ladder (in order): ETA-windowed verification from the continuous
logger; turn detection (track-rate gate) to mark predictions low-confidence
near the airport; then, once weeks of `--log` history exist, a learned
model on [hour, day-of-week, weather, live funnel state] — the log schema
already captures all of it.

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
