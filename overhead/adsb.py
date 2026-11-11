"""Live ADS-B ingestion from public aggregators (no account, no hardware).

Sources tried in order; each normalizes to:
  {icao24, callsign, lat, lon, alt_ft, gs_kt, track_deg, seen_s, on_ground}

  - airplanes.live  https://api.airplanes.live/v2/point/{lat}/{lon}/{nm}
  - adsb.lol        https://api.adsb.lol/v2/point/{lat}/{lon}/{nm}
  - OpenSky         /api/states/all bbox (anonymous; rate-limited, fallback)

Dead-reckoning helper: project an aircraft's position forward along its
ground track for t seconds — used to interpolate between polls.
"""
import json
import math
import time
import urllib.request

UA = {"User-Agent": "overhead/0.1 (personal sky-camera project)"}


def _get(url, timeout=10):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _nm_to_deg(nm, lat):
    return nm / 60.0, nm / 60.0 / max(math.cos(math.radians(lat)), 0.01)


def fetch_point(lat, lon, radius_nm=25.0):
    """Return normalized aircraft list from the first source that answers."""
    for name, url in (
        ("airplanes.live", f"https://api.airplanes.live/v2/point/{lat}/{lon}/{int(radius_nm)}"),
        ("adsb.lol", f"https://api.adsb.lol/v2/point/{lat}/{lon}/{int(radius_nm)}"),
    ):
        try:
            d = _get(url)
            out = []
            for a in d.get("aircraft", []):
                if a.get("lat") is None or a.get("lon") is None:
                    continue
                alt = a.get("alt_baro")
                alt_ft = alt if isinstance(alt, (int, float)) else None
                out.append({
                    "icao24": a.get("hex", "?").strip(),
                    "callsign": (a.get("flight") or a.get("r") or "").strip() or None,
                    "lat": float(a["lat"]), "lon": float(a["lon"]),
                    "alt_ft": float(alt_ft) if alt_ft is not None else None,
                    "gs_kt": float(a["gs"]) if a.get("gs") else None,
                    "track_deg": float(a["track"]) if a.get("track") else None,
                    "seen_s": float(a.get("seen", 0)),
                    "on_ground": alt == "ground",
                    "source": name,
                })
            if out:
                return name, out
        except Exception as e:
            print(f"[adsb] {name} failed: {e}")
    # OpenSky fallback (anonymous, coarse cadence)
    try:
        dlat, dlon = _nm_to_deg(radius_nm, lat)
        url = (f"https://opensky-network.org/api/states/all"
               f"?lamin={lat-dlat}&lomin={lon-dlon}&lamax={lat+dlat}&lomax={lon+dlon}")
        d = _get(url)
        out = []
        for s in d.get("states", []) or []:
            if s[6] is None or s[5] is None:
                continue
            out.append({
                "icao24": s[0].strip(),
                "callsign": (s[1] or "").strip() or None,
                "lat": float(s[6]), "lon": float(s[5]),
                "alt_ft": float(s[7] or s[13] or 0),
                "gs_kt": float(s[9]) if s[9] else None,
                "track_deg": float(s[10]) if s[10] else None,
                "seen_s": float(s[3] or 0),
                "on_ground": False,
                "source": "opensky",
            })
        if out:
            return "opensky", out
    except Exception as e:
        print(f"[adsb] opensky failed: {e}")
    return None, []


def dead_reckon(ac, dt_s):
    """Advance lat/lon along ground track at ground speed for dt_s seconds."""
    if not ac.get("gs_kt") or ac.get("track_deg") is None:
        return ac["lat"], ac["lon"]
    dist_nm = ac["gs_kt"] * (dt_s / 3600.0)
    brg = math.radians(ac["track_deg"])
    dlat = (dist_nm / 60.0) * math.cos(brg)
    dlon = (dist_nm / 60.0) * math.sin(brg) / max(math.cos(math.radians(ac["lat"])), 0.01)
    return ac["lat"] + dlat, ac["lon"] + dlon


class AdsbPoller:
    """Background poller keeping the latest normalized snapshot."""

    def __init__(self, lat, lon, radius_nm=25.0, cadence_s=10.0):
        self.lat, self.lon, self.radius_nm, self.cadence_s = lat, lon, radius_nm, cadence_s
        self.aircraft = []
        self.source = None
        self.last_ok = 0.0
        self._stop = False
        self._t = None

    def start(self):
        import threading
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        import time as _t
        while not self._stop:
            try:
                src, acs = fetch_point(self.lat, self.lon, self.radius_nm)
                if src:
                    self.source, self.aircraft, self.last_ok = src, acs, _t.time()
            except Exception as e:
                print(f"[adsb] poll error: {e}")
            _t.sleep(self.cadence_s)

    def live(self, max_age_s=45.0):
        """Aircraft from the freshest snapshot, dead-reckoned to now."""
        import time as _t
        dt = _t.time() - self.last_ok if self.last_ok else 0
        if dt > max_age_s:
            return []
        out = []
        for ac in self.aircraft:
            if ac["on_ground"]:
                continue
            la, lo = dead_reckon(ac, dt)
            c = dict(ac)
            c["lat"], c["lon"] = la, lo
            out.append(c)
        return out

    def stop(self):
        self._stop = True
