"""Correlate visual tracks with ADS-B aircraft by angular position.

Each confirmed track's pixel center -> (bearing, elevation) via the camera
config; each live aircraft -> (bearing, elevation) via geodesy. Greedy
nearest-neighbor matching inside an angular gate, with consecutive-vote
stabilization so labels don't flicker between adjacent aircraft.
"""
from .geo import pixel_to_angles, project, _ang_diff


class Correlator:
    def __init__(self, cam, width, height, gate_deg=4.0, votes_needed=6,
                 sticky_bonus_deg=2.0, el_weight=1.0):
        self.cam = cam                    # dict: lat, lon, alt_m, az_deg, hfov, vfov
        self.width, self.height = width, height
        self.gate_deg = gate_deg
        self.votes_needed = votes_needed
        self.sticky_bonus_deg = sticky_bonus_deg
        self.el_weight = el_weight

    def step(self, tracks, aircraft):
        """tracks: list of Track; aircraft: normalized ADS-B list. Returns matches."""
        # camera-frame angles for each visual track
        t_angles = []
        for t in tracks:
            cx, cy = t.center
            brg, el = pixel_to_angles(cx, cy, self.width, self.height,
                                      self.cam["az_deg"], self.cam["hfov"], self.cam["vfov"])
            t_angles.append((t, brg, el))

        # sky angles for each aircraft
        a_angles = []
        for ac in aircraft:
            if ac["alt_ft"] is None:
                continue
            brg, el, dist = project(self.cam["lat"], self.cam["lon"], self.cam["alt_m"],
                                    ac["lat"], ac["lon"], ac["alt_ft"] * 0.3048)
            a_angles.append((ac, brg, el, dist))

        # greedy nearest matching on angular distance, with stickiness: an
        # incumbent confirmed match must be beaten by `sticky_bonus_deg`
        pairs = []
        for ti, (t, tb, te) in enumerate(t_angles):
            for ai, (ac, ab, ae, dist) in enumerate(a_angles):
                d = abs(_ang_diff(tb, ab)) + self.el_weight * abs(te - ae)
                if t.match_votes and max(t.match_votes.values(), default=0) >= self.votes_needed \
                        and ac["icao24"] in t.match_votes:
                    d -= self.sticky_bonus_deg   # incumbent advantage
                if d < self.gate_deg:
                    pairs.append((d, ti, ai))
        pairs.sort(key=lambda p: p[0])
        used_t, used_a, matches = set(), set(), []
        for d, ti, ai in pairs:
            if ti in used_t or ai in used_a:
                continue
            used_t.add(ti); used_a.add(ai)
            t, tb, te = t_angles[ti]
            ac, ab, ae, dist = a_angles[ai]
            # report RAW angular error (d may include the sticky bonus)
            raw = abs(_ang_diff(tb, ab)) + self.el_weight * abs(te - ae)
            conf = round(max(0.0, min(1.0, 1.0 - raw / self.gate_deg)), 3)
            t.match_votes = {ac["icao24"]: t.match_votes.get(ac["icao24"], 0) + 1}
            confirmed = (t.match_votes[ac["icao24"]] >= self.votes_needed)
            if confirmed:
                t.callsign = ac["callsign"] or ac["icao24"].upper()
            matches.append({
                "track_id": t.id, "bearing_est": round(tb, 1), "elevation_est": round(te, 1),
                "icao24": ac["icao24"], "callsign": ac["callsign"], "alt_ft": ac["alt_ft"],
                "gs_kt": ac["gs_kt"], "dist_km": round(dist / 1000, 1),
                "angular_err_deg": round(raw, 2), "conf": conf, "confirmed": confirmed,
            })
        return matches
