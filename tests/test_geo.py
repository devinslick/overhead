import math
import sys
sys.path.insert(0, ".")
from overhead.geo import (haversine_m, bearing_deg, elevation_deg,
                          angles_to_pixel, pixel_to_angles)

# Reference: Euless TX camera vs DFW airfield reference point (~32.897, -97.038)
CAM = (32.8373, -97.0827, 180.0)   # lat, lon, alt_m (approx, configurable)
DFW = (32.8970, -97.0380)

def test_distance_bearing():
    d = haversine_m(CAM[0], CAM[1], DFW[0], DFW[1])
    b = bearing_deg(CAM[0], CAM[1], DFW[0], DFW[1])
    assert 7000 < d < 9000, d      # ~8 km ENE
    assert 25 < b < 40, b          # NE-ish
    print(f"OK euless->dfw  dist={d/1000:.2f}km bearing={b:.1f}")

def test_elevation():
    # aircraft 5 km away at 3,000 ft (~914 m): atan2(914-180, 5000) ~ 8.2 deg
    ac = (32.8800, -97.0400, 914.0)
    el = elevation_deg(CAM[0], CAM[1], CAM[2], *ac)
    assert 6 < el < 10, el
    print(f"OK elevation 5km/3000ft -> {el:.1f} deg")

def test_roundtrip():
    W, H = 1280, 720
    for br, el, in [(50.0, 12.0), (30.0, 4.0), (70.0, 20.0)]:
        x, y = angles_to_pixel(br, el, W, H, az_center_deg=45, hfov_deg=60, vfov_deg=35)
        b2, e2 = pixel_to_angles(x, y, W, H, az_center_deg=45, hfov_deg=60, vfov_deg=35)
        assert abs(b2 - br) < 0.01 and abs(e2 - el) < 0.01, (br, el, b2, e2)
    print("OK pixel<->angle roundtrip")

def test_wrapping():
    # dlat +2deg north, dlon +0.2deg east across the antimeridian -> ~north
    b = bearing_deg(-1.0, 179.9, 1.0, -179.9)
    assert 0 < b < 15, b
    print(f"OK antimeridian bearing {b:.1f} (north with slight east)")

if __name__ == "__main__":
    test_distance_bearing()
    test_elevation()
    test_roundtrip()
    test_wrapping()
    print("ALL GEO TESTS PASS")
