"""overhead — camera + YOLO + public ADS-B = what's that plane overhead?

Geodesy helpers: camera (lat/lon/alt) -> aircraft bearing/elevation, and the
inverse mapping for rendering.
"""
import math

R_EARTH_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_M * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point1 to point2. 0=N, clockwise."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def elevation_deg(cam_lat, cam_lon, cam_alt_m, ac_lat, ac_lon, ac_alt_m):
    """Angle above horizon from camera to aircraft (degrees; negative = below)."""
    ground = haversine_m(cam_lat, cam_lon, ac_lat, ac_lon)
    rise = ac_alt_m - cam_alt_m
    return math.degrees(math.atan2(rise, max(ground, 1.0)))


def project(cam_lat, cam_lon, cam_alt_m, ac_lat, ac_lon, ac_alt_m):
    """(bearing_deg, elevation_deg, ground_distance_m) from camera to aircraft."""
    return (bearing_deg(cam_lat, cam_lon, ac_lat, ac_lon),
            elevation_deg(cam_lat, cam_lon, cam_alt_m, ac_lat, ac_lon, ac_alt_m),
            haversine_m(cam_lat, cam_lon, ac_lat, ac_lon))


# --- camera-frame mapping (linear small-angle approx; pinhole would be exact) ---

def angles_to_pixel(bearing, elevation, width, height, az_center_deg, hfov_deg, vfov_deg):
    """Map a bearing/elevation to pixel coords given camera center azimuth/FOV."""
    db = _ang_diff(bearing, az_center_deg)
    de = elevation
    x = width * (0.5 + db / hfov_deg)
    y = height * (0.5 - de / vfov_deg)
    return x, y


def pixel_to_angles(x, y, width, height, az_center_deg, hfov_deg, vfov_deg):
    """Inverse of angles_to_pixel: pixel -> (bearing, elevation)."""
    db = (x / width - 0.5) * hfov_deg
    de = (0.5 - y / height) * vfov_deg
    return (az_center_deg + db) % 360.0, de


def _ang_diff(a, b):
    """Smallest signed difference a-b in degrees, wrapped to [-180, 180]."""
    return (a - b + 540.0) % 360.0 - 180.0
