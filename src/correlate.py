"""Drift backtracking + AIS spacetime correlation. This module is the
project's actual contribution — everything upstream is commodity.

AIS here is SIMULATED and labelled as such (see ais_sim.py). The scoring
code consumes the standard schema
  mmsi,timestamp,lat,lon,sog_knots,cog_deg,heading_deg,nav_status,ship_type
so real Danish-DMA / Indian AIS drops in unchanged.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

# Environmental forcing for the demo window (NE-monsoon-ish: drift toward SW).
# SWAP-IN POINT: replace with ERA5 wind + OSCAR/CMEMS surface current lookups.
WIND_U, WIND_V = -3.0, -4.0     # m/s (east, north)
CUR_U, CUR_V = -0.10, -0.15     # m/s
LEEWAY = 0.03                    # oil drifts at ~3% of wind + current


def backtrack_cone(lat, lon, hours_range=range(1, 25)):
    """Rewind one slick point over a range of hours -> list of
    (lat_origin, lon_origin, hours_back)."""
    u = CUR_U + LEEWAY * WIND_U
    v = CUR_V + LEEWAY * WIND_V
    out = []
    for h in hours_range:
        dx, dy = -u * h * 3600.0, -v * h * 3600.0
        dist = float(np.hypot(dx, dy))
        brg = float(np.degrees(np.arctan2(dx, dy)) % 360)
        lon2, lat2, _ = GEOD.fwd(lon, lat, brg, dist)
        out.append((lat2, lon2, h))
    return out


def ang_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def axis_diff(a, b):
    """Difference between a course (0-360) and an undirected axis (0-180)."""
    d = ang_diff(a % 180, b % 180)
    return min(d, 180 - d)


def score_vessel(track: pd.DataFrame, origins, t_detect, axis_bearing,
                 window_h=24):
    t0 = t_detect - timedelta(hours=window_h)
    seg = track[(track.timestamp >= t0) & (track.timestamp <= t_detect)]
    if seg.empty:
        return None
    seg = seg.sort_values("timestamp").reset_index(drop=True)

    best_km, best_row = 1e9, None
    for lat_o, lon_o, hb in origins:
        t_star = t_detect - timedelta(hours=hb)
        idx = (seg.timestamp - t_star).abs().idxmin()
        r = seg.loc[idx]
        _, _, d = GEOD.inv(lon_o, lat_o, float(r.lon), float(r.lat))
        radius = 8.0 + 1.0 * hb          # drift uncertainty grows with time
        if (d / 1000.0) / radius < best_km:
            best_km, best_row, best_radius = (d / 1000.0) / radius, r, radius
    if best_row is None:
        return None

    best_km, s_space = best_km * best_radius, max(0.0, 1.0 - best_km)
    course_gap = axis_diff(float(best_row.cog_deg), axis_bearing)
    s_head = max(0.0, 1.0 - course_gap / 90.0)

    gaps = seg.timestamp.diff().dt.total_seconds().div(3600).fillna(0)
    max_gap_h = float(gaps.max())
    s_gap = min(1.0, max_gap_h / 6.0)

    stype = str(best_row.get("ship_type", "")).lower()
    s_type = 1.0 if stype in {"tanker", "cargo", "bulk"} else 0.5

    conf = 0.40 * s_space + 0.25 * s_head + 0.20 * s_gap + 0.15 * s_type
    ev = [f"Closest approach {best_km:.1f} km to backtracked origin cone",
          f"Course within {course_gap:.0f}\u00b0 of slick major axis"]
    ev.append(f"AIS silent for {max_gap_h:.1f} h inside the window"
              if max_gap_h > 1.0 else "AIS continuous throughout window")
    ev.append(f"Vessel type: {stype or 'unknown'}")

    return {"mmsi": int(best_row.mmsi),
            "name": str(best_row.get("name", f"MMSI {int(best_row.mmsi)}")),
            "ship_type": stype,
            "confidence": round(float(conf), 3),
            "closest_approach_km": round(best_km, 2),
            "course_vs_axis_deg": round(course_gap, 1),
            "max_ais_gap_h": round(max_gap_h, 2),
            "position_time": best_row.timestamp.isoformat(),
            "evidence": ev}


def rank_suspects(ais: pd.DataFrame, slick_centroid, axis_bearing, t_detect,
                  radar_ships=None, match_km=5.0):
    origins = backtrack_cone(*slick_centroid)
    out = []
    for mmsi, track in ais.groupby("mmsi"):
        s = score_vessel(track, origins, t_detect, axis_bearing)
        if s:
            out.append(s)
    out.sort(key=lambda d: -d["confidence"])

    dark = []
    if radar_ships:
        for sh in radar_ships:
            near_ais = False
            for _, r in ais.iterrows():
                _, _, d = GEOD.inv(sh["lon"], sh["lat"], float(r.lon), float(r.lat))
                if d / 1000.0 < match_km:
                    near_ais = True
                    break
            if not near_ais:
                in_cone = any(
                    GEOD.inv(sh["lon"], sh["lat"], lo, la)[2] / 1000.0 < 15.0
                    for la, lo, _ in origins)
                dark.append({"lon": sh["lon"], "lat": sh["lat"],
                             "inside_origin_cone": bool(in_cone),
                             "note": "Radar contact with NO AIS transponder"
                                     + (" — inside drift-origin cone: PRIME "
                                        "SUSPECT" if in_cone else "")})
    return out, dark, [(la, lo) for la, lo, _ in origins], origins


def validate_ground_truth(origins, truth_dict):
    """Calculates geodesic distance from backtrack cone to official ground-truth collision coordinates."""
    if not truth_dict or "lat" not in truth_dict or "lon" not in truth_dict:
        return None
    t_lat, t_lon = float(truth_dict["lat"]), float(truth_dict["lon"])
    best_km, best_h = 1e9, None
    for la, lo, h in origins:
        _, _, d = GEOD.inv(t_lon, t_lat, lo, la)
        d_km = d / 1000.0
        if d_km < best_km:
            best_km, best_h = d_km, h
    return {
        "ground_truth_lat": t_lat,
        "ground_truth_lon": t_lon,
        "error_km": round(best_km, 2),
        "predicted_discharge_hours_prior": best_h,
        "location_name": truth_dict.get("location_name", "Official Casualty Anchor")
    }


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)

