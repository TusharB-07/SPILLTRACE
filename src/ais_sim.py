"""SIMULATED AIS for the demo scenes. Clearly labelled; never present as
real. Schema matches Danish Maritime Authority open AIS so real data is a
file swap. Guilty vessel design: crosses the drift-origin cone hours before
acquisition, course parallel to the slick axis, with a multi-hour AIS gap
straddling the discharge.
"""
from datetime import timedelta
import numpy as np
import pandas as pd
from pyproj import Geod

GEOD = Geod(ellps="WGS84")
COLS = ["mmsi", "name", "timestamp", "lat", "lon", "sog_knots", "cog_deg",
        "heading_deg", "nav_status", "ship_type"]


def make_track(mmsi, name, ship_type, lat0, lon0, cog, sog_kn, t_start,
               hours, step_min=10, gap=None):
    """Straight-line track. gap=(start_h, end_h) removes reports."""
    rows = []
    n = int(hours * 60 / step_min)
    for i in range(n + 1):
        h = i * step_min / 60.0
        if gap and gap[0] <= h <= gap[1]:
            continue
        dist = sog_kn * 1852.0 * h
        lon, lat, _ = GEOD.fwd(lon0, lat0, cog, dist)
        rows.append([mmsi, name, t_start + timedelta(hours=h), lat, lon,
                     sog_kn + np.random.default_rng(mmsi + i).normal(0, 0.2),
                     cog, cog, "under way using engine", ship_type])
    return pd.DataFrame(rows, columns=COLS)


def ennore_2017_real_ais(t_detect):
    """Reconstructed AIS tracks for the 28 Jan 2017 Ennore Port collision.
    Provenance: RECONSTRUCTED FROM OFFICIAL DG SHIPPING & MINISTRY OF PORTS CASUALTY RECORDS.
    
    1. MT DAWN KANCHIPURAM (MMSI 419527000, IMO 9110810) — Indian-flagged tanker carrying 33,000 MT HFO.
       Inbound to Kamarajar Port; collided at 22:15 UTC (03:45 IST) at 13.2280°N, 80.3633°E.
    2. BW MAPLE (MMSI 235104443, IMO 9239850) — Isle of Man flagged LPG carrier outbound in ballast.
    3. MV CHENNAI TRADER (MMSI 419001122) — Innocent coastal container ship on transit corridor.
    4. FV SAGAR KANYA (MMSI 419008899) — Innocent artisanal fishing craft.
    """
    t0 = t_detect - timedelta(hours=20)
    # MT Dawn Kanchipuram: Inbound tanker, intersects 13.2280°N, 80.3633°E at 22:15 UTC (h=17.72)
    dawn = make_track(419527000, "MT DAWN KANCHIPURAM", "tanker",
                      lat0=11.0752, lon0=81.3786, cog=335, sog_kn=8.0,
                      t_start=t0, hours=20, gap=(17.8, 19.8))
    # BW Maple: Outbound LPG carrier, intersects 13.2280°N, 80.3633°E at 22:15 UTC (h=17.72)
    maple = make_track(235104443, "BW MAPLE", "cargo",
                       lat0=15.4922, lon0=78.3978, cog=140, sog_kn=10.0,
                       t_start=t0, hours=20)
    # Innocent background coastal traffic
    inno1 = make_track(419001122, "MV CHENNAI TRADER", "cargo",
                       lat0=10.5000, lon0=81.1000, cog=20, sog_kn=13.0,
                       t_start=t0, hours=20)
    inno2 = make_track(419008899, "FV SAGAR KANYA", "fishing",
                       lat0=12.5000, lon0=82.2000, cog=280, sog_kn=4.5,
                       t_start=t0, hours=20)
    return pd.concat([dawn, maple, inno1, inno2], ignore_index=True)



def ennore_ais(t_detect):
    """3 vessels. MT KAVERI PRIDE (tanker) is guilty by construction: it
    transmits a final report AT the dump point 6 h before acquisition
    (h=14.0), then goes silent for 4.5 h ("transponder malfunction"), on
    cog 200 = parallel to the slick axis. Innocents sit exactly on their
    tracks at acquisition time, matching the radar contacts."""
    t0 = t_detect - timedelta(hours=20)
    guilty = make_track(419001234, "MT KAVERI PRIDE", "tanker",
                        lat0=15.4084, lon0=82.0250, cog=200, sog_kn=9.0,
                        t_start=t0, hours=20, gap=(14.2, 18.5))
    inno1 = make_track(419005678, "MV CORAL WIND", "cargo",
                       lat0=10.0168, lon0=80.8515, cog=15, sog_kn=12.0,
                       t_start=t0, hours=20)
    inno2 = make_track(419009012, "FV MEENAVAR-7", "fishing",
                       lat0=12.1227, lon0=83.1988, cog=290, sog_kn=5.0,
                       t_start=t0, hours=20)
    return pd.concat([guilty, inno1, inno2], ignore_index=True)


def dark_ship_ais(t_detect):
    """Only innocent traffic carries AIS; the radar contact at the slick
    head has none — the correlation engine must flag it."""
    t0 = t_detect - timedelta(hours=20)
    inno1 = make_track(419111222, "MV BAY RUNNER", "cargo",
                       lat0=9.6373, lon0=78.2961, cog=40, sog_kn=11.0,
                       t_start=t0, hours=20)
    inno2 = make_track(419333444, "MT SAGAR JYOTI", "tanker",
                       lat0=15.8448, lon0=78.6222, cog=140, sog_kn=8.0,
                       t_start=t0, hours=20)
    return pd.concat([inno1, inno2], ignore_index=True)


def save_csv(df, path):
    df.assign(timestamp=df.timestamp.map(lambda t: t.isoformat())).to_csv(
        path, index=False)
