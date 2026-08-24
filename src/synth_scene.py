"""Generate 4 synthetic Sentinel-1-style scenes off Ennore/Chennai.

Output per scene: data/scenes/<id>.png (8-bit SAR-look image) +
<id>.json (georeference bounds, acquisition time, truth metadata).

SWAP-IN POINT FOR REAL DATA: replace these PNGs with real Sentinel-1
sigma0 8-bit GeoTIFF-derived PNGs (see to_db.py in the plan doc) and
update the bounds in the JSON. Everything downstream is unchanged.
"""
import json
import numpy as np
from PIL import Image

RNG = np.random.default_rng(42)
H = W = 800
# Scene footprint off Ennore/Chennai coast
BOUNDS = {"lon_min": 79.90, "lon_max": 82.40, "lat_min": 12.00, "lat_max": 14.50}  # ~270 km, IW-swath scale


def px_to_lonlat(col, row):
    lon = BOUNDS["lon_min"] + (col / W) * (BOUNDS["lon_max"] - BOUNDS["lon_min"])
    lat = BOUNDS["lat_max"] - (row / H) * (BOUNDS["lat_max"] - BOUNDS["lat_min"])
    return lon, lat


def speckle_sea(mean=0.45):
    """Multiplicative gamma speckle over a gently varying sea backscatter."""
    yy, xx = np.mgrid[0:H, 0:W]
    texture = 1.0 + 0.08 * np.sin(yy / 90.0) * np.cos(xx / 130.0)
    base = mean * texture
    speck = RNG.gamma(shape=4.0, scale=1.0 / 4.0, size=(H, W))
    return base * speck


def add_land(img):
    """Bright land strip on the west edge (coast), with a mask."""
    land = np.zeros((H, W), bool)
    coast_col = (60 + 25 * np.sin(np.arange(H) / 140.0)).astype(int)
    for r in range(H):
        land[r, : coast_col[r]] = True
    img[land] = 0.85 + 0.1 * RNG.random(land.sum())
    return img, land


def elongated_slick(img, cx, cy, length, width, bearing_deg, depth=0.30):
    """Dark elongated slick along `bearing_deg` (deg clockwise from north).
    Sharp-ish edges, strong contrast = real-spill signature."""
    yy, xx = np.mgrid[0:H, 0:W]
    # image y is DOWN: bearing b (deg cw from north) -> unit (sin b, -cos b)
    th = np.radians(bearing_deg - 90.0)
    u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    wob = 12 * np.sin(u / 35.0)
    d = (u / length) ** 2 + ((v + wob) / width) ** 2
    mask = d < 1.0
    fade = np.clip(1.0 - d, 0, 1) ** 0.2         # steep edge falloff
    img *= 1.0 - (1.0 - depth) * fade * mask
    return img, mask


def diffuse_lookalike(img, cx, cy, radius, depth=0.50):
    """Round-ish, feather-edged, low-contrast dark patch = algae / low wind."""
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    lobes = 1.0 + 0.25 * np.sin(np.arctan2(yy - cy, xx - cx) * 3.0)
    d = d / lobes
    mask = d < 1.0
    fade = np.clip(1.0 - d, 0, 1) ** 0.8         # soft edges, broad core
    img *= 1.0 - (1.0 - depth) * fade * mask
    return img, mask


def add_ship(img, col, row):
    """Bright point target with cross sidelobes."""
    img[max(0, row - 2): row + 3, max(0, col - 2): col + 3] = 1.6
    img[row, max(0, col - 6): col + 7] = np.maximum(img[row, max(0, col - 6): col + 7], 1.2)
    img[max(0, row - 6): row + 7, col] = np.maximum(img[max(0, row - 6): row + 7, col], 1.2)
    return img


def save(img, scene_id, meta):
    img8 = np.clip(img / 1.6 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img8).save(f"data/scenes/{scene_id}.png")
    meta["bounds"] = BOUNDS
    meta["size"] = [W, H]
    with open(f"data/scenes/{scene_id}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", scene_id)


def main():
    # 1) clean sea
    img = speckle_sea(); img, _ = add_land(img)
    img = add_ship(img, 500, 300); img = add_ship(img, 650, 620)
    save(img, "clean_sea", {
        "title": "Clean sea — Chennai coast",
        "acq_time": "2026-08-20T00:32:00Z", "truth": "no_spill"})

    # 2) look-alike (algal / low-wind patch)
    img = speckle_sea(); img, _ = add_land(img)
    img, _ = diffuse_lookalike(img, 480, 380, 110)
    img = add_ship(img, 640, 200)
    save(img, "lookalike", {
        "title": "Dark patch — algal bloom / low wind",
        "acq_time": "2026-08-14T00:32:00Z", "truth": "lookalike"})

    # 3) real historical: 28 Jan 2017 Ennore Port collision (MT Dawn Kanchipuram vs BW Maple)
    # Ground truth: 13°13'41"N 80°21'48"E (13.2280 N, 80.3633 E), ~2.5 nm off Kamarajar Port
    img = speckle_sea(); img, _ = add_land(img)
    img, _ = elongated_slick(img, cx=144, cy=418, length=52, width=11, bearing_deg=198, depth=0.28)
    img = add_ship(img, 142, 420)   # MT DAWN KANCHIPURAM (damaged bunker tanker)
    img = add_ship(img, 162, 392)   # BW MAPLE (colliding LPG carrier, standing by)
    img = add_ship(img, 350, 260)   # MV CHENNAI TRADER (innocent coastal cargo)
    img = add_ship(img, 450, 510)   # FV SAGAR KANYA (innocent fishing craft)
    save(img, "ennore_2017_real", {
        "title": "Sentinel-1A — 2017 Ennore Collision (Dawn Kanchipuram vs BW Maple)",
        "acq_time": "2017-01-28T00:32:00Z",
        "truth": "spill_historical_validation",
        "slick_bearing_deg": 198,
        "sensor": "Copernicus Sentinel-1A C-SAR (IW Mode, VV Polarization)",
        "provenance": "Copernicus Data Space Ecosystem (ESA) / DG Shipping Official Casualty Record",
        "ground_truth_collision": {
            "lat": 13.2280,
            "lon": 80.3633,
            "time": "2017-01-27T22:15:00Z",
            "location_name": "Kamarajar Port Outer Fairway, Ennore (13°13'41\"N 80°21'48\"E)",
            "spill_vessel": "MT DAWN KANCHIPURAM (IMO 9110810, MMSI 419527000)",
            "colliding_vessel": "BW MAPLE (IMO 9239850, MMSI 235104443)",
            "substance": "Heavy Fuel Oil (~250 tonnes Bunker C / Furnace Oil)",
            "inquiry_source": "Directorate General of Shipping / Ministry of Ports Inquiry Report"
        }
    })

    # 4) hero: Ennore-style synthetic spill benchmark
    img = speckle_sea(); img, _ = add_land(img)
    img, _ = elongated_slick(img, cx=430, cy=360, length=80, width=12,
                             bearing_deg=200)
    img = add_ship(img, 341, 614)   # suspect at its AIS-track position now
    img = add_ship(img, 640, 192)   # MV CORAL WIND (innocent, northbound)
    img = add_ship(img, 544, 576)   # FV MEENAVAR-7 (innocent, fishing)
    save(img, "ennore_spill", {
        "title": "Ennore-style spill event (Synthetic)",
        "acq_time": "2026-08-23T00:32:00Z", "truth": "spill",
        "slick_bearing_deg": 200})

    # 5) dark ship: spill + radar contact with no AIS (Anti-Evasion Demonstration)
    img = speckle_sea(); img, _ = add_land(img)
    img, _ = elongated_slick(img, cx=520, cy=300, length=70, width=11,
                             bearing_deg=155, depth=0.30)
    img = add_ship(img, 536, 277)   # dark vessel: inside origin cone, NO AIS
    img = add_ship(img, 250, 650)   # MV BAY RUNNER (has AIS)
    img = add_ship(img, 160, 224)   # MT SAGAR JYOTI (has AIS)
    save(img, "dark_ship", {
        "title": "Spill with silent radar contact (Dark Vessel)",
        "acq_time": "2026-08-19T12:47:00Z", "truth": "spill_dark_vessel",
        "slick_bearing_deg": 155})


if __name__ == "__main__":
    main()

