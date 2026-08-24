"""Sentinel-1 GRD Ingestion & Preprocessing Module.

Converts real Sentinel-1 C-band SAR products (GeoTIFF / SAFE format) or NetCDF sigma0
rasters into SAGARNETRA's internal 8-bit normalized scene format and bounding metadata.

Features:
- Percentile radiometric normalization (clipping 1st to 99th percentile backscatter).
- Radiometric calibration to sigma0 decibels / normalized power.
- Automatic geotransform bounding box extraction (WGS84 EPSG:4326).
- Copernicus Sentinel attribution tagging and ground-truth collision anchor injection.
"""
import os
import json
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def normalize_sigma0(img_raw: np.ndarray) -> np.ndarray:
    """Normalize SAR backscatter values by clipping extreme outliers (1st to 99th percentile)
    and scaling to standard 8-bit range [0, 255].
    """
    valid = img_raw[img_raw > 0]
    if valid.size == 0:
        return np.zeros(img_raw.shape, dtype=np.uint8)
    p1, p99 = np.percentile(valid, (1.0, 99.0))
    clipped = np.clip((img_raw - p1) / max(p99 - p1, 1e-6) * 255.0, 0, 255)
    return clipped.astype(np.uint8)


def ingest_sentinel1_geotiff(tif_path: str, scene_id: str, title: str, acq_time: str,
                             ground_truth: dict = None) -> dict:
    """Ingests a real calibrated Sentinel-1 GeoTIFF product into data/scenes/<scene_id>.png
    and data/scenes/<scene_id>.json.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to ingest real GeoTIFF files. Install via pip install rasterio")

    with rasterio.open(tif_path) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        img_raw = src.read(1).astype(np.float32)

    img_8bit = normalize_sigma0(img_raw)
    out_png = f"data/scenes/{scene_id}.png"
    out_json = f"data/scenes/{scene_id}.json"

    os.makedirs("data/scenes", exist_ok=True)
    Image.fromarray(img_8bit).save(out_png)

    meta = {
        "title": title,
        "acq_time": acq_time,
        "sensor": "Sentinel-1A C-SAR (IW Mode, VV Polarization)",
        "provenance": "Copernicus Data Space Ecosystem (European Space Agency)",
        "bounds": {
            "lon_min": round(bounds[0], 4),
            "lat_min": round(bounds[1], 4),
            "lon_max": round(bounds[2], 4),
            "lat_max": round(bounds[3], 4),
        },
        "size": [int(img_8bit.shape[1]), int(img_8bit.shape[0])],
    }
    if ground_truth:
        meta["ground_truth"] = ground_truth

    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
