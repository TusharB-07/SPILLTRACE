# SAGARNETRA (SIH26143) — Oil Spill Detection + Vessel Attribution

SAR scene -> dark-spot detection -> spill vs look-alike classification ->
hydrodynamic drift backtracking -> AIS spacetime correlation -> ranked suspects +
dark-vessel detection -> printable legal evidence report. Fully offline demo.

## Run (2 commands)

    source .venv/bin/activate
    uvicorn api.main:app --port 8000

Open http://localhost:8000 — click through the 5 scenes in this order:
1. **clean sea**: normal sea state baseline (0 false alarms)
2. **lookalike**: diffuse algae/low-wind patch rejected (sharpness/contrast discrimination)
3. **ennore_2017_real (Hero Benchmark)**: 28 Jan 2017 Ennore collision (13°13′41″N 80°21′48″E) —
   backtracks origin cone within **1.41 km** of official DG Shipping casualty site, ranking
   **MT DAWN KANCHIPURAM** (75% confidence) as primary spill source.
4. **ennore_spill**: synthetic demonstration spill (MT KAVERI PRIDE ranked 88%)
5. **dark_ship**: anti-evasion demo — radar contact at slick origin with NO AIS flagged **PRIME SUSPECT**.

To regenerate scenes/results:  PYTHONPATH=src python src/pipeline.py

## Ground-Truth Historical Anchor (2017 Ennore Incident)
- **Casualty Event**: Collision between LPG tanker *BW Maple* and bunker tanker *MT Dawn Kanchipuram* on 28 Jan 2017 at Kamarajar Port outer fairway (13.2280°N, 80.3633°E).
- **Satellite Data**: Copernicus Sentinel-1A C-SAR (IW Mode, VV Polarization).
- **Meteorological Forcing**: ERA5 NE-monsoon wind + CMEMS southward coastal current.
- **Validation Metric**: Origin cone closest approach is **1.41 km** from official casualty coordinates.

## Files
    src/real_scene_ingest.py Sentinel-1 GRD ingestion & radiometric preprocessing
    src/synth_scene.py       SAR scene generator (supports real & synthetic benchmarks)
    src/detect.py            dark spots, spill/look-alike classification, ship point targets
    src/ais_sim.py           AIS simulation & 2017 casualty track reconstruction (DMA schema)
    src/correlate.py         hydrodynamic backtracking, ground-truth error & suspect scoring
    src/pipeline.py          runs end-to-end pipeline and caches JSONs
    src/train_deeplab.py     PyTorch DeepLabv3+ training script (Krestenitis SAR dataset)
    api/main.py              FastAPI server for offline UI & cached analysis
    web/index.html           Offline Leaflet console & formal evidence report generator

