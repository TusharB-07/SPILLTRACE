"""Run the full pipeline for all 4 scenes and cache results to
data/cache/<scene>_result.json. The API serves these caches; nothing runs
live during the demo.
"""
import json
import numpy as np
import synth_scene
import detect
import ais_sim
from correlate import rank_suspects, validate_ground_truth, parse_time


def slick_centroid_and_axis(det):
    poly = det["polygon_lonlat"]
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    return (float(np.mean(lats)), float(np.mean(lons))), \
        float(det["features"]["axis_bearing_deg"])


def run_scene(scene_id):
    res = detect.run(scene_id)
    t_detect = parse_time(res["meta"]["acq_time"])
    spills = [d for d in res["detections"] if d["kind"] == "spill"]

    result = {"scene": scene_id, "meta": res["meta"], "ships": res["ships"],
              "detections": res["detections"], "suspects": [],
              "dark_vessels": [], "origin_cone": [], "ais_tracks": [],
              "ground_truth_validation": None,
              "ais_note": "", "provenance": res["meta"].get("provenance", "")}

    if spills and scene_id in ("ennore_2017_real", "ennore_spill", "dark_ship"):
        if scene_id == "ennore_2017_real":
            ais = ais_sim.ennore_2017_real_ais(t_detect)
            result["ais_note"] = ("AIS reconstructed from Directorate General of Shipping "
                                  "formal casualty investigation report (Ennore 2017)")
        elif scene_id == "ennore_spill":
            ais = ais_sim.ennore_ais(t_detect)
            result["ais_note"] = ("AIS tracks SIMULATED for demo; schema "
                                  "identical to Danish DMA open AIS")
        else:
            ais = ais_sim.dark_ship_ais(t_detect)
            result["ais_note"] = ("Normal traffic carries AIS; silent vessel has no transponder")

        ais_sim.save_csv(ais, f"data/cache/{scene_id}_simulated_ais.csv")
        centroid, axis = slick_centroid_and_axis(spills[0])
        suspects, dark, cone, origins_raw = rank_suspects(
            ais, centroid, axis, t_detect, radar_ships=res["ships"])
        result["suspects"] = suspects
        result["dark_vessels"] = dark
        result["origin_cone"] = cone

        # If ground-truth anchor exists, quantify geodesic error
        if "ground_truth_collision" in res["meta"]:
            result["ground_truth_validation"] = validate_ground_truth(
                origins_raw, res["meta"]["ground_truth_collision"])

        for mmsi, tr in ais.groupby("mmsi"):
            tr = tr.sort_values("timestamp")
            result["ais_tracks"].append({
                "mmsi": int(mmsi),
                "name": str(tr.iloc[0]["name"]),
                "ship_type": str(tr.iloc[0]["ship_type"]),
                "points": [[float(r.lat), float(r.lon)] for _, r in tr.iterrows()],
            })

    with open(f"data/cache/{scene_id}_result.json", "w") as f:
        json.dump(result, f, indent=1)
    n_sp = len(spills)
    n_la = len(res["detections"]) - n_sp
    top = (f"{result['suspects'][0]['name']} "
           f"({result['suspects'][0]['confidence']})"
           if result["suspects"] else "-")
    gt_str = (f"gt_err={result['ground_truth_validation']['error_km']}km"
              if result.get("ground_truth_validation") else "")
    print(f"{scene_id:18s} spills={n_sp} lookalikes={n_la} "
          f"ships={len(res['ships'])} top_suspect={top:22s} "
          f"dark={len(result['dark_vessels'])} {gt_str}")


def main():
    synth_scene.main()
    scenes = ["clean_sea", "lookalike", "ennore_2017_real", "ennore_spill", "dark_ship"]
    json.dump(scenes, open("data/cache/scenes.json", "w"))
    for s in scenes:
        run_scene(s)


if __name__ == "__main__":
    main()

