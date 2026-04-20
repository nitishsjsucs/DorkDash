"""
Persistent cache for experiment + swarm results.

Results are saved to JSON files under backend/cache/*.json after each run,
and auto-loaded on backend startup so a fresh clone of the repo can see
past results in the frontend without re-running anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent
EXPERIMENT_FILE = CACHE_DIR / "experiment_snapshot.json"
SWARM_FILE = CACHE_DIR / "swarm_result.json"


def _default_serializer(obj: Any):
    """JSON serializer that tolerates numpy scalars, arrays, and pandas objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=_default_serializer)
    tmp.replace(path)


def save_experiment_snapshot(
    experiment_results: Dict[str, Any],
    sim_snapshot: Optional[Dict[str, Any]] = None,
    adjacency: Optional[Dict[str, Any]] = None,
    bo_posteriors: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """Persist the full bundle the frontend needs to render the Overview tab."""
    payload = {
        "saved_at": time.time(),
        "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_results": experiment_results,
        "sim_snapshot": sim_snapshot,
        "adjacency": adjacency,
        "bo_posteriors": {str(k): v for k, v in (bo_posteriors or {}).items()},
    }
    _atomic_write(EXPERIMENT_FILE, payload)
    return payload


def save_swarm_result(swarm_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "saved_at": time.time(),
        "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": swarm_payload,
    }
    _atomic_write(SWARM_FILE, payload)
    return payload


def load_experiment_snapshot() -> Optional[Dict[str, Any]]:
    if not EXPERIMENT_FILE.exists():
        return None
    try:
        with EXPERIMENT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_swarm_result() -> Optional[Dict[str, Any]]:
    if not SWARM_FILE.exists():
        return None
    try:
        with SWARM_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_status() -> Dict[str, Any]:
    def _meta(path: Path, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if payload is None:
            return {"exists": False}
        return {
            "exists": True,
            "path": str(path.name),
            "saved_at": payload.get("saved_at"),
            "saved_at_iso": payload.get("saved_at_iso"),
        }

    exp = load_experiment_snapshot()
    swm = load_swarm_result()

    exp_meta = _meta(EXPERIMENT_FILE, exp)
    if exp is not None and exp.get("experiment_results"):
        r = exp["experiment_results"]
        exp_meta["n_days"] = r.get("n_days")
        exp_meta["total_steps"] = r.get("total_steps")
        exp_meta["total_records"] = r.get("total_records")

    swm_meta = _meta(SWARM_FILE, swm)
    if swm is not None and swm.get("result"):
        res = swm["result"]
        swm_meta["n_days"] = res.get("n_days")
        swm_meta["n_dashers"] = res.get("n_dashers")
        swm_meta["n_zones"] = res.get("n_zones")

    return {"experiment": exp_meta, "swarm": swm_meta}
