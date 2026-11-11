# COMMON/fl_core/experiment_specs.py
"""
Frozen experiment specifications for Paper 1.

Implements CONTRACT.md §14 (frozen specs) and §15 (governance workflow).
Two-field gate: an experiment executes only if both
    status == "frozen"  AND  authorization == "approved".

E0-E5 are frozen and authorized at contract freeze.
E6-E13 are placeholders with pending authorization.

No other module defines experiment configurations. The runner reads these
specs; the notebook reads these specs; nothing else does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen defaults (CONTRACT.md §14.1)
# Every experiment spec is merged on top of these.
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "model": "MobileNetV2",
    "num_clients": 100,
    "clients_per_round": 20,
    "batch_size": 32,
    "optimizer": "SGD",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "local_epochs": 1,
    "global_rounds": 10,
    "fedprox_mu": 0.01,
    "base_compute_time_s_per_sample_epoch": 0.0001,
    "base_compression_time_s_per_param": 0.00001,
    "timeout_floor_s": 60,
    "num_edges": 4,
    "clients_per_edge": 25,
}

# ---------------------------------------------------------------------------
# Frozen decisions (CONTRACT.md Preamble + Deliverable 1 v4 frozen decisions)
# Recorded here so the runner can assert them.
# ---------------------------------------------------------------------------
FROZEN_DECISIONS: dict[str, Any] = {
    "D1_model_cache": "immutable_backbone_fresh_instance_per_run",
    "D2_dirichlet_rng": "np.random.default_rng(seed)",
    "D3_jitter_seeding": "sha256(partition_hash:round:client_id)",
    "D4_straggler_metrics": ["t_max_over_median", "t_max_over_mean"],
    "D5_scaffold_optimizer": "same_sgd_as_fedavg_fedprox",
    "D6_uniform_8bit_payload": "dense_numel_times_8_bits",
    "D7_empty_edge": "explicit_exclusion_variant_a",
    "D8_timeout": "max_2x_median_nominal_pre_round_or_floor",
}

# ---------------------------------------------------------------------------
# Experiment specifications (§14.2)
# ---------------------------------------------------------------------------
EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {

    # ---------------------------------------------------------------
    # E0 — runtime calibration pilot (frozen + approved)
    # ---------------------------------------------------------------
    "E0": {
        "status": "frozen",
        "authorization": "approved",
        "is_pilot": True,
        "paper_reportable": False,
        "purpose": "runtime_calibration_and_pipeline_validation",
        "global_rounds": 10,
        "local_epochs": 1,
        "seeds": [42],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "compression": "none",
        "aggregation": "fedavg",
        "client_selection": "random",
        "conditions": {},
    },

    # ---------------------------------------------------------------
    # E1 — architecture benchmark (frozen + approved)
    # ---------------------------------------------------------------
    "E1": {
        "status": "frozen",
        "authorization": "approved",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "main_benchmark",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123, 2026],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "compression": "none",
        "aggregation": "fedavg",
        "client_selection": "random",
        "conditions": {},
    },

    # ---------------------------------------------------------------
    # E2 — client selection (frozen + approved)
    # Notebook cell supplies one of {"random", "oort"} via `conditions`.
    # ---------------------------------------------------------------
    "E2": {
        "status": "frozen",
        "authorization": "approved",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "client_selection",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123, 2026],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "compression": "none",
        "aggregation": "fedavg",
        "conditions": {"client_selection": ["random", "oort"]},
    },

    # ---------------------------------------------------------------
    # E3 — optimization (frozen + approved)
    # ---------------------------------------------------------------
    "E3": {
        "status": "frozen",
        "authorization": "approved",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "optimization",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123, 2026],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "compression": "none",
        "client_selection": "random",
        "conditions": {"aggregation": ["fedavg", "fedprox", "scaffold"]},
    },

    # ---------------------------------------------------------------
    # E4 — communication compression (frozen + approved)
    # ---------------------------------------------------------------
    "E4": {
        "status": "frozen",
        "authorization": "approved",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "communication_compression",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123, 2026],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "aggregation": "fedavg",
        "client_selection": "random",
        "conditions": {"compression": ["none", "uniform_8bit"]},
    },

    # ---------------------------------------------------------------
    # E5 — data heterogeneity screening (frozen + approved)
    # Two-seed screening protocol, explicitly labeled.
    # ---------------------------------------------------------------
    "E5": {
        "status": "frozen",
        "authorization": "approved",
        "is_pilot": False,
        "paper_reportable": True,
        "screening": True,
        "purpose": "data_heterogeneity_screening",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123],                       # two-seed screening
        "architectures": ["FLAT", "HFL"],
        "partition_type": "dirichlet",            # fixed for E5
        "compression": "none",
        "aggregation": "fedavg",
        "client_selection": "random",
        "conditions": {"alpha": [10.0, 1.0, 0.1]},
    },

    # ---------------------------------------------------------------
    # E6-E13 — reserved (status=placeholder, authorization=pending)
    # Frozen slot values recorded; full condition specification pending.
    # ---------------------------------------------------------------
    "E6": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "computational_heterogeneity",
        "global_rounds": 100,          # provisional, from upper-bound preselection
        "local_epochs": 3,
        "seeds": [42, 123],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E6_rule.json",   # must exist at authorization
    },
    "E7": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "network_heterogeneity",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E7_rule.json",
    },
    "E8": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "availability_heterogeneity",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E8_rule.json",
    },
    "E9": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "multi_dimensional_heterogeneity",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "dirichlet",
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E9_rule.json",
    },
    "E10": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "oort_times_compute_heterogeneity",
        "global_rounds": 100,
        "local_epochs": 3,
        "seeds": [42, 123],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E10_rule.json",
    },
    "E11": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "scalability",
        "global_rounds": 75,           # upper bound of 50-75 range
        "local_epochs": 3,
        "seeds": [42, 123],
        "architectures": ["FLAT", "HFL"],
        "partition_type": "iid",
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E11_rule.json",
    },
    "E12": {
        "status": "placeholder",
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "fairness_analysis",
        "analysis_only": True,
        "source_experiments": ["E2", "E10"],       # requires both
        "conditions": {},
        "decision_rule_path": "ARCHIVE/E12_rule.json",
    },
    "E13": {
        "status": "placeholder",                     # execution not authorized yet
        "authorization": "pending",
        "is_pilot": False,
        "paper_reportable": True,
        "purpose": "final_confirmation",
        # Two-part structure: confirmation protocol is frozen,
        # shortlist rule is frozen, but execution waits for the shortlist.
        "confirmation": {
            "global_rounds": 200,
            "local_epochs": 5,
            "seeds": [42, 123, 2026],
            "architectures": ["FLAT", "HFL"],
            "partition_type": "iid",
            "compression": "none",
            "aggregation": "fedavg",
            "client_selection": "random",
            "conditions": {},
        },
        "shortlist_rule_path": "ARCHIVE/E13_shortlist_rule.json",
        "shortlist_path": "ARCHIVE/E13_shortlist.json",
        "conditions": {},
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS = (
    "status",
    "authorization",
    "is_pilot",
    "paper_reportable",
    "purpose",
    "conditions",
)
_REQUIRED_FIELDS_TRAINING = (
    "global_rounds",
    "local_epochs",
    "seeds",
    "architectures",
    "partition_type",
)


def _validate_basic_structure(exp_id: str, spec: dict) -> None:
    """Structural checks common to every spec. Fails closed on any deviation."""
    if not isinstance(spec, dict):
        raise ValueError(f"{exp_id}: spec must be a dict, got {type(spec).__name__}")

    for key in _REQUIRED_FIELDS:
        if key not in spec:
            raise ValueError(f"{exp_id}: missing required field '{key}'")

    if spec["status"] not in {"placeholder", "frozen"}:
        raise ValueError(f"{exp_id}: invalid status '{spec['status']}'")
    if spec["authorization"] not in {"pending", "approved"}:
        raise ValueError(f"{exp_id}: invalid authorization '{spec['authorization']}'")

    # Analysis-only experiments (E12) have no training fields.
    # E13 nests its training fields under "confirmation"; use it as
    # the source of truth when present.  # FIX-A applied
    if not spec.get("analysis_only", False):
        source = spec.get("confirmation", spec)
        for key in _REQUIRED_FIELDS_TRAINING:
            if key not in source:
                raise ValueError(f"{exp_id}: missing training field '{key}'")

        if not isinstance(source.get("global_rounds"), int) or source["global_rounds"] <= 0:
            raise ValueError(
                f"{exp_id}: 'global_rounds' must be a positive integer, "
                f"got {source.get('global_rounds')!r}"
            )
        if not isinstance(source.get("local_epochs"), int) or source["local_epochs"] <= 0:
            raise ValueError(
                f"{exp_id}: 'local_epochs' must be a positive integer, "
                f"got {source.get('local_epochs')!r}"
            )
        if not isinstance(source["seeds"], list) or not source["seeds"]:
            raise ValueError(f"{exp_id}: 'seeds' must be a non-empty list")
        if not all(isinstance(s, int) for s in source["seeds"]):
            raise ValueError(f"{exp_id}: 'seeds' must contain ints")
        if not isinstance(source["architectures"], list) or not source["architectures"]:
            raise ValueError(f"{exp_id}: 'architectures' must be a non-empty list")
        if not all(a in {"FLAT", "HFL"} for a in source["architectures"]):
            raise ValueError(
                f"{exp_id}: 'architectures' must be subset of {{FLAT, HFL}}"
            )
        if not isinstance(spec["conditions"], dict):
            raise ValueError(f"{exp_id}: 'conditions' must be a dict")


def _validate_conditions(exp_id: str, spec: dict, overrides: dict | None) -> None:
    """
    Enforce the two rules for `conditions`:

    - If the spec declares condition keys, the overrides MUST provide exactly
      those keys, and each value MUST be in the allowed list.
    - If the spec declares no condition keys, the overrides MUST be empty.

    This is what makes each notebook cell an explicit, single-condition unit
    and prevents a runner call from silently varying two factors at once.
    """
    overrides = overrides or {}
    declared = spec.get("conditions", {})

    if not isinstance(overrides, dict):
        raise ValueError(f"{exp_id}: conditions must be a dict, got {type(overrides).__name__}")

    if declared:
        declared_keys = set(declared.keys())
        provided_keys = set(overrides.keys())
        if declared_keys != provided_keys:
            raise ValueError(
                f"{exp_id}: conditions mismatch. "
                f"Spec declares {sorted(declared_keys)}; "
                f"cell provided {sorted(provided_keys)}. "
                f"Each cell must supply exactly the declared condition keys."
            )
        for key, value in overrides.items():
            allowed = declared[key]
            if value not in allowed:
                raise ValueError(
                    f"{exp_id}: condition '{key}={value!r}' not in allowed values "
                    f"{allowed!r}"
                )
    else:
        if overrides:
            raise ValueError(
                f"{exp_id}: spec declares no conditions; got {overrides!r}. "
                f"Fixed experiments run with empty overrides."
            )


def get_experiment_spec(exp_id: str) -> dict:
    """
    Return a deep copy of the frozen spec for `exp_id`.

    Raises KeyError if `exp_id` is not registered, ValueError if the spec
    is malformed. This is the only function the notebook should call to
    retrieve an experiment's configuration; nothing outside this module
    may hard-code experiment parameters (Preamble rule 2).
    """
    if exp_id not in EXPERIMENT_SPECS:
        raise KeyError(f"Unknown experiment: {exp_id!r}")
    spec = EXPERIMENT_SPECS[exp_id]
    _validate_basic_structure(exp_id, spec)
    # Deep copy via json round-trip (all fields are JSON-serialisable by design)
    return json.loads(json.dumps(spec))


def validate_experiment_spec(exp_id: str, overrides: dict | None = None) -> dict:
    """
    Validate that this experiment may be called with `overrides` and return
    the fully-merged effective configuration.

    Called by the runner before every execution. Raises ValueError on any
    structural or condition mismatch. Does NOT check the two-field gate —
    that is `authorize_execution`.
    """
    spec = get_experiment_spec(exp_id)
    _validate_basic_structure(exp_id, spec)
    _validate_conditions(exp_id, spec, overrides)

    # Merge defaults + spec + overrides (overrides win on conflict)
    effective = {**DEFAULTS}
    for key, value in spec.items():
        if key in {"conditions", "confirmation"}:   # not runtime params
            continue
        effective[key] = value
    # E13's frozen confirmation protocol becomes the effective spec at runtime
    if exp_id == "E13" and "confirmation" in spec:
        for key, value in spec["confirmation"].items():
            if key != "conditions":
                effective[key] = value
    for key, value in (overrides or {}).items():
        effective[key] = value

    # Sanity: effective must contain every training field
    if not spec.get("analysis_only", False):
        for key in _REQUIRED_FIELDS_TRAINING:
            if key not in effective:
                raise ValueError(f"{exp_id}: effective config missing '{key}'")

    return effective


# ---------------------------------------------------------------------------
# Two-field gate + spec hash verification (§13.1a, §15.8)
# ---------------------------------------------------------------------------
def _hash_spec(spec: dict) -> str:
    """Deterministic sha256 over the JSON-canonicalised spec."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def authorize_execution(
    exp_id: str,
    protocol_revisions_dir: Path,
    expected_spec_hash: str | None = None,
) -> None:
    """
    Enforce the two-field gate (§13.1a) and spec-hash binding (§15.8).

    Raises PermissionError if:
      - status != "frozen" or authorization != "approved"
      - the latest protocol revision's spec_hash does not match the current spec

    Called by the runner as the first step of every execution.
    """
    spec = get_experiment_spec(exp_id)

    if spec["status"] != "frozen" or spec["authorization"] != "approved":
        raise PermissionError(
            f"{exp_id} not executable: status={spec['status']!r}, "
            f"authorization={spec['authorization']!r}. Promotion requires an "
            f"ARCHIVE/protocol_revisions/ entry with spec_hash binding."
        )

    current_hash = _hash_spec(spec)

    # ---- FIX-B: verify provenance via contract_freeze_manifest.json ----
    manifest_path = Path(protocol_revisions_dir) / "contract_freeze_manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if exp_id not in manifest:
            raise PermissionError(
                f"{exp_id}: provenance cannot be verified — "
                f"not present in {manifest_path.name}"
            )
        recorded_hash = manifest[exp_id].get("spec_hash")
        if recorded_hash != current_hash:
            raise PermissionError(
                f"{exp_id}: spec hash mismatch. "
                f"manifest={recorded_hash}, current={current_hash}. "
                f"Spec was modified after approval."
            )

    # Find the most recent authorization record for this experiment
    revisions_dir = Path(protocol_revisions_dir) / "protocol_revisions"
    if not revisions_dir.exists():
        if manifest is None:
            raise PermissionError(
                f"{exp_id}: provenance cannot be verified — "
                f"no {manifest_path.name} and no protocol_revisions/ directory."
            )
        if expected_spec_hash is not None and expected_spec_hash != current_hash:
            raise PermissionError(
                f"{exp_id}: spec hash mismatch. "
                f"expected {expected_spec_hash}, current {current_hash}. "
                f"Spec was modified after approval."
            )
        return

    latest_record = None
    latest_mtime = -1.0
    for path in sorted(revisions_dir.glob("REV-*.json")):
        stem = path.stem
        if "_" in stem:
            continue
        try:
            with open(path, encoding="utf-8") as _f:
                _rec = json.load(_f)
        except (json.JSONDecodeError, OSError):
            continue
        if _rec.get("experiment_id") != exp_id:
            continue
        if path.stat().st_mtime > latest_mtime:
            latest_mtime = path.stat().st_mtime
            latest_record = path

    if latest_record is not None:
        with open(latest_record, encoding="utf-8") as f:
            record = json.load(f)
        if record.get("new_authorization") != "approved":
            raise PermissionError(
                f"{exp_id}: latest revision {latest_record.name} "
                f"does not grant 'approved' authorization."
            )
        if record.get("spec_hash") != current_hash:
            raise PermissionError(
                f"{exp_id}: spec hash mismatch vs revision "
                f"{latest_record.name}. expected {record.get('spec_hash')}, "
                f"current {current_hash}. Spec was modified after authorization."
            )

    if expected_spec_hash is not None and expected_spec_hash != current_hash:
        raise PermissionError(
            f"{exp_id}: caller-supplied spec_hash {expected_spec_hash} "
            f"does not match current {current_hash}."
        )


# ---------------------------------------------------------------------------
# Completion gates (§15.7)
# ---------------------------------------------------------------------------
def check_completion_gates(
    exp_id: str,
    dataset: str,
    results_dir: Path,
    logs_dir: Path,
    archive_dir: Path,
) -> None:
    """
    Enforce per-experiment completion gates BEFORE execution.

    Called by the runner after `authorize_execution`. For most experiments
    there is no completion gate; E12 and E13 have one.
    """
    if exp_id == "E12":
        spec = get_experiment_spec(exp_id)

        # FIX-E2: scan source logs for malformed JSON first, so a
        # malformed log is reported before missing-dependency errors.
        for _source in spec.get("source_experiments", []):
            _log_dir = Path(logs_dir) / dataset / _source
            if not _log_dir.exists():
                continue
            for _log_path in sorted(_log_dir.glob("*.jsonl")):
                try:
                    with open(_log_path, encoding="utf-8") as _fh:
                        for _line_no, _line in enumerate(_fh, 1):
                            _stripped = _line.strip()
                            if not _stripped:
                                continue
                            try:
                                json.loads(_stripped)
                            except json.JSONDecodeError as _exc:
                                raise PermissionError(
                                    f"E12 log contains invalid JSON at "
                                    f"{_log_path.name}:{_line_no}: {_exc}"
                                ) from _exc
                except PermissionError:
                    raise
                except OSError as _exc:
                    raise PermissionError(
                        f"E12 log unreadable: {_log_path}: {_exc}"
                    ) from _exc

        # Then check dependency logs (E2, E10)
        for source in spec.get("source_experiments", []):
            log_dir = Path(logs_dir) / dataset / source
            if not log_dir.exists():
                raise PermissionError(
                    f"E12 requires {source} logs; directory does not exist: {log_dir}"
                )
            has_flat = any(log_dir.glob("*flat*.jsonl"))
            has_hfl = any(log_dir.glob("*hfl*.jsonl"))
            if not (has_flat and has_hfl):
                raise PermissionError(
                    f"E12 requires both Flat and HFL logs from {source}; "
                    f"found flat={has_flat}, hfl={has_hfl} in {log_dir}"
                )

    elif exp_id == "E13":
        spec = get_experiment_spec(exp_id)
        rule_path = Path(archive_dir) / "E13_shortlist_rule.json"
        shortlist_path = Path(archive_dir) / "E13_shortlist.json"

        if not rule_path.exists():
            raise PermissionError(
                f"E13 requires frozen shortlist rule at {rule_path}; not found."
            )
        if not shortlist_path.exists():
            raise PermissionError(
                f"E13 requires generated shortlist at {shortlist_path}; not found. "
                f"Run result_analysis.apply_shortlist_rule first."
            )

        with open(rule_path, encoding="utf-8") as f:
            rule = json.load(f)
        with open(shortlist_path, encoding="utf-8") as f:
            shortlist = json.load(f)

        expected_rule_hash = hashlib.sha256(rule_path.read_bytes()).hexdigest()
        if shortlist.get("rule_sha256") != expected_rule_hash:
            raise PermissionError(
                f"E13 shortlist rule_sha256={shortlist.get('rule_sha256')!r} "
                f"does not match frozen rule hash {expected_rule_hash}. "
                f"Shortlist must be regenerated from the archived rule."
            )

        eligible = set(rule.get("eligible_experiments", []))
        for entry in shortlist.get("shortlist", []):
            source_exp = entry.get("source_exp")
            if source_exp not in eligible:
                raise PermissionError(
                    f"E13 shortlist contains source_exp={source_exp!r} "
                    f"not in rule. Eligible: {sorted(eligible)}"
                )

        if rule.get("rule_version") != shortlist.get("rule_version"):
            raise PermissionError(
                f"E13 shortlist rule_version={shortlist.get('rule_version')!r} "
                f"does not match frozen rule_version={rule.get('rule_version')!r}. "
                f"Shortlist must be generated by the archived rule."
            )

        if not shortlist.get("shortlist"):
            raise PermissionError(f"E13 shortlist is empty: {shortlist_path}")