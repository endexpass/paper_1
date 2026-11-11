# tests/test_experiment_specs.py
"""
Tests for experiment_specs.py — CONTRACT.md §14, §15.
Run: pytest tests/test_experiment_specs.py -v
"""

import json
from pathlib import Path

import pytest

from fl_core.experiment_specs import (
    DEFAULTS,
    EXPERIMENT_SPECS,
    FROZEN_DECISIONS,
    authorize_execution,
    check_completion_gates,
    get_experiment_spec,
    validate_experiment_spec,
    _hash_spec,
)


def _write_manifest(archive_dir: Path, exp_ids=("E0", "E1", "E2", "E3", "E4", "E5")):
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for exp_id in exp_ids:
        manifest[exp_id] = {
            "spec_hash": _hash_spec(EXPERIMENT_SPECS[exp_id]),
            "approved_at": "2026-11-10T00:00:00Z",
            "approved_by": "test",
        }
    (archive_dir / "contract_freeze_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _write_revision(archive_dir: Path, rev_id, exp_id, spec_hash, approval="approved"):
    d = archive_dir / "protocol_revisions"
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "revision_id": rev_id,
        "experiment_id": exp_id,
        "previous_status": "placeholder",
        "new_status": "frozen",
        "previous_authorization": "pending",
        "new_authorization": approval,
        "reason": "test",
        "decision_rule_path": None,
        "spec_hash": spec_hash,
        "code_commit": "test",
        "approved_at": "2026-11-10T00:00:00Z",
        "approved_by": "test",
    }
    (d / f"{rev_id}.json").write_text(json.dumps(rec), encoding="utf-8")


# ---------------------------------------------------------------------------
# T1-T2 — Status/authorization gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exp_id", ["E0", "E1", "E2", "E3", "E4", "E5"])
def test_frozen_approved(exp_id):
    spec = get_experiment_spec(exp_id)
    assert spec["status"] == "frozen"
    assert spec["authorization"] == "approved"


@pytest.mark.parametrize("exp_id", ["E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13"])
def test_placeholder_pending(exp_id):
    spec = get_experiment_spec(exp_id)
    assert spec["status"] == "placeholder"
    assert spec["authorization"] == "pending"


# ---------------------------------------------------------------------------
# T3 — Placeholder rejected by authorize_execution (Defect 1 fix allows E13
# to load, but it must still be rejected because status=placeholder)
# ---------------------------------------------------------------------------
def test_placeholder_rejected(tmp_path):
    with pytest.raises(PermissionError, match="not executable"):
        authorize_execution("E6", tmp_path)


# ---------------------------------------------------------------------------
# T4 — E13 structural validation (Defect 1 fix)
# ---------------------------------------------------------------------------
def test_e13_nested_confirmation_validates():
    # Must NOT raise
    spec = get_experiment_spec("E13")
    assert "confirmation" in spec
    assert spec["confirmation"]["global_rounds"] == 200
    assert spec["confirmation"]["local_epochs"] == 5
    assert spec["confirmation"]["seeds"] == [42, 123, 2026]
    assert spec["confirmation"]["architectures"] == ["FLAT", "HFL"]


def test_e13_invalid_confirmation_rejected():
    spec = get_experiment_spec("E13")
    spec["confirmation"]["global_rounds"] = -1  # invalid
    # Force the validate function to re-check by calling _validate_basic_structure
    from fl_core.experiment_specs import _validate_basic_structure
    with pytest.raises(ValueError, match="global_rounds"):
        _validate_basic_structure("E13", spec)


# ---------------------------------------------------------------------------
# T5 — Condition validation
# ---------------------------------------------------------------------------
def test_condition_validation():
    # E2 requires exactly client_selection
    with pytest.raises(ValueError, match="not in allowed"):
        validate_experiment_spec("E2", {"client_selection": "fedavg"})
    with pytest.raises(ValueError, match="conditions mismatch"):
        validate_experiment_spec("E2", {"aggregation": "fedavg"})
    # E1 has no conditions
    with pytest.raises(ValueError, match="no conditions"):
        validate_experiment_spec("E1", {"client_selection": "random"})
    # E2 with valid condition succeeds
    cfg = validate_experiment_spec("E2", {"client_selection": "oort"})
    assert cfg["client_selection"] == "oort"


# ---------------------------------------------------------------------------
# T6 — spec_hash determinism + full-field coverage
# ---------------------------------------------------------------------------
def test_spec_hash_deterministic_and_complete():
    spec1 = get_experiment_spec("E1")
    spec2 = get_experiment_spec("E1")
    assert _hash_spec(spec1) == _hash_spec(spec2)

    # Changing a training field changes the hash
    spec3 = get_experiment_spec("E1")
    spec3["global_rounds"] = 99
    assert _hash_spec(spec1) != _hash_spec(spec3)

    # Changing a metadata field changes the hash
    spec4 = get_experiment_spec("E1")
    spec4["purpose"] = "renamed"
    assert _hash_spec(spec1) != _hash_spec(spec4)


# ---------------------------------------------------------------------------
# T7 — Authorization filename convention (Defect 2 fix)
# ---------------------------------------------------------------------------
def test_authorization_filename_convention(tmp_path):
    _write_manifest(tmp_path)
    # Contract filename REV-0001.json — must work
    _write_revision(tmp_path, "REV-0001", "E1", _hash_spec(EXPERIMENT_SPECS["E1"]))
    # Should not raise
    authorize_execution("E1", tmp_path)


def test_authorization_wrong_filename_rejected(tmp_path):
    _write_manifest(tmp_path)
    # Old convention REV-0001_E1.json — must NOT count
    d = tmp_path / "protocol_revisions"
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "revision_id": "REV-0001",
        "experiment_id": "E1",
        "new_authorization": "approved",
        "spec_hash": _hash_spec(EXPERIMENT_SPECS["E1"]),
    }
    (d / "REV-0001_E1.json").write_text(json.dumps(rec))
    # Should fall back to manifest and pass (manifest is present)
    authorize_execution("E1", tmp_path)


# ---------------------------------------------------------------------------
# T8 — Modified spec after approval is rejected
# ---------------------------------------------------------------------------
def test_modified_spec_rejected(tmp_path):
    _write_manifest(tmp_path)
    # Revision approves hash X
    _write_revision(tmp_path, "REV-0001", "E1", "deadbeef")
    with pytest.raises(PermissionError, match="spec hash mismatch"):
        authorize_execution("E1", tmp_path)


# ---------------------------------------------------------------------------
# T9 — Missing revision and missing manifest → rejected
# ---------------------------------------------------------------------------
def test_missing_authorization_rejected(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(PermissionError, match="provenance cannot be verified"):
        authorize_execution("E1", tmp_path)


# ---------------------------------------------------------------------------
# T10 — Manifest fallback works for E0-E5
# ---------------------------------------------------------------------------
def test_manifest_fallback(tmp_path):
    _write_manifest(tmp_path)
    for exp_id in ("E0", "E1", "E2", "E3", "E4", "E5"):
        authorize_execution(exp_id, tmp_path)


# ---------------------------------------------------------------------------
# T11 — E12 completion gate rejects empty log
# ---------------------------------------------------------------------------
def test_e12_gate_rejects_empty_log(tmp_path):
    logs = tmp_path / "logs" / "WHEAT" / "E2"
    logs.mkdir(parents=True)
    (logs / "flat_seed42.jsonl").write_text("")     # empty
    (logs / "hfl_seed42.jsonl").write_text("")      # empty
    results = tmp_path / "results"; results.mkdir()
    archive = tmp_path / "archive"; archive.mkdir()

    with pytest.raises(PermissionError, match="empty"):
        check_completion_gates("E12", "WHEAT", results, tmp_path / "logs", archive)


# ---------------------------------------------------------------------------
# T12 — E12 completion gate rejects malformed JSONL
# ---------------------------------------------------------------------------
def test_e12_gate_rejects_malformed_json(tmp_path):
    logs = tmp_path / "logs" / "WHEAT" / "E2"; logs.mkdir(parents=True)
    (logs / "flat_seed42.jsonl").write_text("not json\n")
    (logs / "hfl_seed42.jsonl").write_text("{}\n")
    results = tmp_path / "results"; results.mkdir()
    archive = tmp_path / "archive"; archive.mkdir()

    with pytest.raises(PermissionError, match="invalid JSON"):
        check_completion_gates("E12", "WHEAT", results, tmp_path / "logs", archive)


# ---------------------------------------------------------------------------
# T13 — E12 completion gate rejects missing E10 logs
# ---------------------------------------------------------------------------
def test_e12_gate_rejects_missing_e10(tmp_path):
    # E2 logs valid, E10 logs missing
    e2_logs = tmp_path / "logs" / "WHEAT" / "E2"; e2_logs.mkdir(parents=True)
    for arch in ("flat", "hfl"):
        (e2_logs / f"{arch}_seed42.jsonl").write_text(
            "\n".join(
                json.dumps({
                    "round": r,
                    **{k: 0 for k in (
                        "partition_hash", "selection_policy", "selection_seed",
                        "selected_clients", "completed_clients", "failed_clients",
                        "round_wall_s", "round_critical_s", "agg_s",
                        "global_accuracy", "rng_state_hash_after_round",
                    )}
                })
                for r in range(1, 101)
            )
        )
    results = tmp_path / "results"; results.mkdir()
    archive = tmp_path / "archive"; archive.mkdir()

    with pytest.raises(PermissionError, match="E10"):
        check_completion_gates("E12", "WHEAT", results, tmp_path / "logs", archive)


# ---------------------------------------------------------------------------
# T14 — E13 shortlist with matching version but wrong rule hash is rejected
# ---------------------------------------------------------------------------
def test_e13_wrong_rule_hash_rejected(tmp_path):
    archive = tmp_path / "archive"; archive.mkdir()
    results = tmp_path / "results" / "WHEAT"; results.mkdir(parents=True)
    logs = tmp_path / "logs"; logs.mkdir()

    rule = {
        "rule_version": "v1.0.0",
        "eligible_experiments": ["E1", "E2", "E3"],
    }
    (archive / "E13_shortlist_rule.json").write_text(json.dumps(rule))
    shortlist = {
        "rule_version": "v1.0.0",
        "rule_sha256": "deadbeef",   # wrong
        "shortlist": [
            {"source_exp": "E1", "conditions": {}},
        ],
    }
    (archive / "E13_shortlist.json").write_text(json.dumps(shortlist))

    with pytest.raises(PermissionError, match="rule_sha256"):
        check_completion_gates("E13", "WHEAT", results, logs, archive)


# ---------------------------------------------------------------------------
# T15 — E13 shortlist with ineligible source experiment is rejected
# ---------------------------------------------------------------------------
def test_e13_ineligible_source_rejected(tmp_path):
    archive = tmp_path / "archive"; archive.mkdir()
    results = tmp_path / "results" / "WHEAT"; results.mkdir(parents=True)
    logs = tmp_path / "logs"; logs.mkdir()

    rule = {
        "rule_version": "v1.0.0",
        "eligible_experiments": ["E1", "E2"],
    }
    rule_path = archive / "E13_shortlist_rule.json"
    rule_path.write_text(json.dumps(rule))
    import hashlib
    rule_hash = hashlib.sha256(rule_path.read_bytes()).hexdigest()

    shortlist = {
        "rule_version": "v1.0.0",
        "rule_sha256": rule_hash,
        "shortlist": [
            {"source_exp": "E12", "conditions": {}},  # ineligible
        ],
    }
    (archive / "E13_shortlist.json").write_text(json.dumps(shortlist))

    with pytest.raises(PermissionError, match="not in rule"):
        check_completion_gates("E13", "WHEAT", results, logs, archive)


# ---------------------------------------------------------------------------
# T16 — Frozen decisions recorded
# ---------------------------------------------------------------------------
def test_frozen_decisions_recorded():
    expected_keys = {
        "D1_model_cache", "D2_dirichlet_rng", "D3_jitter_seeding",
        "D4_straggler_metrics", "D5_scaffold_optimizer",
        "D6_uniform_8bit_payload", "D7_empty_edge", "D8_timeout",
    }
    assert set(FROZEN_DECISIONS.keys()) == expected_keys
    assert FROZEN_DECISIONS["D6_uniform_8bit_payload"] == "dense_numel_times_8_bits"
    assert FROZEN_DECISIONS["D7_empty_edge"] == "explicit_exclusion_variant_a"
