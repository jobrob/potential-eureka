"""Sprint 8D: sweep / manifest / sampler / attribution unit tests (§10).

All tests are FAST: pure orchestration/analysis logic, scripted/stubbed agents,
tiny synthetic ladders. No SB3 training, no real league runs (heavy compute lives
in ``experiments/``).
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from heat.ml import spaces
from heat.ml.model import PPOConfig
from heat.ml.sweep import (
    Attribution,
    FactorAxis,
    ManifestReader,
    ManifestWriter,
    RunConfig,
    SweepSpec,
    balanced_fields,
    build_attribution,
    grouped_diffs,
    join_ladder,
    ols_effects,
    read_outcomes_jsonl,
    render_attribution_md,
    run_campaign,
    seat_rotations,
    write_outcomes_jsonl,
)
from heat.ml.training import CurriculumConfig, TrainingPhase, default_8c_phases
from heat.simulation.runner import GameOutcome, PlayerOutcome
from heat.simulation.stats import EloRating, TrueSkillRating, compute_elo


# ===========================================================================
# Sweep spec + expansion (§10)
# ===========================================================================


def test_grid_expand_cardinality():
    """axes (2,3) x 2 seeds -> exactly 12 RunConfigs with unique run_ids."""
    spec = SweepSpec(
        axes=(
            FactorAxis("ppo.shaping_weight", (0.0, 0.05)),
            FactorAxis("phases.0.steps", (100_000, 200_000, 300_000)),
        ),
        seeds=(0, 1),
    )
    runs = spec.expand()
    assert len(runs) == 2 * 3 * 2
    assert len({rc.run_id for rc in runs}) == len(runs)


def _factors_key(rc):
    return json.dumps(rc.factors, sort_keys=True, default=str)


def test_run_id_deterministic():
    """Same factor vector -> same run_id across two expand() calls."""
    spec = SweepSpec(
        axes=(FactorAxis("ppo.shaping_weight", (0.0, 0.05)),),
        seeds=(0, 1),
    )
    a = {_factors_key(rc): rc.run_id for rc in spec.expand()}
    b = {_factors_key(rc): rc.run_id for rc in spec.expand()}
    assert a == b


def test_random_respects_max_configs():
    """method='random' with max_configs=N yields exactly N cells, deterministic."""
    spec = SweepSpec(
        axes=(
            FactorAxis("ppo.shaping_weight", (0.0, 0.05, 0.1)),
            FactorAxis("phases.0.steps", (100_000, 200_000, 300_000)),
            FactorAxis("ppo.net_profile", ("small", "large")),
        ),
        seeds=(0,),
        method="random",
        max_configs=4,
    )
    a = spec.expand()
    b = spec.expand()
    assert len(a) == 4  # 4 cells x 1 seed
    assert [rc.run_id for rc in a] == [rc.run_id for rc in b]


def test_lhs_respects_max_configs_and_spreads():
    """method='lhs' with max_configs=N yields N cells, each axis level used."""
    spec = SweepSpec(
        axes=(
            FactorAxis("ppo.shaping_weight", (0.0, 0.05)),
            FactorAxis("phases.0.steps", (100_000, 200_000)),
        ),
        seeds=(0,),
        method="lhs",
        max_configs=4,
    )
    runs = spec.expand()
    assert len(runs) == 4
    # Deterministic.
    assert [rc.run_id for rc in runs] == [rc.run_id for rc in spec.expand()]
    # Each level of each axis appears at least once across the draws (spread).
    shaping_levels = {rc.factors["ppo.shaping_weight"] for rc in runs}
    assert shaping_levels == {0.0, 0.05}


def test_factor_applied_to_config():
    """A 'ppo.shaping_weight' factor sets only that field; others stay base."""
    spec = SweepSpec(
        axes=(FactorAxis("ppo.shaping_weight", (0.05,)),), seeds=(0,)
    )
    rc = spec.expand()[0]
    base = PPOConfig()
    assert rc.ppo.shaping_weight == 0.05
    # Every OTHER field equals the base preset (dataclasses.replace guard).
    for f in dataclasses.fields(PPOConfig):
        if f.name in ("shaping_weight", "seed"):
            continue
        assert getattr(rc.ppo, f.name) == getattr(base, f.name), f.name


def test_phases_factor_applied():
    """A 'phases.0.steps' level edits only phase 0's steps in the phase list."""
    spec = SweepSpec(axes=(FactorAxis("phases.0.steps", (123_456,)),), seeds=(0,))
    rc = spec.expand()[0]
    base = default_8c_phases()
    assert rc.phases[0].steps == 123_456
    # Phase 0's other fields and all later phases are untouched.
    assert rc.phases[0].name == base[0].name
    for i in range(1, len(base)):
        assert rc.phases[i] == base[i]


def test_schedule_pool_kinds_factor():
    """'schedule.pool_kinds' rebuilds the opponent ramp, keeping the solo phase."""
    spec = SweepSpec(
        axes=(FactorAxis("schedule.pool_kinds", ("weak,strong",)),), seeds=(0,)
    )
    rc = spec.expand()[0]
    kinds = [p.pool_kind for p in rc.phases]
    assert kinds == [None, "weak", "strong"]  # solo + 2 opponent phases
    # The derived schedule matches the opponent phases.
    assert [s.pool_kind for s in rc.schedule.stages] == ["weak", "strong"]


def test_steps_ratio_factor_scales_opponent_phases():
    """'phases.steps_ratio' scales each opponent phase's steps, leaves solo."""
    spec = SweepSpec(
        axes=(FactorAxis("phases.steps_ratio", ((1.0, 1.0, 2.0),)),), seeds=(0,)
    )
    rc = spec.expand()[0]
    base = default_8c_phases()
    # Solo (phase 0) unchanged; opponent phases scaled by 1,1,2 in order.
    assert rc.phases[0].steps == base[0].steps
    opp_base = [p for p in base if p.pool_kind is not None]
    opp_new = [p for p in rc.phases if p.pool_kind is not None]
    assert opp_new[0].steps == opp_base[0].steps
    assert opp_new[2].steps == opp_base[2].steps * 2


def test_unknown_factor_path_raises():
    """A typo'd axis path fails loudly at expand time, not silently."""
    spec = SweepSpec(axes=(FactorAxis("ppo.no_such_field", (1,)),), seeds=(0,))
    with pytest.raises(AttributeError):
        spec.expand()


def test_seed_as_axis_rejected():
    """'seed' must come from SweepSpec.seeds, not a FactorAxis."""
    spec = SweepSpec(axes=(FactorAxis("seed", (0, 1)),), seeds=(0,))
    with pytest.raises(ValueError, match="seed"):
        spec.expand()


# ===========================================================================
# Manifest (§10)
# ===========================================================================


def _tiny_spec():
    return SweepSpec(axes=(FactorAxis("ppo.shaping_weight", (0.0, 0.05)),), seeds=(0,))


def test_manifest_roundtrip(tmp_path):
    """started/done/failed rows -> completed_run_ids = done ids; failed carries error."""
    path = str(tmp_path / "manifest.jsonl")
    writer = ManifestWriter(path)
    runs = _tiny_spec().expand()
    rc_done, rc_failed = runs[0], runs[1]

    writer.mark_started(rc_done)
    writer.mark_done(rc_done, checkpoint="ck/done.zip", wall_clock_s=1.0, gate_score=0.7)
    writer.mark_started(rc_failed)
    writer.mark_failed(rc_failed, error="boom")

    reader = ManifestReader(path)
    assert reader.completed_run_ids() == {rc_done.run_id}
    failed = reader.failed_rows()
    assert len(failed) == 1 and failed[0].error == "boom"
    done = reader.done_rows()
    assert done[0].checkpoint == "ck/done.zip" and done[0].gate_score == 0.7


def test_resume_skips_done(tmp_path):
    """run_campaign skips run_ids already 'done' in a pre-seeded manifest."""
    out_dir = str(tmp_path)
    spec = _tiny_spec()
    runs = spec.expand()

    # Pre-seed the manifest: first run already done.
    path = str(tmp_path / "manifest.jsonl")
    writer = ManifestWriter(path)
    writer.mark_done(runs[0], checkpoint="ck/pre.zip", wall_clock_s=0.5)

    trained: list[str] = []

    def fake_train(config, curriculum, *, num_players, phases):
        trained.append(curriculum.run_name)
        return (object(), f"ck/{curriculum.run_name}.zip")

    run_campaign(spec, out_dir=out_dir, train_fn=fake_train, resume=True)

    # Only the not-yet-done run should have trained.
    assert runs[0].curriculum.run_name not in trained
    assert runs[1].curriculum.run_name in trained
    assert ManifestReader(path).completed_run_ids() == {runs[0].run_id, runs[1].run_id}


def test_failure_isolation(tmp_path):
    """A stubbed trainer raising on one config marks it failed; the rest run."""
    out_dir = str(tmp_path)
    spec = SweepSpec(
        axes=(FactorAxis("ppo.shaping_weight", (0.0, 0.05, 0.1)),), seeds=(0,)
    )
    runs = spec.expand()
    bad = runs[1].run_id

    def fake_train(config, curriculum, *, num_players, phases):
        if curriculum.run_name == f"campaign_{bad}":
            raise RuntimeError("kaboom")
        return (object(), f"ck/{curriculum.run_name}.zip")

    manifest = run_campaign(spec, out_dir=out_dir, train_fn=fake_train)
    reader = ManifestReader(manifest)
    assert reader.completed_run_ids() == {runs[0].run_id, runs[2].run_id}
    failed = {r.run_id for r in reader.failed_rows()}
    assert failed == {bad}
    assert "kaboom" in reader.failed_rows()[0].error


# ===========================================================================
# Matchup sampling (§10)
# ===========================================================================


@pytest.mark.parametrize("n", [5, 12, 25])
def test_each_contender_meets_K(n):
    """Every contender appears in >= K size-4 distinct-member fields.

    K is capped at what the distinct-field space allows: with ``n`` contenders
    and 4 seats each contender can appear in at most ``C(n-1, 3)`` distinct
    fields, so the test asks for a feasible K (and the sampler returns that many
    when the target exceeds the cap).
    """
    contenders = [f"c{i}" for i in range(n)]
    feasible = math.comb(n - 1, 3)  # max distinct fields one contender can be in
    K = min(6, feasible)
    fields = balanced_fields(contenders, num_players=4, fields_per_contender=K, seed=0)
    counts = {c: 0 for c in contenders}
    for field in fields:
        assert len(field) == 4
        assert len(set(field)) == 4  # distinct members
        for c in field:
            counts[c] += 1
    assert all(counts[c] >= K for c in contenders), counts


def test_sampling_deterministic():
    """Same (contenders, K, seed) -> identical field list."""
    contenders = [f"c{i}" for i in range(10)]
    a = balanced_fields(contenders, num_players=4, fields_per_contender=5, seed=42)
    b = balanced_fields(contenders, num_players=4, fields_per_contender=5, seed=42)
    assert a == b


def test_seat_rotation_balanced():
    """Across a field's rotations, every member occupies every seat once."""
    field = ["a", "b", "c", "d"]
    rotations = seat_rotations(field, 4)
    assert len(rotations) == 4
    for seat in range(4):
        occupants = {rot[seat] for rot in rotations}
        assert occupants == set(field), seat


def test_far_fewer_than_exhaustive():
    """For n=25, sampled field count << C(25,4)."""
    contenders = [f"c{i}" for i in range(25)]
    fields = balanced_fields(contenders, num_players=4, fields_per_contender=30, seed=0)
    assert len(fields) < math.comb(25, 4)
    # And it is not pathologically tiny: enough to cover the 25 at K=30.
    assert len(fields) >= 25 * 30 // 4


# ===========================================================================
# Per-game persistence + recompute (§10)
# ===========================================================================


def _outcome(game_index, finish_order, types):
    rank_of = {pid: i + 1 for i, pid in enumerate(finish_order)}
    players = tuple(
        PlayerOutcome(
            player_id=pid, name=f"p{pid}", agent_type=types[pid],
            finish_position=rank_of[pid], final_lap=1, final_position=0,
            heat_remaining=0,
        )
        for pid in sorted(types)
    )
    return GameOutcome(
        game_index=game_index, seed=game_index, num_players=len(finish_order),
        winner_id=finish_order[0], winner_name=f"p{finish_order[0]}",
        finish_order=finish_order, total_rounds=1, players=players,
    )


def _synthetic_outcomes(n=20):
    types = {0: "Strong", 1: "Weak"}
    out = []
    for gi in range(n):
        order = (1, 0) if gi % 5 == 0 else (0, 1)
        out.append(_outcome(gi, order, types))
    return out


def test_outcomes_jsonl_roundtrip(tmp_path):
    """jsonl write+reload preserves outcomes; ELO byte-identical (recompute, §7.4)."""
    outcomes = _synthetic_outcomes(20)
    path = str(tmp_path / "games.jsonl")
    write_outcomes_jsonl(outcomes, path)
    reloaded = read_outcomes_jsonl(path)

    assert reloaded == outcomes  # frozen dataclass equality, field-for-field
    r_orig = compute_elo(outcomes, bootstrap=20, bootstrap_seed=1)
    r_reload = compute_elo(reloaded, bootstrap=20, bootstrap_seed=1)
    assert {k: v.rating for k, v in r_orig.items()} == {
        k: v.rating for k, v in r_reload.items()
    }
    assert {k: v.rating_ci for k, v in r_orig.items()} == {
        k: v.rating_ci for k, v in r_reload.items()
    }


def test_recompute_changes_rating_system_only(tmp_path, monkeypatch):
    """Re-rating a stored outcome set replays NO race (no run_batch call)."""
    import heat.simulation.runner as runner_mod

    outcomes = _synthetic_outcomes(16)
    path = str(tmp_path / "games.jsonl")
    write_outcomes_jsonl(outcomes, path)

    def _boom(*a, **k):
        raise AssertionError("run_batch must not be called during recompute")

    monkeypatch.setattr(runner_mod, "run_batch", _boom)
    from heat.simulation.stats import compute_trueskill

    reloaded = read_outcomes_jsonl(path)
    assert compute_elo(reloaded, bootstrap=0) is not None
    assert compute_trueskill(reloaded) is not None


# ===========================================================================
# Attribution (§10)
# ===========================================================================


def _ladder_from_run_ratings(run_ratings):
    """A TrueSkillRating mapping for synthetic run_id -> mu values."""
    return {
        run_id: TrueSkillRating(agent_type=run_id, mu=mu, sigma=1.0, games=100)
        for run_id, mu in run_ratings.items()
    }


def test_grouped_diff_on_synthetic():
    """A factor adding +R at level L1 -> grouped diff recovers ~+R; noise ~0."""
    # Construct rows: factor 'A' adds +10 at level 1; factor 'B' (irrelevant).
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for rep in range(3):
                rating = 100.0 + (10.0 if a == 1 else 0.0) + 0.01 * rep
                rows.append({"A": a, "B": b, "rating": rating})
    diffs_a = grouped_diffs(rows, factor="A", rating_key="rating")
    mean_0 = diffs_a["0"][0]
    mean_1 = diffs_a["1"][0]
    assert mean_1 - mean_0 == pytest.approx(10.0, abs=0.5)

    diffs_b = grouped_diffs(rows, factor="B", rating_key="rating")
    # B is irrelevant: the two level means are ~equal (diff straddles 0).
    assert abs(diffs_b["1"][0] - diffs_b["0"][0]) < 0.5


def test_ols_recovers_known_effect():
    """An additive synthetic design -> OLS coefficients recover planted effects."""
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            # rating = 100 + 10*A + 4*B (perfectly additive, no noise).
            rows.append({"A": a, "B": b, "rating": 100.0 + 10.0 * a + 4.0 * b})
    coefs = ols_effects(rows, factors=["A", "B"], rating_key="rating")
    # Level 1 vs baseline level 0 for each factor.
    assert coefs["A"]["1"][0] == pytest.approx(10.0, abs=1e-6)
    assert coefs["B"]["1"][0] == pytest.approx(4.0, abs=1e-6)


def test_join_ladder_sorted_and_ci():
    """join_ladder ranks best-first and carries a CI band."""
    ratings = {
        "lo": EloRating(agent_type="lo", rating=1400.0, games=10, rating_ci=(1380.0, 1420.0)),
        "hi": EloRating(agent_type="hi", rating=1600.0, games=10, rating_ci=(1580.0, 1620.0)),
    }
    ladder = join_ladder(ratings, rating_attr="rating")
    assert [e.label for e in ladder] == ["hi", "lo"]
    assert ladder[0].rating_ci_lo == 1580.0


def test_build_attribution_joins_manifest_and_ladder(tmp_path):
    """build_attribution joins done manifest rows to ratings + estimates effects."""
    path = str(tmp_path / "manifest.jsonl")
    writer = ManifestWriter(path)
    spec = SweepSpec(axes=(FactorAxis("ppo.shaping_weight", (0.0, 0.05)),), seeds=(0, 1))
    runs = spec.expand()
    for rc in runs:
        writer.mark_started(rc)
        writer.mark_done(rc, checkpoint=f"ck/{rc.run_id}.zip", wall_clock_s=1.0)

    # Synthetic ratings: shaping 0.05 is +20 mu over 0.0.
    run_ratings = {}
    for rc in runs:
        run_ratings[rc.run_id] = 100.0 + (20.0 if rc.factors["ppo.shaping_weight"] == 0.05 else 0.0)
    ratings = _ladder_from_run_ratings(run_ratings)

    attr = build_attribution(
        ManifestReader(path).rows(), ratings,
        factors=["ppo.shaping_weight"], rating_attr="mu",
    )
    assert len(attr.ladder) == len(runs)
    eff = next(e for e in attr.effects if e.factor == "ppo.shaping_weight")
    # Grouped diff recovers the +20.
    assert eff.group_means["0.05"][0] - eff.group_means["0.0"][0] == pytest.approx(20.0, abs=0.5)


def test_report_states_caveats():
    """attribution.md carries the §9.3 honesty prose (regression guard)."""
    ratings = _ladder_from_run_ratings({"r0": 100.0, "r1": 120.0})
    attr = Attribution(ladder=join_ladder(ratings, rating_attr="mu"), effects=[], run_ratings={})
    md = render_attribution_md(attr)
    lowered = md.lower()
    assert "observational" in lowered
    assert "causal" in lowered
    assert "confidence interval" in lowered or "cis" in lowered or "ci" in lowered
    assert "interact" in lowered
    assert "imbalance" in lowered


# ===========================================================================
# Contract guard (§10) -- 8D must not touch the codec.
# ===========================================================================


def test_codec_unchanged_8d():
    """8D never edits spaces.py: the obs/action/codec contract is frozen."""
    assert spaces.OBS_DIM == 104
    assert spaces.ACTION_DIM == 516
    assert spaces.CODEC_VERSION == 2
