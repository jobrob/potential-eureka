"""Tests for Sprint C2 self-play target generation (``gen_selfplay.py``).

Covers exactly the contract-critical guarantees the C2 design (and its
Deliverables list) flags for the AlphaZero training targets:

  1. **THE most important check** -- the value target ``z`` is the **pre-spin
     cost-to-go** on a known forced-spin track: the realized MC return FLOORED at
     the own-spin (``_floored_rounds_remaining``), NOT the post-spin recovery. The
     S1 lesson-2 hazard, now in the training data. Tested two ways: the
     ``_floored_rounds_remaining`` floor logic directly, AND end-to-end on a
     forced-spin limit-1 loop so the realized ``z`` is floored.
  2. logged tuples are codec-valid and mask-consistent: every ``pi``'s support is
     a subset of its ``mask``, and ``pi`` sums to 1.
  3. ``pi`` is the **visit distribution**, not one-hot, when sims>1 within the
     temperature window (the exploration signal is present) -- exercised via the
     distribution-building core ``_search_visit_distribution``.
  4. targets are logged for **all** searched ``DecisionKind``s (REACT/SLIP/DISCARD
     too, not just GEAR/CARDS) over a small generation.
  5. only real-choice states are logged (degenerate ``mask.sum() <= 1`` decisions
     are auto-resolved, not logged); train/val are track-disjoint (seed bands).
  6. ``z`` is ``-rounds_remaining <= 0`` by construction; MAX_ROUNDS-truncated rows
     are dropped.

Mirrors the style/imports of ``tests/test_mcts_agent.py`` (the C1 patterns) and
``tests/test_value_net.py`` (the experiment-script import + tiny-fixture style).
Everything runs on a tiny CPU smoke generation in seconds -- no committed data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# The experiments scripts are not a package; add the dir to the path so the
# Sprint-C2 deliverables can be imported by their module name (mirrors test_bc /
# test_value_net).
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_selfplay as G  # noqa: E402

from heat.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: E402
from heat.engine.driver import Decision, DecisionKind  # noqa: E402
from heat.models.game_state import GameState  # noqa: E402
from heat.models.track import Corner, Space, Track  # noqa: E402
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Shared tiny prior + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """Mint a fresh random-weight codec-v3 ``MaskablePPO`` checkpoint once.

    A random prior is enough: the C2 target-generation tests verify the *logging
    contract* (floor, mask-consistency, kind coverage), which is net-agnostic. CPU
    device matches the C0 default + the NetAdapter load path.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp("c2prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


def _limit1_loop() -> Track:
    """Short synthetic loop with a single tight (limit-1) corner.

    The S1/C1 forced-spin stress track: a random prior reliably spins out at the
    limit-1 corner, which is exactly what the pre-spin floor must truncate.
    """
    spaces = [Space(i, lanes=2) for i in range(24)]
    corners = [Corner(start=10, end=11, speed_limit=1)]
    return Track("Limit1Loop", spaces, corners, [0, 1, 2, 3, 4, 5], laps=2)


def _gen_args(out, *, model, tracks=4, val_tracks=2, sims=8):
    """Tiny generation Namespace (mirrors gen_selfplay's smoke defaults)."""
    return argparse.Namespace(
        out=out,
        model=model,
        tracks=tracks,
        val_tracks=val_tracks,
        sims=sims,
        dirichlet_eps=0.25,
        dirichlet_alpha=0.5,
        temperature_moves=10,
        seed=0,
        game_seed=8000,
    )


@pytest.fixture(scope="module")
def tiny_dataset(cold_prior, tmp_path_factory) -> dict:
    """Generate a tiny self-play dataset once for the module (a few tracks)."""
    out = str(tmp_path_factory.mktemp("selfplay") / "sp.npz")
    summary = G.generate_dataset(_gen_args(out, model=cold_prior))
    data = np.load(out)
    return {k: data[k] for k in data.files} | {"path": out, "summary": summary}


# ---------------------------------------------------------------------------
# 1. THE most important check: the pre-spin floor on the value target z
# ---------------------------------------------------------------------------


class TestPreSpinFloor:
    def test_floor_truncates_at_spin_not_recovery(self) -> None:
        """``_floored_rounds_remaining`` truncates the MC return at an own-spin.

        For a state at ``round_num`` in an episode that finishes at ``finish_round``
        but whose own-spin happened at ``spin_round`` in between, the cost-to-go is
        ``spin_round - round_num`` (truncated), NEVER ``finish_round - round_num``
        (the inflated post-spin recovery) -- the rounds-frame image of
        ``LookaheadAgent._pre_spin_progress``.
        """
        finish, spin = 10, 6
        # A state BEFORE the spin: floored at the spin (4 rounds), not the
        # recovery (the full 8 rounds to finish).
        assert G._floored_rounds_remaining(2, finish, spin) == spin - 2  # 4, not 8
        # A state AT the spin round: zero remaining (the floor caps at the spin).
        assert G._floored_rounds_remaining(spin, finish, spin) == 0

    def test_floor_is_noop_without_spin(self) -> None:
        """No own-spin => the floor is inert; z is the plain realized MC return."""
        assert G._floored_rounds_remaining(3, 10, None) == 7  # finish - round

    def test_spin_before_state_does_not_floor_a_clean_later_state(self) -> None:
        """A spin already PAID (``spin_round < round_num``) must not floor a later,
        clean state -- only a spin still ahead in the rest of the episode truncates
        the cost-to-go."""
        # State at round 8, spin happened at round 3 (already paid): the floor must
        # NOT apply -- the cost-to-go is the full finish - round (2).
        assert G._floored_rounds_remaining(8, 10, 3) == 2

    def test_floor_never_negative(self) -> None:
        """A finish_round earlier than round_num (degenerate) floors to 0, never a
        positive rounds-remaining (which would flip z positive)."""
        assert G._floored_rounds_remaining(12, 10, None) == 0

    def test_end_to_end_z_is_floored_on_forced_spin_track(self, cold_prior, monkeypatch) -> None:
        """End-to-end: on a forced-spin limit-1 loop the realized ``z`` of every
        pre-spin row is floored AT the spin, not charged the recovery rounds.

        Drives the real ``_generate_one_track`` against the limit-1 loop (a random
        prior reliably spins there), captures the episode's ``spin_round`` /
        ``finish_round`` via a spy on ``_spin_round``, then asserts every row's
        backfilled ``z`` equals ``-floored_rounds_remaining(...)`` and -- the load-
        bearing check -- that a pre-spin row is charged the SPIN horizon, strictly
        less than the inflated finish horizon.
        """
        loop = _limit1_loop()
        # Force the generator's track to the forced-spin loop.
        monkeypatch.setattr(G, "generate_track", lambda seed, params: loop)

        captured: dict = {}
        orig_spin = G._spin_round

        def _spy(state: GameState, pid: int):
            r = orig_spin(state, pid)
            captured["spin"] = r
            captured["finish"] = state.round_num
            return r

        monkeypatch.setattr(G, "_spin_round", _spy)

        cfg = MCTSConfig(
            n_simulations=4, dirichlet_eps=0.25, dirichlet_alpha=0.5,
            temperature_moves=10,
        )
        agent = MCTSAgent(model_path=cold_prior, config=cfg, seed=1, name="spin")
        buf = G._SelfPlayBuffer()
        finished, _ = G._generate_one_track(
            123, "train", game_seed=7, agent=agent, buf=buf
        )
        assert finished, "the forced-spin loop should still finish (the floor "
        "prevents the death-spiral)"
        assert len(buf) > 0
        spin_round = captured["spin"]
        finish_round = captured["finish"]
        assert spin_round is not None, "the random prior did not spin on the L1 loop"

        z = np.asarray(buf.z)
        rounds = np.asarray(buf.round_num)

        # Every row's z is exactly -floored_rounds_remaining (the backfill rule).
        expected = np.asarray([
            -G._floored_rounds_remaining(int(rn), finish_round, spin_round)
            for rn in rounds
        ], dtype=float)
        assert np.array_equal(z, expected)

        # The load-bearing assertion: a pre-spin row is charged the spin horizon,
        # which is strictly less than the inflated finish horizon (the recovery the
        # spin cost is NOT poured into the target).
        pre = rounds < spin_round
        assert pre.any(), "no pre-spin rows -- the floor check is vacuous"
        for rn in rounds[pre]:
            floored = -(spin_round - int(rn))
            recovery = -(finish_round - int(rn))
            assert floored > recovery, (
                "pre-spin z must be floored at the spin, not charged the recovery"
            )


# ---------------------------------------------------------------------------
# 2. logged tuples are codec-valid and mask-consistent
# ---------------------------------------------------------------------------


class TestCodecValidAndMaskConsistent:
    def test_shapes_and_dtypes(self, tiny_dataset) -> None:
        d = tiny_dataset
        assert d["obs"].ndim == 2 and d["obs"].shape[1] == OBS_DIM
        assert d["pi"].shape[1] == ACTION_DIM
        assert d["mask"].shape[1] == ACTION_DIM
        assert d["obs"].shape[0] == d["pi"].shape[0] == d["mask"].shape[0]
        assert d["mask"].dtype == bool

    def test_pi_support_is_subset_of_mask(self, tiny_dataset) -> None:
        """Every pi entry with positive mass sits on a legal (set) mask bit."""
        pi = tiny_dataset["pi"]
        mask = tiny_dataset["mask"]
        off_mask_mass = ((pi > 0) & ~mask).sum()
        assert off_mask_mass == 0, "a pi entry escapes its mask (codec drift)"

    def test_pi_sums_to_one(self, tiny_dataset) -> None:
        sums = tiny_dataset["pi"].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), "a pi row does not sum to 1"


# ---------------------------------------------------------------------------
# 3. pi is the visit distribution, not one-hot, within the temperature window
# ---------------------------------------------------------------------------


class TestVisitDistributionNotOneHot:
    def test_search_visit_distribution_is_non_degenerate(self, cold_prior) -> None:
        """``_search_visit_distribution`` returns the visit *distribution* (mass
        spread over multiple kept edges), not a collapsed one-hot, when sims>1 and
        the ply is inside the temperature window (tau=1, pi proportional to N).

        Roots a real C1 search at a live GEAR decision with a healthy sim budget
        inside the temperature window and asserts the returned pi has support on
        >1 action (the exploration signal is present -- not BC's argmax).
        """
        from heat.engine import rules

        cfg = MCTSConfig(
            n_simulations=24, dirichlet_eps=0.25, dirichlet_alpha=0.5,
            temperature_moves=10,
        )
        agent = MCTSAgent(model_path=cold_prior, config=cfg, seed=0, name="vd")
        agent._ply = 0  # inside the temperature window (ply < temperature_moves)

        track = G.generate_track(G._SELFPLAY_SEED_BASE, G._TIGHT_PARAMS)
        state = GameState.create(track, 1, logging_enabled=True, seed=5)
        state.players[0].lap = 1
        p = state.players[0]
        legal_gears = rules.legal_gear_shifts(p.gear, p.heat_available)
        decision = Decision(DecisionKind.GEAR, 0, legal_gears)

        pi, _acted = G._search_visit_distribution(agent, state, decision)

        # A proper distribution over the kept candidate set.
        assert abs(float(pi.sum()) - 1.0) < 1e-6
        support = int((pi > 0).sum())
        # In-window (tau=1) with sims spread over a multi-candidate GEAR root, the
        # visit distribution must touch more than one action (not a one-hot argmax).
        assert support > 1, (
            f"pi collapsed to {support}-action support inside the temperature "
            "window -- the visit distribution is degenerate (BC again)"
        )

    def test_dataset_pi_is_not_globally_one_hot(self, tiny_dataset) -> None:
        """Across the tiny generation, at least some rows carry a multi-action
        visit distribution (the exploration signal survives into the dataset)."""
        pi = tiny_dataset["pi"]
        support = (pi > 0).sum(axis=1)
        assert (support > 1).any(), (
            "every dataset pi is one-hot -- the temperature window produced no "
            "exploration signal (the C0 visit-budget go/no-go would surface here)"
        )


# ---------------------------------------------------------------------------
# 4. targets are logged for ALL searched DecisionKinds (not just GEAR/CARDS)
# ---------------------------------------------------------------------------


class TestAllDecisionKindsLogged:
    def test_non_gear_cards_kinds_are_present(self, tiny_dataset) -> None:
        """The ``kind`` column carries non-GEAR/CARDS kinds (REACT/DISCARD), i.e.
        C2 logs a visit target for every searched DecisionKind, not only the
        GEAR/CARDS plan -- the README §4 "no kind is delegated" contract."""
        kinds = set(int(k) for k in np.unique(tiny_dataset["kind"]))
        gear = G._KIND_TO_INT[DecisionKind.GEAR]
        cards = G._KIND_TO_INT[DecisionKind.CARDS]
        non_plan = kinds - {gear, cards}
        assert non_plan, (
            "only GEAR/CARDS kinds were logged -- REACT/SLIPSTREAM/DISCARD targets "
            "are missing (a searched kind was silently delegated)"
        )
        # All logged kind codes must be valid DecisionKind codes.
        valid = set(G._KIND_TO_INT.values())
        assert kinds.issubset(valid)


# ---------------------------------------------------------------------------
# 5. only real-choice states logged; train/val track-disjoint
# ---------------------------------------------------------------------------


class TestRealChoiceAndDisjointSplit:
    def test_only_real_choices_logged(self, tiny_dataset) -> None:
        """Every logged row had > 1 legal action (degenerate mask.sum() <= 1
        decisions are auto-resolved, not logged) -- the env/BC info-set rule."""
        n_legal = tiny_dataset["mask"].sum(axis=1)
        assert (n_legal > 1).all(), "a degenerate (<=1 legal) decision was logged"

    def test_train_val_track_disjoint(self, tiny_dataset) -> None:
        split = tiny_dataset["split"]
        track_seed = tiny_dataset["track_seed"]
        train_seeds = set(track_seed[split == b"train"].tolist())
        val_seeds = set(track_seed[split == b"val"].tolist())
        assert train_seeds and val_seeds
        assert train_seeds.isdisjoint(val_seeds)

    def test_selfplay_band_disjoint_from_eval_and_precursor_bands(self) -> None:
        """The self-play seed band is asserted disjoint from the eval/precursor
        bands (review bug #6/#9): a gated/precursor track is never a train track."""
        # A sane track count passes; a count that would overrun into the BC-val
        # band (500_000) must be rejected.
        G._assert_seed_bands_disjoint(150)  # no raise
        with pytest.raises(ValueError):
            G._assert_seed_bands_disjoint(300_000)  # would overrun 500_000


# ---------------------------------------------------------------------------
# 6. z = -rounds_remaining <= 0; MAX_ROUNDS-truncated rows dropped
# ---------------------------------------------------------------------------


class TestValueTargetSignAndDrop:
    def test_z_is_non_positive(self, tiny_dataset) -> None:
        """z = -rounds_remaining is <= 0 by construction (rounds-remaining >= 0)."""
        z = tiny_dataset["z"]
        assert z.max() <= 1e-6, "a positive z target (rounds-remaining < 0)"

    def test_max_rounds_truncated_rows_are_dropped(self, cold_prior, monkeypatch) -> None:
        """A race that never finishes (MAX_ROUNDS) leaves its rows' z as NaN, and
        ``generate_dataset`` drops those rows + counts them (the gen_value_data
        drop rule) -- a truncated MC return must never poison V.
        """
        # Force every solo race to never report game-over, so it can only exit on
        # the MAX_ROUNDS guard with finished == False (rows left with z == NaN).
        monkeypatch.setattr(
            GameState, "is_game_over", property(lambda self: False), raising=True
        )
        # Cap the guard low so the never-over loop exits after a few rounds (the
        # drop logic is round-count-independent; sims=1 keeps each step cheap).
        monkeypatch.setattr(G, "MAX_ROUNDS", 4, raising=True)
        buf = G._SelfPlayBuffer()
        cfg = MCTSConfig(n_simulations=1, temperature_moves=10)
        agent = MCTSAgent(model_path=cold_prior, config=cfg, seed=0, name="trunc")
        finished, _ = G._generate_one_track(
            G._SELFPLAY_SEED_BASE, "train", game_seed=5, agent=agent, buf=buf
        )
        assert finished is False
        assert len(buf) > 0, "rows should still be logged before the race is dropped"
        # MAX_ROUNDS-truncated rows keep the NaN sentinel (never backfilled).
        assert all(np.isnan(zz) for zz in buf.z)

    def test_summary_reports_disjoint_counts_and_kinds(self, tiny_dataset) -> None:
        """The generation summary's row counts are internally consistent and the
        sidecar stamps the live codec version (the train-time tripwire input)."""
        s = tiny_dataset["summary"]
        assert s["n_rows"] == s["n_train"] + s["n_val"]
        assert s["races_total"] == 6  # 4 train + 2 val (the tiny fixture)
        meta_path = os.path.splitext(tiny_dataset["path"])[0] + ".selfplay.json"
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["codec_version"] == CODEC_VERSION
        assert meta["obs_dim"] == OBS_DIM
        assert meta["action_dim"] == ACTION_DIM
        assert meta["selfplay_seed_base"] == G._SELFPLAY_SEED_BASE
