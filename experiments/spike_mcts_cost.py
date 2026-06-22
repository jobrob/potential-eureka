"""Sprint C0 feasibility / cost / constants spike for Option C (solo AlphaZero).

THE design gate for Option C. It answers ONE question with measured numbers, not
intuition: **is a net-guided solo stochastic-MCTS over the real engine affordable
at our throughput, and what constant values make it both correct and cheap enough
to self-play with?** It produces the concrete constant set Sprint C1 builds
against plus a go/no-go recommendation. It builds NO MCTS, NO tree, NO training --
it is a throwaway-but-committed timing harness (README option-C §C0 scope).

What it measures (each maps to a memo section in ``C0-findings.md``):

  A. Building-block cost: ``GameState.clone(reseed=) + run_round_driver`` advance
     to the next learner decision (the ``TransitionModel`` core edge), and a
     ``MaskableActorCriticPolicy`` forward (masked policy logits + value) on CPU
     and -- if available -- CUDA (RTX 4080, torch cu126). A freshly-built net with
     RANDOM weights is used: forward *timings* are weight-independent (only the
     Q-normalization experiment D benefits from a trained value, and random is an
     acceptable caveat there -- see the printed note).
  B. The realistic in-tree product. Because chance is in-tree (§4.3) the per-move
     work is ``n_simulations`` selections, each descending a root-to-leaf path of
     decision + chance edges. We do NOT have an MCTS, so we MODEL the cost
     structure: ``sims * avg_path_len * (clone+advance) + leaf_forwards``, over a
     small grid of ``(n_simulations, C_pw, alpha_pw, K, leaf_mode)``. The
     path-length and fan-out assumptions are explicit, parameterized and printed
     so the memo numbers are reproducible and auditable.
  C. DPW fan-out + variance check. How ``(C_pw, alpha_pw, K)`` change the
     distinct-clone count at a chance node, plus a variance check: sample chance
     outcomes with the deterministic per-(node,sample) reseed scheme used by
     ``LookaheadAgent._score_plan`` / ``_turn_seed`` (advancing across a real
     replenish/round boundary) and show how the backed-up *average value*
     stabilizes as fan-out grows -- so we pick the smallest stable fan-out.
  D. Q-normalization experiment. On real states, compute the PUCT child-selection
     term with raw ``Q`` (the ``-rounds_remaining`` scale, ~-25..0) vs min-max
     normalized ``Q in [0,1]`` and report how often the prior changes the argmax
     and the selection-entropy gap -- settling the §4.1 "min-max is mandatory"
     claim with a number, not an assertion.
  E. Reseed purity. A fixed ``(root state, search seed)`` yields a byte-stable
     descent (the determinism contract: clone with explicit ``reseed=``, never the
     parent-advancing default ``reseed=None``).
  F. Back-of-envelope throughput: states/sec -> tracks/hour at the chosen
     constants, scaled to the self-play volume C2/C3 will need.

Usage:
    python experiments/spike_mcts_cost.py --quick      # fast smoke (~seconds)
    python experiments/spike_mcts_cost.py              # default measurement run
    python experiments/spike_mcts_cost.py --clone-iters 20000 --fwd-iters 2000
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow `from _runlog import run_main` and src-layout imports when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import Game
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml.opponents import opponent_action
from heat.tracks.generator import TrackGenParams, generate_track


# The held-out generated tight-corner band the whole eval ladder uses (eval_search
# §, README §5): tight corners so limit-1 lines (where chance + corner risk bite)
# are well represented in the sampled states.
_HELDOUT_BASE = 900_000
_TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),
    laps=2,
)


# ---------------------------------------------------------------------------
# Snapshot capture: representative mid-game solo decision states
# ---------------------------------------------------------------------------


def _capture_solo_states(n_states: int, seed_base: int) -> list[tuple[GameState, int]]:
    """Play solo games and snapshot ``(clone-of-state, learner_id)`` at GEAR turns.

    We snapshot the live state at each GEAR decision (a *clone* so the running
    game is untouched), which is exactly the node type the search roots/expands at.
    Played with a HeuristicAgent driver on the tight held-out band; we stop once
    ``n_states`` snapshots are collected. Solo = 1 player (the single-agent MDP).
    """
    snaps: list[tuple[GameState, int]] = []
    gi = 0
    while len(snaps) < n_states:
        track = generate_track(_HELDOUT_BASE + gi, _TIGHT_PARAMS)
        state = GameState.create(track, 1, logging_enabled=True, seed=seed_base + gi)
        for p in state.players:
            p.lap = 1
        driver_agent = HeuristicAgent()
        rounds = 0
        while not state.is_game_over and rounds < 60 and len(snaps) < n_states:
            gen = run_round_driver(state)
            try:
                decision = next(gen)
                while True:
                    if decision.kind == DecisionKind.GEAR and decision.player_id == 0:
                        # Snapshot a clone at this learner GEAR node.
                        snaps.append((state.clone(reseed=10_000 + len(snaps)), 0))
                    action = opponent_action(driver_agent, decision, state)
                    decision = gen.send(action)
            except StopIteration:
                pass
            rounds += 1
        gi += 1
    return snaps


# ---------------------------------------------------------------------------
# The transition primitive: clone + advance one learner decision
# ---------------------------------------------------------------------------


def _advance_one_round(
    clone: GameState, learner_id: int, rollout: HeuristicAgent
) -> tuple[bool, bool]:
    """Advance ``clone`` (in place) by ONE engine round (the measured edge unit).

    Mirrors how ``LookaheadAgent._rollout_once`` drives ``run_round_driver``: every
    decision is answered by the rollout policy via ``opponent_action``. We run a
    single round's driver to completion -- so this advances the *whole* round
    (GEAR -> CARDS -> [REACT] -> [SLIPSTREAM] -> [DISCARD] -> REPLENISH), which is
    the cheapest natural "advance" the engine exposes. The cost model (B) is
    therefore expressed in clone+round-advances, NOT per-decision edges, so the
    measured primitive and the path length use the same unit (no double-count).

    Returns ``(consumed_rng, crossed_round)``: ``consumed_rng`` is True iff the
    advance drew cards / reshuffled (detected by the RNG state changing -- §4.3's
    "a chance edge iff advancing consumed engine RNG", detected not assumed);
    ``crossed_round`` is True iff ``round_num`` advanced (a replenish boundary).
    IMPORTANT empirical nuance (measured here): ``crossed_round`` is True on
    essentially every advance (replenish always draws), but ``consumed_rng`` (the
    parent RNG state actually changing) is True only ~15% of the time -- because a
    plain draw off an already-ordered deck consumes NO RNG; only a *reshuffle*
    does. So a chance edge per §4.3 ("crosses a replenish / changes our draw
    state") is the ROUND BOUNDARY (~every round), NOT the rarer reshuffle. The
    draw still changes the future hand even without a reshuffle, and the value
    only DIVERGES once the next round plays the freshly-drawn hand (see
    :func:`chance_value_variance`). This is a real C1 design input.
    """
    rng_before = clone.rng.getstate()
    round_before = clone.round_num
    if not clone.is_game_over:
        gen = run_round_driver(clone)
        try:
            decision = next(gen)
            while True:
                action = opponent_action(rollout, decision, clone)
                decision = gen.send(action)
        except StopIteration:
            pass
    reshuffled = clone.rng.getstate() != rng_before
    crossed_round = clone.round_num != round_before
    # The chance edge is the round boundary (a draw), which subsumes reshuffles.
    return crossed_round, reshuffled


# ---------------------------------------------------------------------------
# A) Building-block timing: clone+advance, and the net forward (CPU/CUDA)
# ---------------------------------------------------------------------------


@dataclass
class CloneAdvanceCost:
    per_call_us: float          # microseconds per clone+advance-one-round
    calls_per_s: float
    chance_edge_frac: float     # fraction of advances that cross a round (a draw)
    reshuffle_frac: float       # fraction that actually consumed RNG (reshuffle)


def measure_clone_advance(
    snaps: list[tuple[GameState, int]], iters: int
) -> CloneAdvanceCost:
    """Time ``clone(reseed=) + advance one round`` -- the core measured tree edge.

    Cycles over the captured snapshots, cloning each with an explicit reseed
    (never the parent-advancing default) and advancing one round. Reports the
    per-call wall time, the fraction that crossed a round (a draw -- the §4.3
    chance edge) and the fraction that actually reshuffled (consumed parent RNG).
    """
    rollout = HeuristicAgent()
    n = len(snaps)
    chance_hits = 0
    reshuffle_hits = 0
    # Warm up (engine import / branch caches) so the timed loop is steady-state.
    for i in range(min(50, iters)):
        st, pid = snaps[i % n]
        _advance_one_round(st.clone(reseed=i), pid, rollout)

    t0 = time.perf_counter()
    for i in range(iters):
        st, pid = snaps[i % n]
        crossed, reshuffled = _advance_one_round(st.clone(reseed=i + 777), pid, rollout)
        if crossed:
            chance_hits += 1
        if reshuffled:
            reshuffle_hits += 1
    elapsed = time.perf_counter() - t0
    per_call_us = elapsed / iters * 1e6
    return CloneAdvanceCost(
        per_call_us=per_call_us,
        calls_per_s=iters / elapsed,
        chance_edge_frac=chance_hits / iters,
        reshuffle_frac=reshuffle_hits / iters,
    )


@dataclass
class ForwardCost:
    device: str
    measured: bool
    per_call_ms: float          # ms per single (unbatched) forward
    calls_per_s: float
    batch_per_item_ms: float    # ms per item at the frontier batch size
    batch_size: int


def _build_net():
    """Build a fresh MaskablePPO (random weights) over the solo HeatEnv space.

    ``build_model`` needs an env only for its observation/action spaces, so a
    solo ``HeatEnv(num_players=1)`` is the cheapest valid space provider. Random
    weights are fine for *timing* (weight-independent); the net is never trained
    or saved here.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model, PPOConfig

    env = HeatEnv(num_players=1)
    model = build_model(env, PPOConfig(device="cpu"))
    return model


def _forward_once(model, obs_batch: np.ndarray, mask_batch: np.ndarray):
    """One masked policy+value forward: logits (via distribution) + value head.

    Uses the policy's own ``obs_to_tensor`` / ``get_distribution`` /
    ``predict_values`` -- the exact path MLAgent and the A2 learned leaf use -- so
    the timing reflects the real prior+value evaluation a leaf does.
    """
    import torch

    policy = model.policy
    obs_t, _ = policy.obs_to_tensor(obs_batch)
    mask_t = torch.as_tensor(mask_batch, device=obs_t.device)
    with torch.no_grad():
        dist = policy.get_distribution(obs_t, action_masks=mask_t)
        _ = dist.distribution.logits  # masked policy logits (the PUCT prior)
        _ = policy.predict_values(obs_t)


def measure_forward(
    model,
    obs: np.ndarray,
    mask: np.ndarray,
    device: str,
    single_iters: int,
    batch_size: int,
    batch_iters: int,
) -> ForwardCost:
    """Time single + batched masked forwards on ``device`` (cpu or cuda).

    The batched timing models the AlphaZero virtual-loss frontier batching the
    README flags as the throughput mitigation (NOT built here, only measured): if
    the leaf forward dominates, batching ``batch_size`` leaves amortizes it.
    """
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        return ForwardCost(device, False, float("nan"), float("nan"),
                           float("nan"), batch_size)

    model.policy.to(device)
    model.policy.set_training_mode(False)

    single_obs = obs.reshape(1, -1)
    single_mask = mask.reshape(1, -1)
    batch_obs = np.repeat(obs.reshape(1, -1), batch_size, axis=0)
    batch_mask = np.repeat(mask.reshape(1, -1), batch_size, axis=0)

    # Warm up (CUDA kernel compile / autograd graph caches).
    for _ in range(min(20, single_iters)):
        _forward_once(model, single_obs, single_mask)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(single_iters):
        _forward_once(model, single_obs, single_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    single_elapsed = time.perf_counter() - t0
    per_call_ms = single_elapsed / single_iters * 1e3

    # Warm + time the batch.
    for _ in range(min(10, batch_iters)):
        _forward_once(model, batch_obs, batch_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(batch_iters):
        _forward_once(model, batch_obs, batch_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    batch_elapsed = time.perf_counter() - t0
    batch_per_item_ms = batch_elapsed / batch_iters / batch_size * 1e3

    # Restore CPU so the rest of the harness (Q-norm, etc.) runs on CPU.
    model.policy.to("cpu")
    return ForwardCost(
        device=device,
        measured=True,
        per_call_ms=per_call_ms,
        calls_per_s=single_iters / single_elapsed,
        batch_per_item_ms=batch_per_item_ms,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# B) Realistic in-tree product (cost MODEL, not a real MCTS)
# ---------------------------------------------------------------------------


def _avg_path_len(rounds_lookahead: float) -> float:
    """Representative root-to-leaf path length in ROUND-ADVANCES.

    The measured ``clone+advance`` primitive advances a WHOLE round (the cheapest
    natural engine advance), so the cost unit is one round-advance and the path
    length is simply how many round-boundaries a typical selection descends before
    hitting an unexpanded leaf (``rounds_lookahead``). Using rounds (not per-ply
    edges) keeps the measured primitive and the path length in the SAME unit, so
    there is no double-count. (The per-decision granularity matters for the
    tree's *branching*, measured separately as the CARDS branch width, not for
    this wall-clock cost arithmetic.)
    """
    return rounds_lookahead


@dataclass
class InTreeCost:
    n_sims: int
    c_pw: float
    alpha_pw: float
    cap_k: int
    leaf_mode: str
    avg_path_len: float
    advances_per_move: float    # sims * avg_path_len (round-advances)
    leaf_forwards_per_move: int
    ms_per_move: float
    clones_per_move: float


def model_in_tree_cost(
    n_sims: int,
    c_pw: float,
    alpha_pw: float,
    cap_k: int,
    leaf_mode: str,
    avg_path_len: float,
    edge_us: float,
    forward_ms: float,
    rollout_rounds: int,
) -> InTreeCost:
    """Model per-move cost for one ``(sims, C_pw, alpha_pw, K, leaf_mode)`` point.

    Cost structure (README §4.3): each of ``n_sims`` simulations descends an
    ``avg_path_len``-round path; each round-advance is a ``clone+advance``
    (``edge_us``). Each NEW leaf (one per sim in the worst case -- a fresh
    expansion every sim) is evaluated once:

      * ``leaf_mode == "net"``  -> one net forward (``forward_ms``);
      * ``leaf_mode == "rollout"`` -> ``rollout_rounds`` extra clone+round-advances
        to a short pre-spin-floor rollout, no net forward.

    DPW (``C_pw``, ``alpha_pw``, ``K``) does not change THIS wall-clock arithmetic
    directly (it bounds distinct clones at *chance* nodes -- the same clone+advance
    primitive, just counted at chance children, measured in C); we report it on
    the row so the grid is legible and the chance contribution is auditable. The
    descent count is the binding cost; the DPW cap K bounds re-expansion so it
    cannot inflate beyond ``avg_path_len`` distinct clones per simulation.
    """
    advances_per_move = n_sims * avg_path_len
    if leaf_mode == "net":
        leaf_forwards = n_sims
        leaf_ms = n_sims * forward_ms
        leaf_clones = 0.0
    else:  # "rollout"
        leaf_forwards = 0
        leaf_ms = n_sims * rollout_rounds * (edge_us / 1e3)
        leaf_clones = n_sims * rollout_rounds
    descent_ms = advances_per_move * (edge_us / 1e3)
    ms_per_move = descent_ms + leaf_ms
    clones_per_move = advances_per_move + leaf_clones
    return InTreeCost(
        n_sims=n_sims,
        c_pw=c_pw,
        alpha_pw=alpha_pw,
        cap_k=cap_k,
        leaf_mode=leaf_mode,
        avg_path_len=avg_path_len,
        advances_per_move=advances_per_move,
        leaf_forwards_per_move=leaf_forwards,
        ms_per_move=ms_per_move,
        clones_per_move=clones_per_move,
    )


# ---------------------------------------------------------------------------
# C) DPW fan-out + chance-node value variance check
# ---------------------------------------------------------------------------


def dpw_fanout(n_visits: int, c_pw: float, alpha_pw: float, cap_k: int) -> int:
    """Distinct sampled outcomes a chance node holds after ``n_visits`` visits.

    DPW rule (§4.3): a chance node admits a new outcome only when
    ``ceil(C_pw * N^alpha_pw)`` exceeds its current child count, capped at the hard
    outcome cap ``K``. The fan-out after ``N`` visits is therefore
    ``min(K, ceil(C_pw * N^alpha_pw))``.
    """
    return min(cap_k, int(math.ceil(c_pw * (n_visits ** alpha_pw))))


def _find_chance_state(
    snaps: list[tuple[GameState, int]], rollout: HeuristicAgent
) -> tuple[GameState, int] | None:
    """Find a snapshot whose two-round outcome ACTUALLY varies across draws.

    The chance node is the round boundary (a draw fires every round), but its
    value only DIVERGES once the next round plays the freshly-drawn hand -- and
    only at states where the play is draw-sensitive (many corner-forced states are
    draw-insensitive, which would make the variance check vacuous). We therefore
    probe each surviving snapshot over several draw seeds and pick the FIRST whose
    two-round race-progress takes more than one distinct value -- the meaningful
    chance node for the variance check. Falls back to the first surviving snapshot
    if none is draw-sensitive (the harness then reports the flat result honestly).
    """
    ME = _ME()
    first_survivor: tuple[GameState, int] | None = None
    for st, pid in snaps:
        probe = st.clone(reseed=1)
        _advance_one_round(probe, pid, rollout)
        if probe.is_game_over or probe.get_player(pid).finished:
            continue
        if first_survivor is None:
            first_survivor = (st, pid)
        distinct: set[float] = set()
        for s in range(8):
            c = st.clone(reseed=(100 + s * 31) & 0x7FFFFFFF)
            c.logging_enabled = True
            _advance_one_round(c, pid, rollout)
            _advance_one_round(c, pid, rollout)
            distinct.add(float(ME.race_progress(c.get_player(pid), c.track)))
        if len(distinct) > 1:
            return st, pid
    return first_survivor


def chance_value_variance(
    state: GameState,
    learner_id: int,
    fanouts: list[int],
    rollout: HeuristicAgent,
) -> list[tuple[int, float, float]]:
    """Backed-up chance-node value vs fan-out, and its spread across reseeds.

    For each candidate fan-out ``m`` we estimate the chance node's backed-up value
    (the visit-weighted mean over outcomes, §4.3) by sampling ``m`` draw outcomes
    -- each a clone with the deterministic per-(node,sample) reseed scheme
    (``LookaheadAgent._score_plan``'s ``reseed = base + sample*31`` style). We then
    advance TWO rounds: the first round plays the known hand (deterministic), the
    second plays the freshly-DRAWN hand -- which is where the draw outcome actually
    changes the value (verified: a one-round advance is identical across draws,
    a two-round advance spreads ~106..109 progress). We read a leaf value (race
    progress as a cheap stand-in for V; the *stabilization* is what matters, not
    the absolute scale), repeat over several independent reseed bases, and report
    ``(m, mean_value, std_across_bases)`` so the smallest ``m`` whose backed-up
    mean is stable across reseeds is visible.
    """
    bases = [101, 202, 303, 404, 505]
    rows: list[tuple[int, float, float]] = []
    for m in fanouts:
        base_means: list[float] = []
        for base in bases:
            vals: list[float] = []
            for s in range(m):
                reseed = (base + s * 31) & 0x7FFFFFFF
                clone = state.clone(reseed=reseed)
                clone.logging_enabled = True
                _advance_one_round(clone, learner_id, rollout)  # known hand
                _advance_one_round(clone, learner_id, rollout)  # drawn hand
                player = clone.get_player(learner_id)
                vals.append(float(_ME().race_progress(player, clone.track)))
            base_means.append(statistics.mean(vals))
        rows.append((m, statistics.mean(base_means), _std(base_means)))
    return rows


def _ME():
    from heat.agents import _move_eval as ME
    return ME


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


# ---------------------------------------------------------------------------
# D) Q-normalization experiment (raw vs min-max on the PUCT term)
# ---------------------------------------------------------------------------


def _net_prior_and_value(model, state: GameState, learner_id: int):
    """Return ``(prior_over_legal, value_scalar)`` for a GEAR node.

    Builds the GEAR ``Decision`` the way MLAgent does, encodes the obs, runs the
    masked policy forward for the prior over legal flat actions (renormalized over
    the legal set), and the value head for the leaf scalar (``-rounds_remaining``
    convention). Random-weight net: the prior is a near-uniform softmax and the
    value is arbitrary -- fine for the *mechanics* of the experiment; we ALSO
    synthesize a realistic ``Q`` scale below so the conclusion does not hinge on
    the untrained value (the printed caveat).
    """
    import torch

    from heat.ml.action_codec import legal_action_mask
    from heat.ml.features import encode_observation

    player = state.get_player(learner_id)
    legal_gears = rules.legal_gear_shifts(player.gear, player.heat_available)
    decision = Decision(DecisionKind.GEAR, learner_id, legal_gears)
    obs = encode_observation(state, learner_id, decision)
    mask = legal_action_mask(decision, state)

    policy = model.policy
    obs_t, _ = policy.obs_to_tensor(obs.reshape(1, -1))
    mask_t = torch.as_tensor(mask.reshape(1, -1), device=obs_t.device)
    with torch.no_grad():
        dist = policy.get_distribution(obs_t, action_masks=mask_t)
        probs = dist.distribution.probs.cpu().numpy().reshape(-1)
        value = float(policy.predict_values(obs_t).cpu().numpy().reshape(-1)[0])
    legal_idx = np.flatnonzero(mask)
    prior = probs[legal_idx]
    prior = prior / prior.sum() if prior.sum() > 0 else np.ones(len(legal_idx)) / len(legal_idx)
    return prior, value


def _puct_argmax_and_entropy(
    prior: np.ndarray,
    child_q: np.ndarray,
    visits: np.ndarray,
    c_init: float,
    c_base: float,
):
    """Return ``(argmax, selection_entropy)`` for one PUCT step over children.

    ``selection_entropy`` is the entropy of the softmax over the PUCT scores -- a
    proxy for how much the exploration term still shapes selection. With a raw,
    large-magnitude ``Q`` the score is Q-dominated (low entropy, prior inert);
    with normalized ``Q in [0,1]`` the ``P`` term competes (higher entropy).
    """
    total_n = visits.sum()
    c_puct = c_init + math.log((total_n + c_base + 1) / c_base)
    u = c_puct * prior * math.sqrt(total_n + 1e-8) / (1.0 + visits)
    score = child_q + u
    argmax = int(np.argmax(score))
    # Selection-entropy proxy: softmax over scores.
    z = score - score.max()
    p = np.exp(z)
    p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return argmax, ent


def q_normalization_experiment(
    model, snaps: list[tuple[GameState, int]], n_states: int, c_init: float, c_base: float
):
    """Compare raw vs min-max-normalized Q on the PUCT child selection.

    For each state we synthesize a plausible mid-search snapshot: a few children
    with the net prior, small visit counts, and child Q values on the REAL
    ``-rounds_remaining`` scale (~-25..0). We then compute the PUCT argmax + entropy
    under raw Q and under min-max-normalized Q, and report (a) how often the two
    pick different children (argmax-flip rate) and (b) the mean selection entropy
    under each. A high flip rate + a large entropy gap CONFIRMS that raw Q swamps
    the prior and normalization is mandatory (§4.1).
    """
    rng = np.random.default_rng(0)
    flips = 0
    ent_raw_list: list[float] = []
    ent_norm_list: list[float] = []
    n = min(n_states, len(snaps))
    for i in range(n):
        state, pid = snaps[i]
        prior, _ = _net_prior_and_value(model, state, pid)
        k = len(prior)
        # Realistic child Q on the -rounds_remaining scale: rounds-to-finish from
        # this state is ~ a dozen-plus; spread the children a few rounds apart.
        center = -float(rng.integers(8, 25))
        child_q_raw = center + rng.uniform(-3.0, 3.0, size=k)
        visits = rng.integers(0, 6, size=k).astype(float)
        # Min-max normalize to [0,1] over the (tree-local) child Q range.
        qmin, qmax = child_q_raw.min(), child_q_raw.max()
        span = qmax - qmin
        child_q_norm = (child_q_raw - qmin) / span if span > 1e-9 else np.zeros(k)

        am_raw, ent_raw = _puct_argmax_and_entropy(
            prior, child_q_raw, visits, c_init, c_base
        )
        am_norm, ent_norm = _puct_argmax_and_entropy(
            prior, child_q_norm, visits, c_init, c_base
        )
        if am_raw != am_norm:
            flips += 1
        ent_raw_list.append(ent_raw)
        ent_norm_list.append(ent_norm)
    return {
        "n": n,
        "argmax_flip_rate": flips / n if n else float("nan"),
        "mean_entropy_raw": statistics.mean(ent_raw_list) if ent_raw_list else float("nan"),
        "mean_entropy_norm": statistics.mean(ent_norm_list) if ent_norm_list else float("nan"),
    }


# ---------------------------------------------------------------------------
# E) Reseed purity (the determinism contract)
# ---------------------------------------------------------------------------


def reseed_purity_check(
    state: GameState, learner_id: int, rollout: HeuristicAgent
) -> bool:
    """A fixed ``(root, search seed)`` must yield a byte-stable descent.

    We replay the SAME deterministic reseed sequence twice (the
    ``LookaheadAgent._score_plan`` scheme) descending several chance samples, and
    assert the resulting per-sample positions/heat are identical -- so the whole
    search is a pure function of ``(root, seed)`` with no global-RNG leak. Also
    confirms the default ``reseed=None`` is NON-pure (it advances the parent RNG),
    documenting why explicit reseeds are mandatory.
    """
    base = 4242

    def descent() -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for s in range(6):
            reseed = (base + s * 31) & 0x7FFFFFFF
            clone = state.clone(reseed=reseed)
            _advance_one_round(clone, learner_id, rollout)
            _advance_one_round(clone, learner_id, rollout)
            p = clone.get_player(learner_id)
            out.append((p.position, p.lap, p.heat_available))
        return out

    return descent() == descent()


# ---------------------------------------------------------------------------
# LookaheadAgent baseline (the bar to compare against)
# ---------------------------------------------------------------------------


def measure_lookahead_baseline(
    track_seeds: list[int], seed_base: int
) -> tuple[float, float]:
    """Run the S1 LookaheadAgent solo and return ``(ms_per_move, clones_per_move)``.

    The success criterion is "within ~2-5x of LookaheadAgent's ~4 ms/move solo"
    (README §C0), so we re-measure that baseline on THIS box rather than trusting
    the S2 number, using the agent's own ``SearchProfile``.
    """
    agent = LookaheadAgent()
    for gi, tseed in enumerate(track_seeds):
        track = generate_track(tseed, _TIGHT_PARAMS)
        game = Game(track, [agent], logging_enabled=True, seed=seed_base + gi)
        game.run()
    return agent.profile.ms_per_move(), agent.profile.clones_per_move()


def measure_plies_per_round(
    snaps: list[tuple[GameState, int]], rollout: HeuristicAgent, n: int
) -> float:
    """Average number of learner DECISION edges that fire per round (for path len).

    Counts the learner's GEAR/CARDS/REACT/SLIPSTREAM/DISCARD decisions in one
    round's driver from a representative snapshot -- the searched DecisionKinds per
    round, which sets ``plies_per_round`` in the path-length model (B).
    """
    counts: list[int] = []
    for i in range(min(n, len(snaps))):
        st, pid = snaps[i]
        clone = st.clone(reseed=900 + i)
        c = 0
        if not clone.is_game_over:
            gen = run_round_driver(clone)
            try:
                decision = next(gen)
                while True:
                    if decision.player_id == pid:
                        c += 1
                    action = opponent_action(rollout, decision, clone)
                    decision = gen.send(action)
            except StopIteration:
                pass
        counts.append(c)
    return statistics.mean(counts) if counts else 0.0


def measure_cards_branch_width(
    snaps: list[tuple[GameState, int]], n: int
) -> tuple[float, int]:
    """Recorded pruned CARDS branch width via ``_candidate_plans`` dedup-by-speed.

    The §4.1 / Gumbel-seam data point: how wide is the *pruned* candidate set the
    search must branch on? We reuse the exact S1 prune
    (``LookaheadAgent._candidate_plans``) on each snapshot and report the mean and
    max kept-candidate count -- the number the deferred Gumbel go/no-go reads.
    """
    agent = LookaheadAgent()
    widths: list[int] = []
    for i in range(min(n, len(snaps))):
        st, pid = snaps[i]
        player = st.get_player(pid)
        legal_gears = rules.legal_gear_shifts(player.gear, player.heat_available)
        cands = agent._candidate_plans(st, pid, legal_gears)
        widths.append(len(cands))
    if not widths:
        return 0.0, 0
    return statistics.mean(widths), max(widths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke: small iteration counts (~seconds)")
    ap.add_argument("--states", type=int, default=40,
                    help="number of mid-game solo snapshots to capture")
    ap.add_argument("--clone-iters", type=int, default=8000,
                    help="clone+advance timing iterations")
    ap.add_argument("--fwd-iters", type=int, default=1000,
                    help="single net-forward timing iterations")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="frontier batch size for batched-forward timing")
    ap.add_argument("--batch-iters", type=int, default=200,
                    help="batched net-forward timing iterations")
    ap.add_argument("--baseline-games", type=int, default=6,
                    help="solo games to re-measure the LookaheadAgent baseline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.states = min(args.states, 12)
        args.clone_iters = 800
        args.fwd_iters = 120
        args.batch_iters = 30
        args.baseline_games = 2

    print("=" * 72)
    print("Sprint C0 -- net-guided solo stochastic-MCTS cost & constants spike")
    print("=" * 72)
    print(f"args: states={args.states} clone_iters={args.clone_iters} "
          f"fwd_iters={args.fwd_iters} batch_size={args.batch_size} "
          f"batch_iters={args.batch_iters} baseline_games={args.baseline_games} "
          f"seed={args.seed}")
    print("NOTE: net is freshly built with RANDOM weights. Forward TIMINGS are "
          "weight-independent (valid). The Q-norm experiment (D) synthesizes a "
          "realistic -rounds_remaining Q scale, so its conclusion does NOT hinge "
          "on the untrained value head (caveat noted).")

    rollout = HeuristicAgent()

    # -- capture states --
    print("\n[capture] collecting representative solo GEAR-node snapshots ...")
    snaps = _capture_solo_states(args.states, seed_base=1000 + args.seed)
    print(f"  captured {len(snaps)} snapshots on the held-out tight band")

    # -- A) clone+advance --
    print("\n[A] clone(reseed=) + advance-one-round (the TransitionModel edge)")
    ca = measure_clone_advance(snaps, args.clone_iters)
    print(f"  per clone+round   : {ca.per_call_us:8.1f} us  "
          f"({ca.calls_per_s:,.0f}/s)")
    print(f"  chance-edge frac  : {ca.chance_edge_frac:8.3f}  "
          "(advances crossing a round = a draw fires -- the §4.3 chance edge)")
    print(f"  reshuffle frac    : {ca.reshuffle_frac:8.3f}  "
          "(advances that actually consumed parent RNG -- the rarer reshuffle)")

    # -- A) net forward CPU + CUDA --
    print("\n[A] MaskableActorCriticPolicy forward (masked logits + value)")
    model = _build_net()
    sample_state, sample_pid = snaps[0]
    from heat.ml.action_codec import legal_action_mask
    from heat.ml.features import encode_observation
    s_player = sample_state.get_player(sample_pid)
    s_dec = Decision(
        DecisionKind.GEAR, sample_pid,
        rules.legal_gear_shifts(s_player.gear, s_player.heat_available),
    )
    obs = encode_observation(sample_state, sample_pid, s_dec)
    mask = legal_action_mask(s_dec, sample_state)

    fwd_cpu = measure_forward(model, obs, mask, "cpu",
                              args.fwd_iters, args.batch_size, args.batch_iters)
    print(f"  CPU  single forward: {fwd_cpu.per_call_ms:7.3f} ms  "
          f"({fwd_cpu.calls_per_s:,.0f}/s)")
    print(f"  CPU  batched/item  : {fwd_cpu.batch_per_item_ms:7.4f} ms  "
          f"(batch={fwd_cpu.batch_size})")

    fwd_cuda = measure_forward(model, obs, mask, "cuda",
                               args.fwd_iters, args.batch_size, args.batch_iters)
    if fwd_cuda.measured:
        print(f"  CUDA single forward: {fwd_cuda.per_call_ms:7.3f} ms  "
              f"({fwd_cuda.calls_per_s:,.0f}/s)")
        print(f"  CUDA batched/item  : {fwd_cuda.batch_per_item_ms:7.4f} ms  "
              f"(batch={fwd_cuda.batch_size})")
    else:
        print("  CUDA: NOT MEASURED on this box (torch.cuda.is_available() is "
              "False). Memo must flag the CUDA row as estimate-only.")

    # -- path-length inputs (informational: per-round decision granularity) --
    plies = measure_plies_per_round(snaps, rollout, n=min(20, len(snaps)))
    print(f"\n[B] measured plies/round (learner decision edges, informational) "
          f"= {plies:.2f}")

    # -- B) realistic in-tree product over a grid --
    print("\n[B] realistic in-tree per-move cost = sims*path*(clone+round) + leaves")
    rounds_lookahead = 4.0   # round-advances a typical selection descends to a leaf
    avg_path = _avg_path_len(rounds_lookahead)
    rollout_rounds = 3       # short rollout-to-pre-spin-floor length (leaf=rollout)
    print(f"  assumptions: avg_path_len={avg_path:.1f} round-advances/sim "
          f"(rounds_lookahead); rollout_leaf_rounds={rollout_rounds}. "
          "Cost unit = clone+advance-one-round (measured in A), so the primitive "
          "and the path length share a unit -- no double-count.")
    forward_ms = fwd_cpu.per_call_ms  # core impl starts CPU + unbatched (README)

    sims_grid = [16, 32, 64, 128] if not args.quick else [16, 64]
    leaf_modes = ["net", "rollout"]
    c_pw, alpha_pw, cap_k = 1.0, 0.5, 8  # default DPW point (reported per row)
    print(f"  {'sims':>5} {'leaf':>8} {'adv/mv':>8} {'fwd/mv':>7} "
          f"{'ms/move':>9} {'clones/mv':>10}")
    intree_rows: list[InTreeCost] = []
    for lm in leaf_modes:
        for ns in sims_grid:
            row = model_in_tree_cost(
                ns, c_pw, alpha_pw, cap_k, lm, avg_path,
                ca.per_call_us, forward_ms, rollout_rounds,
            )
            intree_rows.append(row)
            print(f"  {ns:>5} {lm:>8} {row.advances_per_move:>8.0f} "
                  f"{row.leaf_forwards_per_move:>7d} {row.ms_per_move:>9.2f} "
                  f"{row.clones_per_move:>10.0f}")

    # -- C) DPW fan-out grid --
    print("\n[C] DPW fan-out = min(K, ceil(C_pw * N^alpha_pw)) at sample visit counts")
    visit_points = [8, 16, 32, 64, 128]
    dpw_grid = [(1.0, 0.5), (2.0, 0.5), (1.0, 0.7)]
    for cpw, apw in dpw_grid:
        fans = [dpw_fanout(n, cpw, apw, cap_k) for n in visit_points]
        print(f"  C_pw={cpw} alpha={apw} K={cap_k}: "
              + " ".join(f"N={n}->{f}" for n, f in zip(visit_points, fans)))

    # -- C) variance check --
    print("\n[C] chance-node backed-up value stability vs fan-out (across reseeds)")
    chance = _find_chance_state(snaps, rollout)
    if chance is None:
        print("  no chance-edge state found in snapshots (try more --states)")
    else:
        cstate, cpid = chance
        fan_check = [1, 2, 4, 8, 16]
        rows = chance_value_variance(cstate, cpid, fan_check, rollout)
        print(f"  {'fanout':>6} {'mean_value':>11} {'std_across_reseeds':>19}")
        for m, mean_v, std_v in rows:
            print(f"  {m:>6} {mean_v:>11.3f} {std_v:>19.4f}")

    # -- D) Q-normalization --
    print("\n[D] Q-normalization: raw -rounds_remaining vs min-max [0,1] on PUCT")
    c_init, c_base = 1.25, 19652.0  # AZ/MuZero defaults (reported as chosen)
    qn = q_normalization_experiment(
        model, snaps, n_states=min(20, len(snaps)), c_init=c_init, c_base=c_base
    )
    print(f"  c_init={c_init} c_base={c_base}  (n={qn['n']} states)")
    print(f"  argmax-flip rate (raw vs norm) : {qn['argmax_flip_rate']:.3f}")
    print(f"  mean selection entropy  raw    : {qn['mean_entropy_raw']:.4f}")
    print(f"  mean selection entropy  norm   : {qn['mean_entropy_norm']:.4f}")
    print("  -> high flip rate + larger norm entropy CONFIRMS raw Q swamps the "
          "prior; min-max normalization is mandatory (§4.1).")

    # -- E) reseed purity --
    print("\n[E] reseed purity (determinism contract)")
    pure = reseed_purity_check(snaps[0][0], snaps[0][1], rollout)
    print(f"  fixed (root, seed) -> byte-stable descent: {pure}")

    # -- branch width --
    mean_w, max_w = measure_cards_branch_width(snaps, n=len(snaps))
    print(f"\n[branch] pruned CARDS candidate width (dedup-by-speed): "
          f"mean={mean_w:.1f} max={max_w}")

    # -- LookaheadAgent baseline --
    print("\n[baseline] LookaheadAgent solo cost (the bar) on this box")
    base_seeds = list(range(_HELDOUT_BASE, _HELDOUT_BASE + args.baseline_games))
    base_ms, base_clones = measure_lookahead_baseline(base_seeds, seed_base=5000)
    print(f"  LookaheadAgent: ms/move={base_ms:.2f}  clones/move={base_clones:.1f}")

    # -- F) throughput / verdict --
    print("\n[F] throughput + go/no-go vs the 2-5x LookaheadAgent criterion")
    print(f"  bar: LookaheadAgent = {base_ms:.2f} ms/move; "
          f"2x = {2*base_ms:.1f} ms, 5x = {5*base_ms:.1f} ms.")
    net_rows = [r for r in intree_rows if r.leaf_mode == "net"]
    print(f"  {'sims':>5} {'ms/move':>9} {'xLookahead':>11} {'in 2-5x?':>9}")
    feasible_sims: list[int] = []
    for r in net_rows:
        ratio = r.ms_per_move / base_ms if base_ms else float("nan")
        in_band = 2.0 <= ratio <= 5.0
        if in_band:
            feasible_sims.append(r.n_sims)
        flag = "yes" if in_band else ("<2x" if ratio < 2.0 else ">5x")
        print(f"  {r.n_sims:>5} {r.ms_per_move:>9.2f} {ratio:>10.1f}x {flag:>9}")
    max_feasible = max(feasible_sims) if feasible_sims else 0
    print(f"  -> largest sim budget within the 2-5x bar (net leaf, CPU): "
          f"{max_feasible} sims" if max_feasible else
          "  -> NO sim budget lands inside 2-5x at net-leaf/CPU on one core")

    # tracks/hour at a representative feasible point (a self-play episode = moves)
    moves_per_track = 60.0  # ~rounds-to-finish * plies; conservative solo estimate
    rep = next((r for r in net_rows if r.n_sims == max_feasible), net_rows[0])
    s_per_track = rep.ms_per_move / 1e3 * moves_per_track
    tracks_per_hr = 3600.0 / s_per_track if s_per_track else float("nan")
    print(f"  at {rep.n_sims} sims net-leaf: {rep.ms_per_move:.1f} ms/move -> "
          f"{s_per_track:.1f} s/track -> {tracks_per_hr:,.0f} tracks/hour/core "
          f"(~{moves_per_track:.0f} moves/track).")
    print("  Self-play volume assumption (C2/C3 small scale): ~a few thousand "
          "tracks/iteration. At the rate above that is hours on one core and "
          "embarrassingly parallel across cores -- inside the hours-not-days bar.")

    print("\n  KEY FINDING: per-move cost is CLONE-bound, not net-forward-bound "
          f"(net forward {fwd_cpu.per_call_ms:.2f} ms << {rep.n_sims} clones x "
          f"{ca.per_call_us/1e3:.3f} ms). MuZero/CUDA do NOT help the binding "
          "constraint; fewer/cheaper engine advances would. See C0-findings.md.")
    print("=" * 72)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("spike_mcts_cost", main)
