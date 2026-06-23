"""Sprint C2 -- self-play target generation from the C1 stochastic MCTS.

Turns on the self-play exploration machinery the C1 ``MCTSAgent`` already
builds (root Dirichlet noise + the temperature schedule, both OFF for C1's
deterministic parity eval) and runs the search over many GENERATED solo tracks
to produce the AlphaZero training targets README §4.6 specifies:

  * ``obs``       -- ``encode_observation(state, pid, decision)``, the frozen
    codec observation at the decision (the policy is conditioned on the pending
    decision kind, exactly as BC logs it);
  * ``pi``        -- the MCTS **visit distribution** ``N(a)^{1/τ_t}/ΣN`` over the
    *full* ``ACTION_DIM`` flat space (the policy target -- NEVER the argmax: that
    would be BC again, with BC's gap). Support is exactly the kept candidate set;
  * ``mask``      -- ``legal_action_mask(decision, state)`` (so ``π``'s support is
    a subset of the mask -- asserted, and unit-tested);
  * ``z``         -- the **Monte-Carlo value target** ``−rounds_remaining`` with
    the **pre-spin floor** applied (see below), backfilled after the episode ends.

The single most important correctness item: the pre-spin floor on ``z``
-----------------------------------------------------------------------
The C1 leaf discipline floors a *spun* line so a reckless recovery cannot inflate
the value. In the MC value-target frame (``−rounds_remaining``) the analogous
hazard is: a state logged BEFORE an own-spin would otherwise be charged for all
the recovery rounds the spin cost, poisoning its cost-to-go with a disaster the
*state itself* did not cause. So when an own-spin occurs at round ``r_spin >=
round_num`` somewhere in the rest of the episode, the MC return for that state is
**truncated at the spin**: ``rounds_remaining = r_spin - round_num`` rather than
``finish_round - round_num``. This is the rounds-frame image of
``LookaheadAgent._pre_spin_progress`` ("credit only up to the corner the car
failed to clear"): the cost-to-go is counted only up to the spin, never through
the inflated recovery. ``tests/test_az_targets.py`` pins this on a forced-spin
track -- the realized MC return FLOORED, not the post-spin recovery.

Why search at EVERY decision (not just GEAR)
--------------------------------------------
C1's *acting* runs one full search at the GEAR root (the joint gear+cards plan)
and reads the net prior greedily for REACT/SLIP/DISCARD. But the C2 policy target
must be a genuine *visit distribution* for **all** searched ``DecisionKind``s
(README §4: the codec encodes every kind, so none is delegated). So the generator
roots an independent C1 search at every real-choice learner decision and logs that
root's visit distribution -- the search IS C1's ``_simulate`` core, rooted at the
live decision, with exploration on. This is generation-time only; it does not
change C1's acting contract.

Reuse (verbatim where possible)
-------------------------------
``gen_demos.py``'s solo driver loop, codec snapping, drop-and-count of off-table
REACT, and the track-disjoint seed-band machinery are reused. The seed band here
is **disjoint** from both the held-out eval band (``eval_search._HELDOUT_BASE =
900_000``) and the BC/value precursor bands (``100_000`` train / ``500_000`` val)
-- asserted programmatically at startup (review bug #6/#9).

Output: a compressed ``.npz`` (``obs``/``pi``/``mask``/``z``/``kind``/
``track_seed``) + a ``.selfplay.json`` provenance sidecar (codec version, the full
search config incl. DPW/PUCT/Dirichlet/temperature, the seed band).

Usage:
    python experiments/gen_selfplay.py --smoke
    python experiments/gen_selfplay.py --tracks 200 --sims 16 --model checkpoints/c_prior.zip
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from heat.agents.mcts_agent import (
    MCTSAgent,
    MCTSConfig,
    EngineTransitionModel,
    NetAdapter,
    Node,
    NodeKind,
)
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.ml.action_codec import encode_action_index, legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
from heat.tracks.generator import TrackGenParams, generate_track


# Mirror the eval harness's held-out, limit-1-weighted GENERATED distribution
# (experiments/eval_search.py) so the self-play data is drawn from the same corner
# mix the trained net is gated on.
_TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),  # weight toward tight (limit-1) corners
    laps=2,
)

#: Self-play seed band. DISJOINT from the held-out eval band (900_000+,
#: eval_search._HELDOUT_BASE) AND from the BC/value precursor bands (100_000 train
#: / 500_000 val in gen_demos / gen_value_data). 300_000 sits cleanly between, with
#: a 100_000-wide window before the 500_000 val band -- asserted at startup.
_SELFPLAY_SEED_BASE = 300_000
#: The bands we must stay disjoint from (base, width-guard) for the assertion.
_EVAL_HELDOUT_BASE = 900_000
_BC_TRAIN_BASE = 100_000
_BC_VAL_BASE = 500_000

#: The DecisionKind -> int code (kept in sync with gen_demos / train_bc).
_KIND_TO_INT: dict[DecisionKind, int] = {
    DecisionKind.GEAR: 0,
    DecisionKind.CARDS: 1,
    DecisionKind.REACT: 2,
    DecisionKind.SLIPSTREAM: 3,
    DecisionKind.DISCARD: 4,
}


def _assert_seed_bands_disjoint(n_tracks: int) -> None:
    """Fail fast if the self-play band overlaps the eval or precursor bands.

    The self-play band is ``[_SELFPLAY_SEED_BASE, _SELFPLAY_SEED_BASE+n_tracks)``.
    A trained-on track that is later *gated on* (or that the precursor net was
    trained/validated on) silently leaks generalization -- the review's bug #6/#9.
    We require a clear gap to every other band, not merely non-overlap of the
    exact ranges, so even a generous re-band stays safe.
    """
    hi = _SELFPLAY_SEED_BASE + n_tracks
    for name, base in (
        ("eval-heldout", _EVAL_HELDOUT_BASE),
        ("bc-train", _BC_TRAIN_BASE),
        ("bc-val", _BC_VAL_BASE),
    ):
        # The other bands are wide (100_000-track windows); require our band to sit
        # entirely below or above the other band's plausible 100_000-wide window.
        other_lo, other_hi = base, base + 100_000
        if not (hi <= other_lo or _SELFPLAY_SEED_BASE >= other_hi):
            raise ValueError(
                f"self-play seed band [{_SELFPLAY_SEED_BASE}, {hi}) overlaps the "
                f"{name} band [{other_lo}, {other_hi}); pick a disjoint base "
                "(review bug #6/#9: a gated/precursor track must never be a "
                "self-play track)"
            )


@dataclass
class _SelfPlayBuffer:
    """Growable column store for the self-play target tuples.

    ``round_num`` is kept per-row so the Monte-Carlo value target can be
    backfilled (and pre-spin-floored) once the episode ends; ``z`` starts unset
    and is filled in :meth:`backfill_episode`.
    """

    obs: list[np.ndarray] = field(default_factory=list)
    pi: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    kind: list[int] = field(default_factory=list)
    round_num: list[int] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    track_seed: list[int] = field(default_factory=list)
    split: list[str] = field(default_factory=list)

    def add(
        self,
        *,
        obs: np.ndarray,
        pi: np.ndarray,
        mask: np.ndarray,
        kind: int,
        round_num: int,
        track_seed: int,
        split: str,
    ) -> int:
        self.obs.append(obs)
        self.pi.append(pi)
        self.mask.append(mask)
        self.kind.append(kind)
        self.round_num.append(round_num)
        self.z.append(float("nan"))  # backfilled on episode completion
        self.track_seed.append(track_seed)
        self.split.append(split)
        return len(self.obs) - 1

    def __len__(self) -> int:
        return len(self.obs)


# ---------------------------------------------------------------------------
# The generator: a C1 search rooted at an arbitrary decision, exploration ON
# ---------------------------------------------------------------------------


def _greedy_acted_edge(agent: MCTSAgent, root: Node):
    """The greediest acted edge over the SEARCHED root edges (the C5 traj knob).

    The trajectory-greediness control (C5): instead of the exploratory acted move
    (Gumbel-sampled, or temperature-sampled in the C1/PUCT window), drive the
    *trajectory* with the lower-variance greedy move so fewer self-play episodes
    spin out to MAX_ROUNDS and the MC value target / finished-val split survive.
    This touches ONLY the *acted* action -- the logged completed-Q / visit-count
    ``pi`` target is built independently and is byte-identical with this knob on or
    off (C4's entropy win is preserved; the trajectory is decoupled from the target).

    Greedy = most-visited searched edge (the AZ acting rule, stable on first-seen
    order). Falls back to ``_best_edge`` if no edge was searched (tiny-sim budget).
    """
    searched = [e for e in root.edges if e.n > 0]
    if not searched:
        return agent._best_edge(root, at_root=True)
    best = searched[0]
    best_key = (-1, 0)
    for i, e in enumerate(searched):
        key = (e.n, -i)  # most-visited, ties broken by first-seen order
        if key > best_key:
            best_key = key
            best = e
    return best


def _search_visit_distribution(
    agent: MCTSAgent,
    state: GameState,
    decision: Decision,
    *,
    traj_greedy: bool = False,
) -> tuple[np.ndarray, object]:
    """Run a C1 MCTS rooted at ``decision`` and return ``(pi, acted_action)``.

    This is C1's ``_simulate`` core, rooted at the live ``decision`` (any
    ``DecisionKind``) with self-play exploration ON (root Dirichlet noise +
    temperature schedule, read from ``agent.config``). It returns:

      * ``pi``  -- the full ``ACTION_DIM`` visit-distribution target
        ``N(a)^{1/τ_t}/ΣN`` over the kept candidate edges (zeros elsewhere). With
        ``temperature_moves`` active for this ply ``τ=1`` (so ``π ∝ N``); past the
        schedule ``τ→0`` (so ``π`` is one-hot on the most-visited edge -- the
        greedy target), exactly README §4.5/§4.6.
      * ``acted_action`` -- the engine action to actually play (sampled from the
        visit distribution during the temperature window, greedy after) -- so the
        *trajectory* explores while the *target* is the full distribution.

    The search is a pure function of ``(state, agent.seed, ply)`` (the C1
    determinism contract): Dirichlet/temperature draws come from the agent's
    per-turn RNG seeded off the turn signature, never global RNG.
    """
    pid = decision.player_id
    agent.to_move_pid = pid
    sig = agent._turn_signature(state, pid)
    turn_seed = agent._turn_seed(sig)

    root = Node(
        kind=NodeKind.DECISION,
        state=state,
        action_path=(),
        to_move=pid,
        decision=decision,
    )
    # Root reference (so Dirichlet noise targets the ROOT prior only) + the
    # deterministic per-turn RNG for noise/temperature sampling.
    import random as _random

    agent._root_node = root
    agent._search_rng = _random.Random(turn_seed)

    model = EngineTransitionModel(pid)
    agent._q_min = float("inf")
    agent._q_max = float("-inf")
    agent._clones_this_move = 0

    t0 = time.perf_counter()
    agent._run_root_search(root, model, turn_seed)
    agent.profile.record_move(agent._clones_this_move, time.perf_counter() - t0)

    # Gumbel root selector (Sprint C4): build the COMPLETED-Q policy target
    # π = softmax(logits + σ(completedQ)) instead of the visit-count distribution.
    # Same (obs, π, mask, z) output schema; only the *policy* target changes.
    if agent.config.root_selector == "gumbel":
        return _completed_q_distribution(agent, root, decision, traj_greedy=traj_greedy)

    # Build the visit-distribution policy target over the full flat action space.
    pi = np.zeros(ACTION_DIM, dtype=np.float64)
    in_window = agent._ply < agent.config.temperature_moves
    counts: list[tuple[int, float]] = []  # (flat_index, weight)
    for edge in root.edges:
        try:
            flat = encode_action_index(decision, edge.action)
        except (ValueError, IndexError):
            # Off-table action (codec can't encode, e.g. a non-tabled REACT). It
            # cannot be a policy target; skip it from the distribution.
            continue
        if not (0 <= flat < ACTION_DIM):
            continue
        if in_window:
            weight = float(edge.n)  # τ=1 => N(a)^1
        else:
            weight = float(edge.n)  # τ→0 handled below (one-hot on argmax N)
        counts.append((flat, weight))

    if not counts:
        # Degenerate (no encodable edge searched): one-hot on the acted action's
        # flat index if it encodes, else a uniform-over-mask fallback. This is the
        # tiny-sim / all-off-table edge case; callers drop unencodable acted moves.
        pi_built = False
    else:
        if in_window:
            total = sum(w for _, w in counts)
            if total > 0.0:
                for flat, w in counts:
                    pi[flat] = w / total
                pi_built = True
            else:
                pi_built = False
        else:
            # τ→0: one-hot on the most-visited (greedy) encodable edge.
            flat_best = max(counts, key=lambda fw: fw[1])[0]
            pi[flat_best] = 1.0
            pi_built = True

    # The acted action: reuse C1's _best_edge acting rule (temperature sampling in
    # the window, most-visited after) so the trajectory matches the schedule -- OR,
    # with the C5 traj_greedy knob, the greedy most-visited edge (lower-variance,
    # more finishes). The logged pi target above is unchanged either way.
    if traj_greedy:
        acted_edge = _greedy_acted_edge(agent, root)
    else:
        acted_edge = agent._best_edge(root, at_root=True)
    acted_action = acted_edge.action

    if not pi_built:
        # Fall back to a one-hot target on the acted action if it encodes.
        try:
            flat = encode_action_index(decision, acted_action)
            if 0 <= flat < ACTION_DIM:
                pi[:] = 0.0
                pi[flat] = 1.0
        except (ValueError, IndexError):
            pass

    return pi, acted_action


def _completed_q_distribution(
    agent: MCTSAgent,
    root: Node,
    decision: Decision,
    *,
    traj_greedy: bool = False,
) -> tuple[np.ndarray, object]:
    """Build the Gumbel **completed-Q** policy target (Sprint C4).

    ``π(a) = softmax_a( logits_a + σ(completedQ_a) )`` over the kept candidate set
    (zeros elsewhere in the full ``ACTION_DIM`` vector), where

      * ``logits_a`` is the net's log-prior over the kept set (``GumbelRootResult``);
      * ``completedQ_a = Q̂_a = _normalize_q(edge.q())`` for a SEARCHED edge
        (``edge.n > 0``);
      * ``completedQ_a = v̂`` (the root's own normalized value ``root_value_norm``)
        for a sampled-but-unvisited / un-sampled action -- the "completion" step
        (Sprint C4 decision #2: the standard AlphaZero choice, NOT the interior
        FPU-reduced parent value).

    The σ transform reuses ``agent._gumbel_sigma`` (the same already-normalized
    ``[0,1]`` Q̂ the interior PUCT uses, so the −rounds_remaining scale stays tamed).
    The output schema is IDENTICAL to the visit-count path -- ``(pi, acted_action)``
    with ``pi`` a probability vector whose support ⊆ mask and sums to 1 -- so
    ``train_az`` and the ``.npz``/``MLAgent`` contract are unchanged. The same
    off-table drop-and-count + degenerate fallback the caller applies still hold.
    """
    gr = agent._gumbel_result
    assert gr is not None and gr.edges is root.edges

    edges = gr.edges
    # σ's max_n is the largest visit count over the searched root edges (Danihelka).
    max_n = max((e.n for e in edges), default=0)

    # completedQ per edge, then the un-normalized score logits + σ(completedQ).
    scores: list[tuple[int, float]] = []  # (flat_index, score)
    for i, edge in enumerate(edges):
        try:
            flat = encode_action_index(decision, edge.action)
        except (ValueError, IndexError):
            # Off-table action (codec can't encode): cannot be a policy target.
            continue
        if not (0 <= flat < ACTION_DIM):
            continue
        if edge.n > 0:
            q_hat = agent._normalize_q(edge.q())
        else:
            q_hat = gr.root_value_norm  # completion baseline v̂ (decision #2)
        score = gr.logits[i] + agent._gumbel_sigma(q_hat, max_n)
        scores.append((flat, score))

    pi = np.zeros(ACTION_DIM, dtype=np.float64)
    pi_built = False
    if scores:
        # Numerically-stable softmax over the kept candidate scores.
        raw = np.array([s for _, s in scores], dtype=np.float64)
        raw -= raw.max()
        exp = np.exp(raw)
        total = float(exp.sum())
        if total > 0.0 and np.isfinite(total):
            probs = exp / total
            for (flat, _), p in zip(scores, probs):
                pi[flat] = p
            pi_built = True

    # The acted action: the Gumbel-selected argmax(g + logits + σ(Q̂)) (via
    # _best_edge's Gumbel-aware root branch) -- OR, with the C5 traj_greedy knob,
    # the greedy most-visited searched survivor (lower-variance, more finishes). The
    # logged completed-Q pi target above is unchanged either way (C4's entropy win
    # is preserved; the trajectory is decoupled from the target).
    if traj_greedy:
        acted_edge = _greedy_acted_edge(agent, root)
    else:
        acted_edge = agent._best_edge(root, at_root=True)
    acted_action = acted_edge.action

    if not pi_built:
        # Degenerate (no encodable edge): fall back to a one-hot on the acted action.
        try:
            flat = encode_action_index(decision, acted_action)
            if 0 <= flat < ACTION_DIM:
                pi[:] = 0.0
                pi[flat] = 1.0
        except (ValueError, IndexError):
            pass

    return pi, acted_action


def _spin_round(state: GameState, player_id: int) -> int | None:
    """Round number of the FIRST own-spin in this episode, or ``None``.

    Read from the (logging-enabled) game state's event log. Used to truncate the
    MC value target at the spin (the pre-spin floor): a state logged before this
    round is charged cost-to-go only up to here, never through the recovery.
    """
    best: int | None = None
    for e in state.event_log:
        if e.event_type == "spin_out" and e.player_id == player_id:
            r = int(e.round_num)
            if best is None or r < best:
                best = r
    return best


def _floored_rounds_remaining(round_num: int, finish_round: int, spin_round: int | None) -> int:
    """MC rounds-to-go from ``round_num`` with the pre-spin floor applied.

    The realized MC return is ``finish_round - round_num``. If an own-spin occurs
    at ``spin_round >= round_num`` later in the episode, the return is **truncated
    at the spin** (``spin_round - round_num``) so the recovery rounds the spin cost
    do not poison the target -- the rounds-frame image of
    ``LookaheadAgent._pre_spin_progress``. A spin BEFORE this state (``spin_round <
    round_num``) is already paid and does not floor a later, clean state.
    """
    horizon = finish_round - round_num
    if spin_round is not None and spin_round >= round_num:
        horizon = min(horizon, spin_round - round_num)
    return max(0, horizon)


def _generate_one_track(
    track_seed: int,
    split: str,
    *,
    game_seed: int,
    agent: MCTSAgent,
    buf: _SelfPlayBuffer,
    traj_greedy: bool = False,
) -> tuple[bool, int]:
    """Drive one full solo self-play episode, logging a target at each real choice.

    Mirrors ``gen_demos._generate_one_track`` (the proven solo driver loop:
    ``player.lap = 1`` init, the ``MAX_ROUNDS`` guard, the StopIteration re-arming
    of ``run_round_driver``), but at every learner decision with ``> 1`` legal
    action it roots a C1 search (exploration ON), logs the ``(obs, π, mask)``
    target, and PLAYS the searched/sampled action (so the trajectory carries the
    exploration). Degenerate decisions (``mask.sum() <= 1``) are auto-resolved with
    the lone legal action and not logged (the env's information set, gen_demos
    rule #1).

    On completion, backfills ``z = −floored_rounds_remaining`` for every row.
    Returns ``(finished, n_unencodable)`` -- whether the race finished cleanly
    (rows kept) and the number of real-choice acted moves that had no codec
    encoding (dropped + counted, the gen_demos drop-and-count, S3 lesson #2).
    """
    track = generate_track(track_seed, _TIGHT_PARAMS)
    # logging_enabled=True so the per-episode spin event log is available for the
    # pre-spin-floor backfill (the single most important correctness item).
    state = GameState.create(track, 1, logging_enabled=True, seed=game_seed)
    for player in state.players:
        player.lap = 1  # mirror Game.__init__ / HeatEnv.reset

    learner_id = 0
    row_ids: list[int] = []
    n_unencodable = 0
    # Reset the agent's episode ply counter so the temperature schedule restarts.
    agent._ply = 0

    gen = run_round_driver(state)
    send_value: object = None

    while True:
        if state.is_game_over or state.round_num > MAX_ROUNDS:
            break
        try:
            decision = gen.send(send_value)
        except StopIteration:
            if state.is_game_over or state.round_num > MAX_ROUNDS:
                break
            gen = run_round_driver(state)
            send_value = None
            continue

        if decision.player_id != learner_id:
            # Solo: never taken (no other seats). Defensive parity with gen_demos.
            from heat.ml.opponents import opponent_action
            from heat.agents.heuristic_agent import HeuristicAgent

            send_value = opponent_action(HeuristicAgent(), decision, state)
            continue

        mask = legal_action_mask(decision, state)
        n_legal = int(mask.sum())

        if n_legal <= 1:
            # Degenerate decision: auto-resolve with the lone legal action exactly
            # as HeatEnv does, and do NOT log (matches the BC/env info set).
            send_value = _lone_legal_action(decision, state, mask)
            continue

        # Real choice: root a C1 search (exploration ON), log the visit target.
        obs = encode_observation(state, learner_id, decision)
        pi, acted_action = _search_visit_distribution(
            agent, state, decision, traj_greedy=traj_greedy
        )
        agent._ply += 1

        # Drop-and-count: if the acted action has no codec encoding, it cannot be a
        # learnable target (gen_demos / S3 lesson #2). Skip logging, still play it.
        try:
            acted_flat = encode_action_index(decision, acted_action)
            encodable = 0 <= acted_flat < ACTION_DIM and bool(mask[acted_flat])
        except (ValueError, IndexError):
            encodable = False

        if not encodable:
            n_unencodable += 1
            send_value = acted_action
            continue

        # Contract guards: π is a probability vector, its support is within the
        # mask, and it sums to 1 (the dataset can never carry an off-mask target).
        support = pi > 0.0
        if support.any():
            if not bool(mask[support].all()):
                raise AssertionError(
                    f"pi support escapes the mask for {decision.kind} "
                    "(codec drift / candidate-mask mismatch)"
                )
            s = float(pi.sum())
            if not (abs(s - 1.0) < 1e-6):
                raise AssertionError(f"pi does not sum to 1 ({s}) for {decision.kind}")
        else:
            # No positive support (degenerate search) -- skip logging this state
            # rather than store an all-zero policy target.
            send_value = acted_action
            continue

        row_id = buf.add(
            obs=obs,
            pi=pi.astype(np.float32),
            mask=mask,
            kind=_KIND_TO_INT[decision.kind],
            round_num=state.round_num,
            track_seed=track_seed,
            split=split,
        )
        row_ids.append(row_id)
        send_value = acted_action

    finished = state.is_game_over and not (state.round_num > MAX_ROUNDS)
    if finished:
        finish_round = state.round_num
        spin_round = _spin_round(state, learner_id)
        for row_id in row_ids:
            rr = _floored_rounds_remaining(
                buf.round_num[row_id], finish_round, spin_round
            )
            buf.z[row_id] = float(-rr)  # z = −rounds_remaining (floored)
    else:
        # MAX_ROUNDS-truncated: no valid finish_round, mark rows for dropping.
        for row_id in row_ids:
            buf.z[row_id] = float("nan")
    return finished, n_unencodable


# ---------------------------------------------------------------------------
# Sprint C6: the perfect-info 1v1 win/loss generator
# ---------------------------------------------------------------------------


def _winloss_z(state: GameState, learner_id: int, opponent_id: int) -> float:
    """The learner's realized 1v1 outcome (``+1`` / ``−1`` / ``0``) at game end.

    Delegates to ``mcts_agent._winloss_result`` so the backfilled training target
    and the search leaf compute the IDENTICAL binary outcome (finish-order when a
    seat crossed, else race-progress on the MAX_ROUNDS tie -- resolved-sub-decision
    3, draw only on an exact progress tie).
    """
    from heat.agents.mcts_agent import _winloss_result

    return _winloss_result(state, learner_id, opponent_id)


def _make_opponent(opponent_snapshot: str | None):
    """Build the opponent seat agent for net-vs-frozen 1v1 self-play, or ``None``.

    ``None`` (net-vs-net) means BOTH seats are the current net (the search itself),
    so no separate opponent agent is needed -- both seats are driven by the
    learner's two-player search and both log. A non-None ``opponent_snapshot`` is a
    fixed adversary: a ``FrozenSnapshotAgent`` path (a past best-of-gen checkpoint)
    or one of the heuristic anchor sentinels ``"strong"`` / ``"weak"``.
    """
    if opponent_snapshot is None:
        return None
    if opponent_snapshot == "strong":
        from heat.agents.strong_heuristic import StrongHeuristicAgent

        return StrongHeuristicAgent(strength=2)
    if opponent_snapshot == "weak":
        from heat.agents.heuristic_agent import HeuristicAgent

        return HeuristicAgent()
    from heat.ml.training import FrozenSnapshotAgent

    return FrozenSnapshotAgent(opponent_snapshot)


def _generate_one_1v1(
    track_seed: int,
    split: str,
    *,
    game_seed: int,
    agent: MCTSAgent,
    opponent,
    buf: _SelfPlayBuffer,
    log_seats: tuple[int, ...],
    traj_greedy: bool = False,
) -> tuple[bool, int]:
    """Drive one perfect-info 1v1 self-play game, logging a target per learner move.

    The two-seat analogue of :func:`_generate_one_track` (Sprint C6): a 2-player
    game is driven turn-by-turn; at every searched decision owned by a seat in
    ``log_seats`` (with ``> 1`` legal action) a **two-player minimax** MCTS is
    rooted at that decision (``MCTSConfig.two_player`` on, ``opponent_id`` set to
    the other seat), the ``(obs, π, mask)`` target is logged, and the searched/
    sampled action is played. Decisions owned by a non-logged seat (a frozen
    snapshot / heuristic anchor when self-play is net-vs-frozen) are resolved by
    ``opponent``; degenerate decisions (``mask.sum() <= 1``) are auto-resolved and
    not logged.

    ``log_seats`` is ``(0, 1)`` for net-vs-net (BOTH seats are the current net, so
    both log -- each from its own mover's perspective) or ``(0,)`` for
    net-vs-frozen (only the current-net seat's rows are training targets; the
    snapshot is a fixed adversary, not a learner).

    On completion every row's ``z`` is backfilled to the realized 1v1 outcome
    (``+1`` / ``−1`` / ``0``) FROM THAT ROW'S MOVER's perspective -- always defined
    (the game always finishes, or MAX_ROUNDS decides on progress), so no episode
    or row is dropped. Returns ``(finished, n_unencodable)``; ``finished`` is True
    for every 1v1 game (the value target is always defined -- the structural win).
    """
    track = generate_track(track_seed, _TIGHT_PARAMS)
    state = GameState.create(track, 2, logging_enabled=True, seed=game_seed)
    for player in state.players:
        player.lap = 1  # mirror Game.__init__ / HeatEnv.reset

    # Per-row mover seat, so z is backfilled from the right perspective.
    row_ids: list[int] = []
    row_movers: list[int] = []
    n_unencodable = 0
    agent._ply = 0

    gen = run_round_driver(state)
    send_value: object = None

    while True:
        if state.is_game_over or state.round_num > MAX_ROUNDS:
            break
        try:
            decision = gen.send(send_value)
        except StopIteration:
            if state.is_game_over or state.round_num > MAX_ROUNDS:
                break
            gen = run_round_driver(state)
            send_value = None
            continue

        seat = decision.player_id
        if seat not in log_seats:
            # A non-logged seat (frozen snapshot / anchor): resolve via opponent.
            from heat.ml.opponents import opponent_action

            send_value = opponent_action(opponent, decision, state)
            continue

        mask = legal_action_mask(decision, state)
        n_legal = int(mask.sum())
        if n_legal <= 1:
            send_value = _lone_legal_action(decision, state, mask)
            continue

        opp_id = 1 - seat  # the other seat in the 1v1
        obs = encode_observation(state, seat, decision)
        # Point the two-player search at THIS mover (the opponent is the other
        # seat); the search reads the same net's prior from each seat's obs.
        agent.config.two_player = True
        agent.config.opponent_id = opp_id
        pi, acted_action = _search_visit_distribution(
            agent, state, decision, traj_greedy=traj_greedy
        )
        agent._ply += 1

        try:
            acted_flat = encode_action_index(decision, acted_action)
            encodable = 0 <= acted_flat < ACTION_DIM and bool(mask[acted_flat])
        except (ValueError, IndexError):
            encodable = False
        if not encodable:
            n_unencodable += 1
            send_value = acted_action
            continue

        support = pi > 0.0
        if support.any():
            if not bool(mask[support].all()):
                raise AssertionError(
                    f"pi support escapes the mask for {decision.kind} "
                    "(codec drift / candidate-mask mismatch)"
                )
            s = float(pi.sum())
            if not (abs(s - 1.0) < 1e-6):
                raise AssertionError(f"pi does not sum to 1 ({s}) for {decision.kind}")
        else:
            send_value = acted_action
            continue

        row_id = buf.add(
            obs=obs,
            pi=pi.astype(np.float32),
            mask=mask,
            kind=_KIND_TO_INT[decision.kind],
            round_num=state.round_num,
            track_seed=track_seed,
            split=split,
        )
        row_ids.append(row_id)
        row_movers.append(seat)
        send_value = acted_action

    # Backfill the win/loss z from each row's own mover's perspective. The 1v1
    # game is ALWAYS decided (finish order, or MAX_ROUNDS progress), so every row
    # gets a defined ±1/0 target -- no episode/row drop (the structural win).
    opp_of = {0: 1, 1: 0}
    for row_id, mover in zip(row_ids, row_movers):
        buf.z[row_id] = float(_winloss_z(state, mover, opp_of[mover]))
    return True, n_unencodable


def _lone_legal_action(decision: Decision, state: GameState, mask: np.ndarray) -> object:
    """The single legal action for a degenerate decision (mask.sum() <= 1).

    Decodes the one set mask bit back to the concrete engine action so the driver
    advances exactly as ``HeatEnv._forced_action`` would. Falls back to a kind-safe
    default if the mask is somehow empty (never expected; every decision has >= 1
    legal action).
    """
    from heat.ml.action_codec import decode_action

    flat = int(np.argmax(mask)) if mask.any() else -1
    if flat >= 0:
        return decode_action(decision, flat, state)
    return EngineTransitionModel._legal_default(decision)


# ---------------------------------------------------------------------------
# Sprint C7: per-game work unit + the process-level cached NetAdapter (model
# reuse), so the per-game model-reload tax is gone and games can be distributed
# across a process pool with a DETERMINISTIC, byte-identical merge.
# ---------------------------------------------------------------------------


#: Process-level shared NetAdapter, keyed by model path. The adapter's heavy SB3
#: module comes from the agent's process-level ``_MODEL_CACHE`` (mcts_agent), so
#: the policy loads ONCE per process; reusing one adapter also amortizes the cheap
#: ``.meta.json`` validation. Shared read-only across the process's games -- the
#: search never mutates the weights, so this is safe and determinism is unchanged.
_NET_ADAPTER_CACHE: dict[str, NetAdapter] = {}


def _cached_net_adapter(model_path: str) -> NetAdapter:
    """Return a process-shared :class:`NetAdapter` for ``model_path`` (Sprint C7).

    The first game in a process builds the adapter and (on its first leaf eval)
    loads the SB3 module into the process model cache; every later game in that
    process reuses the same adapter and the cached module -- no per-game reload.
    """
    adapter = _NET_ADAPTER_CACHE.get(model_path)
    if adapter is None:
        adapter = NetAdapter(model_path)
        _NET_ADAPTER_CACHE[model_path] = adapter
    return adapter


@dataclass
class _GameSpec:
    """An immutable, picklable description of one self-play game to run.

    Carries only plain data (the model PATH, seeds, the search config, the
    split/track key) so it ships into a ``multiprocessing`` worker by value -- the
    NetAdapter pickle-by-path contract. The per-game agent seed / game seed are
    captured here exactly as the serial loop derived them (``args.seed + i`` /
    ``args.game_seed + i`` with ``i`` the per-band index), so a worker reproduces
    the byte-identical game.
    """

    split: str
    track_seed: int
    agent_seed: int
    game_seed: int
    model_path: str
    cfg: MCTSConfig
    two_player: bool
    opponent_snapshot: str | None
    log_seats: tuple[int, ...]
    traj_greedy: bool


@dataclass
class _GameResult:
    """The byte-identical result of running one :class:`_GameSpec`.

    The per-game :class:`_SelfPlayBuffer` keeps the rows in driver order; the
    deterministic merge concatenates results sorted by ``(split_rank, track_seed)``
    so the assembled dataset is identical regardless of ``--workers``. The profile
    counts are summed across games (the accounting preserved across workers).
    """

    split: str
    track_seed: int
    buf: _SelfPlayBuffer
    finished: bool
    n_unencodable: int
    clones: int
    moves: int
    seconds: float


def _run_one_game(spec: _GameSpec) -> _GameResult:
    """Run one self-play game and return its rows + profile (Sprint C7 work unit).

    A fresh :class:`MCTSAgent` per game (clean per-game profile / ply / RNG) -- the
    C2-C6 contract -- but the heavy net is the process-shared cached
    :class:`NetAdapter` (``net=``), so NO model reload happens per game. The search
    is the byte-identical pure function of ``(state, seed)``; the only change is
    where the weights come from. Runs serially in the main process (``workers=1``)
    or inside a pool worker (``workers>1``); identical either way.
    """
    cfg = spec.cfg
    agent = MCTSAgent(
        net=_cached_net_adapter(spec.model_path),
        model_path=spec.model_path,
        config=cfg,
        seed=spec.agent_seed,
        name="MCTSGen",
    )
    buf = _SelfPlayBuffer()
    if spec.two_player:
        opponent = _make_opponent(spec.opponent_snapshot)
        finished, n_unenc = _generate_one_1v1(
            spec.track_seed,
            spec.split,
            game_seed=spec.game_seed,
            agent=agent,
            opponent=opponent,
            buf=buf,
            log_seats=spec.log_seats,
            traj_greedy=spec.traj_greedy,
        )
    else:
        finished, n_unenc = _generate_one_track(
            spec.track_seed,
            spec.split,
            game_seed=spec.game_seed,
            agent=agent,
            buf=buf,
            traj_greedy=spec.traj_greedy,
        )
    return _GameResult(
        split=spec.split,
        track_seed=spec.track_seed,
        buf=buf,
        finished=finished,
        n_unencodable=n_unenc,
        clones=agent.profile.clones,
        moves=agent.profile.moves,
        seconds=agent.profile.seconds,
    )


#: The split order in the assembled dataset (train rows then val rows), matching
#: the serial band loop so the deterministic merge reproduces the serial layout.
_SPLIT_RANK: dict[str, int] = {"train": 0, "val": 1}


def _merge_into(buf: _SelfPlayBuffer, results: list[_GameResult]) -> None:
    """Append every game's rows into ``buf`` in the DETERMINISTIC serial order.

    Sorted by ``(split_rank, track_seed)`` -- exactly the order the serial band
    loop produces (train band ascending, then val band ascending) -- with each
    game's rows kept in their original driver order. So the assembled arrays are
    byte-identical regardless of ``--workers`` (the hard determinism contract).
    """
    ordered = sorted(results, key=lambda r: (_SPLIT_RANK[r.split], r.track_seed))
    for r in ordered:
        gb = r.buf
        buf.obs.extend(gb.obs)
        buf.pi.extend(gb.pi)
        buf.mask.extend(gb.mask)
        buf.kind.extend(gb.kind)
        buf.round_num.extend(gb.round_num)
        buf.z.extend(gb.z)
        buf.track_seed.extend(gb.track_seed)
        buf.split.extend(gb.split)


def generate_dataset(args: argparse.Namespace) -> dict:
    """Generate the train+val self-play dataset; return a stats summary."""
    _assert_seed_bands_disjoint(args.tracks + args.val_tracks)

    buf = _SelfPlayBuffer()
    # Sprint C6: the perfect-info 1v1 win/loss mode (default OFF == the C2-C5 solo
    # generator, byte-for-byte). When on, the generator drives 2-seat games with
    # the two-player minimax search and backfills z = the realized ±1/0 outcome.
    two_player = bool(getattr(args, "two_player", False))
    opponent_snapshot = getattr(args, "opponent_snapshot", None)
    cfg = MCTSConfig(
        n_simulations=args.sims,
        dirichlet_eps=args.dirichlet_eps,
        dirichlet_alpha=args.dirichlet_alpha,
        temperature_moves=args.temperature_moves,
        root_selector=getattr(args, "root_selector", "puct"),
        two_player=two_player,
    )
    # C5 trajectory-greediness knob (the data-starvation fix): drive the *acted*
    # trajectory greedily so fewer races spin to MAX_ROUNDS, while the logged pi
    # target is unchanged. Default OFF = the C4 exploratory trajectory.
    traj_greedy = bool(getattr(args, "traj_greedy", False))

    # Sprint C7: number of parallel self-play workers. Default 1 == the serial
    # C2-C6 path, byte-for-byte. >1 distributes the per-track games across a
    # multiprocessing pool; the deterministic seed-sorted merge below makes the
    # assembled dataset byte-identical regardless of this value.
    workers = max(1, int(getattr(args, "workers", 1) or 1))

    bands = [
        ("train", _SELFPLAY_SEED_BASE, args.tracks),
        ("val", _SELFPLAY_SEED_BASE + args.tracks, args.val_tracks),
    ]

    t0 = time.perf_counter()
    races_total = 0
    races_dropped = 0
    total_unencodable = 0
    total_clones = 0
    total_moves = 0
    total_search_s = 0.0
    # Per-split finished-race counts (the C5 val-starvation precondition: the loop
    # hard-raises when the finished-only val split is empty, never silently trains
    # an uncontrolled critic -- the exact C4 mechanical bug).
    races_finished_by_split = {"train": 0, "val": 0}
    races_total_by_split = {"train": 0, "val": 0}

    # Sprint C6: net-vs-net logs BOTH seats; net-vs-frozen-snapshot logs only the
    # current-net seat (the snapshot is a fixed adversary, not a learner).
    log_seats = (0,) if (two_player and opponent_snapshot) else (0, 1)

    # Build the per-game specs in the serial band order (the per-band index ``i``
    # derives the agent/game seeds exactly as the serial loop did).
    specs: list[_GameSpec] = []
    for split, base, n_tracks in bands:
        for i in range(n_tracks):
            specs.append(_GameSpec(
                split=split,
                track_seed=base + i,
                agent_seed=args.seed + i,
                game_seed=args.game_seed + i,
                model_path=args.model,
                cfg=cfg,
                two_player=two_player,
                opponent_snapshot=opponent_snapshot,
                log_seats=log_seats,
                traj_greedy=traj_greedy,
            ))

    if workers == 1:
        results = [_run_one_game(spec) for spec in specs]
    else:
        # Spawn-safe pool (Windows): each worker re-imports this module and loads
        # the model ONCE via the process-level cache, then reuses it for its share
        # of games. ``imap_unordered`` is fine -- the merge re-sorts by seed, so
        # completion order does not affect the assembled (deterministic) dataset.
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(specs)) or 1) as pool:
            results = list(pool.imap_unordered(_run_one_game, specs))

    _merge_into(buf, results)

    # Accounting (summed across workers, deterministic-order for the race counts).
    for r in sorted(results, key=lambda r: (_SPLIT_RANK[r.split], r.track_seed)):
        races_total += 1
        races_total_by_split[r.split] += 1
        total_unencodable += r.n_unencodable
        if not r.finished:
            races_dropped += 1
        else:
            races_finished_by_split[r.split] += 1
        total_clones += r.clones
        total_moves += r.moves
        total_search_s += r.seconds

    elapsed = time.perf_counter() - t0

    if len(buf) == 0:
        raise RuntimeError("no self-play targets logged (check track/sim counts)")

    obs = np.asarray(buf.obs, dtype=np.float32)
    pi = np.asarray(buf.pi, dtype=np.float32)
    mask = np.asarray(buf.mask, dtype=bool)
    kind = np.asarray(buf.kind, dtype=np.int8)
    z = np.asarray(buf.z, dtype=np.float32)
    track_seed = np.asarray(buf.track_seed, dtype=np.int64)
    split = np.asarray(buf.split, dtype="S5")

    # Drop rows whose episode was MAX_ROUNDS-truncated (z is NaN) -- their MC
    # target is undefined and would poison V (the gen_value_data drop rule).
    keep = ~np.isnan(z)
    n_dropped_rows = int((~keep).sum())
    obs, pi, mask, kind, z, track_seed, split = (
        obs[keep], pi[keep], mask[keep], kind[keep], z[keep], track_seed[keep], split[keep]
    )

    if obs.shape[0] == 0:
        raise RuntimeError(
            "all episodes hit MAX_ROUNDS without finishing -- no valid rows"
        )
    if obs.shape[1] != OBS_DIM or mask.shape[1] != ACTION_DIM or pi.shape[1] != ACTION_DIM:
        raise RuntimeError(
            f"shape mismatch: obs {obs.shape} pi {pi.shape} mask {mask.shape} vs "
            f"contract OBS_DIM={OBS_DIM} ACTION_DIM={ACTION_DIM}"
        )
    # z-target sanity. Sprint C6: the win/loss z is bounded ``[−1, 1]`` (so the
    # old positive-z-is-a-bug assertion is INVERTED to a bound check). The solo
    # ``z = −rounds_remaining`` path keeps the ``z <= 0`` invariant unchanged.
    if two_player:
        if z.min() < -1.0 - 1e-6 or z.max() > 1.0 + 1e-6:
            raise AssertionError(
                f"win/loss z out of bounds (min={z.min()}, max={z.max()}): "
                "z must be the ±1/0 1v1 outcome in [−1, 1]"
            )
    elif z.max() > 1e-6:
        raise AssertionError(
            f"positive z target ({z.max()}): a value-target labeling bug "
            "(z must be −rounds_remaining <= 0)"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=obs,
        pi=pi,
        mask=mask,
        kind=kind,
        z=z,
        track_seed=track_seed,
        split=split,
    )

    # Provenance sidecar (codec version + FULL search config + seed band).
    meta = {
        "codec_version": CODEC_VERSION,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "num_players": 2 if two_player else 1,
        "generator": (
            "MCTSAgent (C6) perfect-info 1v1 win/loss self-play"
            if two_player
            else "MCTSAgent (C1) with self-play exploration ON"
        ),
        "value_target": "winloss" if two_player else "rounds",
        "prior_model": args.model,
        "search_config": {
            "n_simulations": cfg.n_simulations,
            "c_pw": cfg.c_pw,
            "alpha_pw": cfg.alpha_pw,
            "k_cap": cfg.k_cap,
            "c_init": cfg.c_init,
            "c_base": cfg.c_base,
            "fpu_reduction": cfg.fpu_reduction,
            "dirichlet_eps": cfg.dirichlet_eps,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "temperature_moves": cfg.temperature_moves,
            "root_selector": cfg.root_selector,
            "gumbel_m": cfg.gumbel_m,
            "gumbel_c_visit": cfg.gumbel_c_visit,
            "gumbel_c_scale": cfg.gumbel_c_scale,
            "traj_greedy": traj_greedy,
            "two_player": two_player,
            "opponent_snapshot": opponent_snapshot,
        },
        "selfplay_seed_base": _SELFPLAY_SEED_BASE,
        "train_tracks": args.tracks,
        "val_tracks": args.val_tracks,
        "seed": args.seed,
        "game_seed": args.game_seed,
        "n_rows": int(obs.shape[0]),
        "races_total": races_total,
        "races_dropped_max_rounds": races_dropped,
        "rows_dropped_max_rounds": n_dropped_rows,
        "unencodable_dropped": total_unencodable,
    }
    meta_path = os.path.splitext(args.out)[0] + ".selfplay.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    int_to_kind = {v: k.name for k, v in _KIND_TO_INT.items()}
    train_sel = split == b"train"
    val_sel = split == b"val"

    def kind_balance(sel: np.ndarray) -> dict[str, int]:
        ks, counts = np.unique(kind[sel], return_counts=True)
        return {int_to_kind[int(k)]: int(c) for k, c in zip(ks, counts)}

    # π entropy (mean over rows): a real visit distribution has entropy > 0; a
    # collapsed argmax (BC again) would be near 0 -- the C0 visit-budget go/no-go
    # surfaces here.
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(pi > 0, pi * np.log(pi), 0.0).sum(axis=1)
    summary = {
        "n_rows": int(obs.shape[0]),
        "n_train": int(train_sel.sum()),
        "n_val": int(val_sel.sum()),
        "races_total": races_total,
        "races_dropped_max_rounds": races_dropped,
        "rows_dropped_max_rounds": n_dropped_rows,
        "unencodable_dropped": total_unencodable,
        # Per-split finished counts (the C5 val-starvation precondition signal).
        "races_finished_train": races_finished_by_split["train"],
        "races_finished_val": races_finished_by_split["val"],
        "races_total_train": races_total_by_split["train"],
        "races_total_val": races_total_by_split["val"],
        "traj_greedy": traj_greedy,
        "z_mean": float(z.mean()),
        "z_min": float(z.min()),
        "pi_entropy_mean": float(ent.mean()),
        "pi_entropy_cards": float(ent[kind == _KIND_TO_INT[DecisionKind.CARDS]].mean())
        if (kind == _KIND_TO_INT[DecisionKind.CARDS]).any() else float("nan"),
        "ms_per_move": (total_search_s / total_moves * 1000.0) if total_moves else float("nan"),
        "clones_per_move": (total_clones / total_moves) if total_moves else float("nan"),
        "elapsed_s": round(elapsed, 1),
        "out": args.out,
        "meta_path": meta_path,
        "train_kind_balance": kind_balance(train_sel),
        "val_kind_balance": kind_balance(val_sel),
    }
    return summary


def _print_summary(s: dict) -> None:
    print("\n=== gen_selfplay summary ===")
    print(f"  wrote {s['n_rows']} target tuples to {s['out']}")
    print(f"  sidecar: {s['meta_path']}")
    print(
        f"  train: {s['n_train']} rows; val: {s['n_val']} rows (track-disjoint)"
    )
    print(
        f"  races: {s['races_total']} total, {s['races_dropped_max_rounds']} dropped "
        f"(MAX_ROUNDS) -> {s['rows_dropped_max_rounds']} rows dropped; "
        f"{s['unencodable_dropped']} off-table acted moves dropped"
    )
    print(
        f"  finished by split: train {s.get('races_finished_train', '?')}/"
        f"{s.get('races_total_train', '?')}  val {s.get('races_finished_val', '?')}/"
        f"{s.get('races_total_val', '?')}  (traj_greedy={s.get('traj_greedy')})"
    )
    print(f"  z (-rounds_remaining, floored): mean={s['z_mean']:.2f} min={s['z_min']:.0f}")
    print(
        f"  pi entropy (exploration signal): overall={s['pi_entropy_mean']:.3f} "
        f"cards={s['pi_entropy_cards']:.3f}  (~0 => collapsed argmax / BC)"
    )
    print(
        f"  search cost: ms/move={s['ms_per_move']:.2f} clones/move={s['clones_per_move']:.1f}"
    )
    print(f"  generation time: {s['elapsed_s']}s")
    print("  train decision-kind balance:")
    for k, c in sorted(s["train_kind_balance"].items()):
        print(f"    {k:<10} {c:>7}  ({c / max(1, s['n_train']) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/selfplay.npz",
                        help="output .npz path (sidecar .selfplay.json alongside)")
    parser.add_argument("--model", type=str, required=True,
                        help="SB3 MaskablePPO checkpoint for the search prior+value "
                             "(the C2 PRE-TRAINING / warm-prior net; .meta.json "
                             "sidecar required, codec v3)")
    parser.add_argument("--tracks", type=int, default=120,
                        help="number of GENERATED self-play TRAIN tracks")
    parser.add_argument("--val-tracks", type=int, default=30,
                        help="number of self-play VAL tracks (track-disjoint)")
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTSAgent n_simulations (C0 default 16)")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25,
                        help="root Dirichlet noise weight eps (C0 default 0.25)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="root Dirichlet concentration alpha (C0 default 0.5)")
    parser.add_argument("--temperature-moves", type=int, default=10,
                        help="plies sampling a~N (tau=1) before tau->0 (C0 default 10)")
    parser.add_argument("--root-selector", type=str, default="puct",
                        choices=["puct", "gumbel"],
                        help="root action selector + policy target (C4): 'puct' "
                             "(default, visit-count target) or 'gumbel' "
                             "(Gumbel top-m + Sequential Halving, completed-Q target)")
    parser.add_argument("--traj-greedy", action="store_true",
                        help="C5 data-starvation fix: drive the *acted* trajectory "
                             "greedily (most-visited searched edge) so fewer races "
                             "spin out; the logged pi target is unchanged (default "
                             "OFF = the C4 exploratory trajectory)")
    parser.add_argument("--two-player", action="store_true",
                        help="C6: perfect-info 1v1 win/loss self-play. Drives 2-seat "
                             "games with the two-player minimax search and backfills "
                             "z = the realized ±1/0 outcome (always defined). Default "
                             "OFF = the C2-C5 solo generator")
    parser.add_argument("--opponent-snapshot", type=str, default=None,
                        help="C6: opponent seat for net-vs-frozen 1v1 (a "
                             "FrozenSnapshotAgent checkpoint path, or 'strong'/'weak' "
                             "for the heuristic anchors). Omit for net-vs-net (both "
                             "seats are the current net and both log)")
    parser.add_argument("--workers", type=int, default=1,
                        help="C7: parallel self-play workers (multiprocessing pool). "
                             "Default 1 = serial (byte-for-byte the C2-C6 path); >1 "
                             "distributes per-track games across cores with a "
                             "deterministic seed-sorted merge (dataset byte-identical "
                             "regardless of --workers). Each worker loads the model "
                             "once via the process-level cache (no per-game reload)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base agent search seed (mixed per track)")
    parser.add_argument("--game-seed", type=int, default=8000,
                        help="base per-track game RNG seed")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny smoke run (6 train / 2 val tracks, sims=8)")
    args = parser.parse_args()

    if args.smoke:
        args.tracks = 6
        args.val_tracks = 2
        args.sims = 8

    print(
        f"gen_selfplay: tracks={args.tracks} val={args.val_tracks} sims={args.sims} "
        f"dirichlet=(eps={args.dirichlet_eps},alpha={args.dirichlet_alpha}) "
        f"T_moves={args.temperature_moves} root_selector={args.root_selector} "
        f"prior={args.model} "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})"
    )
    summary = generate_dataset(args)
    _print_summary(summary)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from _runlog import run_main

    run_main("gen_selfplay", main)
