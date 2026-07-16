"""Multi-seat / held-out evaluation harness for Direction-A policies (Sprint A7).

A7 is the instrument that defines "done" for Direction A: score a policy vs
weak/strong heuristics **across seat counts (2-6)** and **across track splits**
(the fixed Tiny-Heat bed vs the reserved held-out generated band) with Wilson
lower-bound confidence, plus a **fixed-anchor self-improvement probe** (the
statistically honest version of A5's disputed G2).

Design decisions (see ``docs/direction-A/A7-eval-harness.md``):

* **Reuse the generator's seed-namespace split.** The held-out set is
  :func:`track_sampler(params, base_seed=0) <heat.tracks.generator.track_sampler>`
  over seeds ``900_000 + i`` -- the identity/eval namespace reaching the reserved
  held-out band. Training namespaces use ``base_seed != 0`` and land in a
  structurally disjoint seed region, so a training campaign can never regenerate
  a held-out eval track (a test *verifies* this by fingerprint).
* **Games run through the A2/A5 machinery.** A policy seat rotates through the
  field, every other seat scripted (:class:`_EvalCollector`, promoted here from
  :mod:`heat.ml.selfplay.recipe`). No SB3 / ``evaluate_ml`` dependency.
* **"Win" = first place; placement reward reported alongside.** At >2 seats a
  binary top-half signal hides skill, so the headline metric is first-place rate
  with :func:`~heat.simulation.stats.wilson_interval` bounds against the
  ``1/num_players`` chance line, and mean placement reward as the graded
  secondary.
* **Heuristic-only baselines run the same code path.** :func:`evaluate_policy`
  accepts a :class:`~heat.agents.base.BaseAgent` in the policy seat too (it skips
  encoding and just plays), so the strong-vs-weak baseline grid is certified by
  the exact code it certifies.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import pickle
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.engine.driver import Decision, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml.opponents import opponent_action
from heat.ml.selfplay.multiseat import MultiSeatCollector
from heat.ml.selfplay.policy import PPOPolicy
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.ml.spaces import _placement_reward
from heat.simulation.stats import wilson_interval
from heat.tracks.generator import TrackGenParams, track_sampler

#: First seed of the reserved held-out eval band (identity namespace only). The
#: generator documents ``900_000+`` as the held-out band reachable ONLY from
#: ``base_seed == 0`` (see :mod:`heat.tracks.generator`).
HELDOUT_BASE: int = 900_000


@dataclass(frozen=True)
class _CellSpec:
    """Picklable identity for one ordered parallel evaluation cell."""

    index: int
    opponent: str
    opponent_index: int
    seat_count: int
    seat_index: int
    split: str
    split_index: int
    base_seed: int


_WORKER_POLICY: PPOPolicy | BaseAgent | None = None
_WORKER_OPPONENTS: dict[str, Callable[[], BaseAgent]] | None = None
_WORKER_SPLITS: dict[str, list[Track]] | None = None
_WORKER_GAMES: int = 0
_WORKER_DEVICE: torch.device | None = None


# ---------------------------------------------------------------------------
# Promoted eval collector (single implementation; recipe.py imports it back)
# ---------------------------------------------------------------------------


class _EvalCollector(MultiSeatCollector):
    """Collector that captures each game's terminal state (for eval scoring).

    Promoted from :mod:`heat.ml.selfplay.recipe` so there is exactly one
    implementation; the recipe imports it back for its periodic vs-weak /
    vs-oldest eval. Overrides only the :meth:`_on_game_end` seam to stash the
    final :class:`~heat.models.game_state.GameState`.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.last_state: GameState | None = None

    def _on_game_end(
        self, state: GameState, terminated: bool, truncated: bool
    ) -> None:
        self.last_state = state


# ---------------------------------------------------------------------------
# Held-out track split
# ---------------------------------------------------------------------------


def held_out_tracks(
    n: int = 20, params: TrackGenParams | None = None
) -> list[Track]:
    """Return ``n`` held-out generated tracks (seeds ``900_000 .. 900_000+n``).

    Uses ``track_sampler(params, base_seed=0)`` -- the identity/eval namespace,
    the only one that reaches the reserved held-out band. ``params=None`` (the
    default) draws from the full generator distribution;
    :func:`heat.ml.selfplay.tiny_heat.tiny_heat_params` may be passed for
    tiny-bed work.
    """
    sampler = track_sampler(params, base_seed=0)
    return [sampler(HELDOUT_BASE + i) for i in range(n)]


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCell:
    """One (opponent, seat_count, split) evaluation cell.

    Attributes:
        opponent: opponent label (e.g. ``"weak"`` / ``"strong"`` / ``"anchor"``).
        seat_count: total seats in the game.
        split: track-split label (e.g. ``"tiny"`` / ``"heldout"`` / ``"anchor"``).
        games: games played in this cell.
        wins: games the policy seat finished **first**.
        win_rate: ``wins / games``.
        wilson_lb / wilson_ub: Wilson 95% interval on the first-place rate.
        chance: the chance-line first-place rate ``1 / seat_count`` (or ``0.5``
            for the two-player anchor probe).
        mean_placement_reward: mean signed placement reward (the graded secondary).
    """

    opponent: str
    seat_count: int
    split: str
    games: int
    wins: int
    win_rate: float
    wilson_lb: float
    wilson_ub: float
    chance: float
    mean_placement_reward: float

    def beats_chance(self, lb_margin: float = 0.0) -> bool:
        """True if the Wilson lower bound clears the chance line by ``lb_margin``."""
        return self.wilson_lb >= self.chance + lb_margin


@dataclass(frozen=True)
class EvalReport:
    """A grid of :class:`EvalCell`s plus the optional anchor probe.

    Attributes:
        cells: one cell per (opponent, seat_count, split).
        games_per_cell: games run per grid cell.
        seed: base seed the grid was run under.
        anchor: the optional :func:`evaluate_vs_anchor` cell.
    """

    cells: list[EvalCell]
    games_per_cell: int
    seed: int
    anchor: EvalCell | None = field(default=None)

    def clears_bar(
        self, opponent: str, lb_margin: float = 0.0, *, split: str | None = None
    ) -> bool:
        """True if ``opponent``'s Wilson-LB clears chance by ``lb_margin`` in
        every seat count (the A8 skill-bar predicate).

        With ``split`` given, only that split's cells are considered; otherwise
        all splits must clear. Returns ``False`` if no matching cell exists.
        """
        cells = [
            c
            for c in self.cells
            if c.opponent == opponent and (split is None or c.split == split)
        ]
        if not cells:
            return False
        return all(c.beats_chance(lb_margin) for c in cells)

    def to_markdown(self) -> str:
        """Render the grid (and anchor probe, if any) as a markdown table."""
        lines: list[str] = []
        lines.append(
            f"Eval grid (games/cell={self.games_per_cell}, seed={self.seed}):"
        )
        lines.append("")
        lines.append(
            "| opponent | split | seats | games | wins | win% | Wilson LB | "
            "Wilson UB | chance | mean reward | LB>chance |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in sorted(
            self.cells, key=lambda c: (c.opponent, c.split, c.seat_count)
        ):
            beats = "yes" if c.wilson_lb > c.chance else "no"
            lines.append(
                f"| {c.opponent} | {c.split} | {c.seat_count} | {c.games} | "
                f"{c.wins} | {c.win_rate * 100:.1f}% | {c.wilson_lb:.3f} | "
                f"{c.wilson_ub:.3f} | {c.chance:.3f} | "
                f"{c.mean_placement_reward:+.3f} | {beats} |"
            )
        if self.anchor is not None:
            a = self.anchor
            verdict = (
                "CLIMBS (LB > 0.5)"
                if a.wilson_lb > 0.5
                else "does NOT clear 0.5"
            )
            lines.append("")
            lines.append("Fixed-anchor self-improvement probe (chance = 0.500):")
            lines.append("")
            lines.append(
                "| games | wins | win% | Wilson LB | Wilson UB | verdict |"
            )
            lines.append("|---|---|---|---|---|---|")
            lines.append(
                f"| {a.games} | {a.wins} | {a.win_rate * 100:.1f}% | "
                f"{a.wilson_lb:.3f} | {a.wilson_ub:.3f} | {verdict} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Game execution + scoring
# ---------------------------------------------------------------------------


def _episode_flags(state: GameState) -> tuple[bool, bool]:
    """Return ``(terminated, truncated)`` -- the same rule the collector uses."""
    terminated = state.is_game_over
    truncated = (not terminated) and (state.round_num > MAX_ROUNDS)
    return terminated, truncated


def _first_place(state: GameState, seat: int) -> bool:
    """True if ``seat`` finished first (``finish_order == 1``).

    A truncated game (nobody finished) is not a first-place game for anyone --
    the honest signal (no fabricated winner at a time-limit cutoff).
    """
    player = state.get_player(seat)
    return player.finished and player.finish_order == 1


def _play_scripted_game(
    track: Track, num_players: int, agents: dict[int, BaseAgent], seed: int
) -> GameState:
    """Play one all-scripted game and return the terminal state.

    Mirrors :meth:`MultiSeatCollector._play_one_game`'s driver loop exactly (same
    ``run_round_driver`` + ``opponent_action`` plumbing, same episode-flag rule),
    but every seat is a scripted :class:`~heat.agents.base.BaseAgent`. Used for
    the heuristic-in-policy-seat baseline grids so they share the harness's
    scoring/rotation path.
    """
    state = GameState.create(track, num_players, seed=seed)
    for player in state.players:  # mirror Game.__init__: everyone on lap 1.
        player.lap = 1

    gen = run_round_driver(state)
    send_value: object = None
    while True:
        terminated, truncated = _episode_flags(state)
        if terminated or truncated:
            break
        try:
            decision: Decision = gen.send(send_value)
        except StopIteration:
            terminated, truncated = _episode_flags(state)
            if terminated or truncated:
                break
            gen = run_round_driver(state)
            send_value = None
            continue
        seat = decision.player_id
        send_value = opponent_action(agents[seat], decision, state)
    return state


def _run_cell(
    policy: PPOPolicy | BaseAgent,
    opp_factory: Callable[[], BaseAgent],
    seat_count: int,
    tracks: Sequence[Track],
    games: int,
    base_seed: int,
    device: torch.device,
) -> tuple[int, float]:
    """Play ``games`` of ``policy`` vs an ``opp_factory`` field; return
    ``(first_place_wins, total_placement_reward)``.

    The policy seat rotates game-to-game; every other seat is a fresh
    ``opp_factory()`` instance; the track cycles through ``tracks``. A
    :class:`~heat.agents.base.BaseAgent` policy skips encoding and plays through
    :func:`_play_scripted_game`; a :class:`PPOPolicy` **samples** (``act``) via
    :class:`_EvalCollector`. Both paths draw the per-game seed from a freshly
    seeded RNG identically, so equal ``g`` plays the same game. Policy sampling
    also runs inside a forked, per-game-seeded torch RNG context: repeated
    evaluation of the same checkpoint is byte-for-byte reproducible and does
    not consume the caller/trainer's torch RNG stream.
    """
    wins = 0
    total_reward = 0.0
    n_tracks = len(tracks)
    for g in range(games):
        policy_seat = g % seat_count
        track = tracks[g % n_tracks]
        rng = np.random.default_rng(base_seed + g)
        scripted: dict[int, BaseAgent] = {
            s: opp_factory() for s in range(seat_count) if s != policy_seat
        }
        if isinstance(policy, BaseAgent):
            game_seed = int(rng.integers(0, 2**31 - 1))
            agents = dict(scripted)
            # A fresh policy-seat agent per game, mirroring the per-seat
            # opponents and run_batch's documented per-game-factory contract, so
            # a stateful scripted agent cannot carry state across games. (In
            # practice StrongHeuristicAgent's per-turn plan cache is self-
            # validating, so this is defensive rather than load-bearing.)
            agents[policy_seat] = copy.deepcopy(policy)
            state = _play_scripted_game(track, seat_count, agents, game_seed)
        else:
            collector = _EvalCollector(track, seat_count, scripted_seats=scripted)
            cuda_devices: list[int] = []
            if device.type == "cuda":
                cuda_devices = [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]
            policy_seed = (base_seed + g + 0xA7E7A7) % (2**63 - 1)
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(policy_seed)
                # n_steps=1 -> one complete game (finish-the-in-flight-game).
                collector.collect(policy, 1, device, rng, gamma=1.0)
            assert collector.last_state is not None  # a game always ends
            state = collector.last_state
        total_reward += _placement_reward(state, policy_seat)
        if _first_place(state, policy_seat):
            wins += 1
    return wins, total_reward


def _make_cell(
    opponent: str,
    seat_count: int,
    split: str,
    games: int,
    wins: int,
    total_reward: float,
    *,
    chance: float | None = None,
) -> EvalCell:
    """Assemble an :class:`EvalCell` from raw counts (Wilson bounds computed)."""
    lb, ub = wilson_interval(wins, games)
    return EvalCell(
        opponent=opponent,
        seat_count=seat_count,
        split=split,
        games=games,
        wins=wins,
        win_rate=wins / games if games else 0.0,
        wilson_lb=lb,
        wilson_ub=ub,
        chance=(1.0 / seat_count) if chance is None else chance,
        mean_placement_reward=total_reward / games if games else 0.0,
    )


def _as_track_list(value: Track | Sequence[Track]) -> list[Track]:
    """Normalize a split value (single track or list) to a list of tracks."""
    if isinstance(value, Track):
        return [value]
    return list(value)


def _cell_seed(seed: int, oi: int, si: int, pi: int) -> int:
    """Deterministic, disjoint per-cell base seed (opp/seat/split indices)."""
    return seed * 1_000_000 + oi * 100_000 + si * 10_000 + pi * 1_000


def _preflight_pickle(value: object, name: str) -> None:
    """Raise a named error before spawning when one parallel input is invalid."""
    try:
        pickle.dumps(value)
    except (pickle.PickleError, TypeError, AttributeError) as exc:
        raise TypeError(f"parallel evaluation input {name} is not picklable") from exc


def _init_eval_worker(
    policy: PPOPolicy | BaseAgent,
    opponents: dict[str, Callable[[], BaseAgent]],
    splits: dict[str, list[Track]],
    games_per_cell: int,
) -> None:
    """Install one policy, opponent map, and track grid in each CPU worker."""
    global _WORKER_POLICY, _WORKER_OPPONENTS, _WORKER_SPLITS
    global _WORKER_GAMES, _WORKER_DEVICE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_POLICY = policy
    _WORKER_OPPONENTS = opponents
    _WORKER_SPLITS = splits
    _WORKER_GAMES = games_per_cell
    _WORKER_DEVICE = torch.device("cpu")


def _run_parallel_cell(spec: _CellSpec) -> tuple[int, EvalCell]:
    """Run one initialized worker's cell and preserve its parent order index."""
    if (
        _WORKER_POLICY is None
        or _WORKER_OPPONENTS is None
        or _WORKER_SPLITS is None
        or _WORKER_DEVICE is None
    ):  # pragma: no cover - initializer contract
        raise RuntimeError("parallel evaluation worker was not initialized")
    wins, total = _run_cell(
        _WORKER_POLICY,
        _WORKER_OPPONENTS[spec.opponent],
        spec.seat_count,
        _WORKER_SPLITS[spec.split],
        _WORKER_GAMES,
        spec.base_seed,
        _WORKER_DEVICE,
    )
    return spec.index, _make_cell(
        spec.opponent,
        spec.seat_count,
        spec.split,
        _WORKER_GAMES,
        wins,
        total,
    )


# ---------------------------------------------------------------------------
# Public evaluation entry points
# ---------------------------------------------------------------------------


def evaluate_policy(
    policy: PPOPolicy | BaseAgent,
    *,
    opponents: dict[str, Callable[[], BaseAgent]] | None = None,
    seat_counts: tuple[int, ...] = (2, 3, 4, 6),
    splits: dict[str, Track | Sequence[Track]] | None = None,
    games_per_cell: int = 50,
    seed: int = 0,
    device: torch.device | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> EvalReport:
    """Score ``policy`` over the (opponent x seat_count x split) grid.

    For each cell the policy seat rotates through the field game-to-game, every
    other seat is a fresh opponent instance, and the track cycles through the
    split's list. The policy **samples** (``act``); a
    :class:`~heat.agents.base.BaseAgent` in the policy seat plays directly (the
    heuristic-only baseline path, §5 G1).

    Args:
        policy: the :class:`PPOPolicy` under test, or a
            :class:`~heat.agents.base.BaseAgent` for a heuristic-only baseline grid.
        opponents: ``{label: factory}`` scripted-opponent factories; defaults to
            ``{"weak": HeuristicAgent, "strong": StrongHeuristicAgent}``.
        seat_counts: seat counts to sweep (default ``(2, 3, 4, 6)``).
        splits: ``{label: track | tracks}``; defaults to
            ``{"tiny": tiny_heat_track(), "heldout": held_out_tracks()}``.
        games_per_cell: games per grid cell (default 50).
        seed: base seed (per-cell seeds derive from it deterministically).
        device: torch device (defaults to CPU).
        parallel: run independent cells in spawned CPU processes when true.
        max_workers: optional process limit for parallel evaluation.

    Returns:
        An :class:`EvalReport` with one :class:`EvalCell` per grid cell.
    """
    opps: dict[str, Callable[[], BaseAgent]]
    if opponents is None:
        opps = {"weak": HeuristicAgent, "strong": StrongHeuristicAgent}
    else:
        opps = opponents
    if splits is None:
        from heat.ml.selfplay.tiny_heat import tiny_heat_track

        splits = {"tiny": tiny_heat_track(), "heldout": held_out_tracks()}
    if device is None:
        device = torch.device("cpu")

    if parallel:
        if device.type == "cuda":
            raise ValueError("parallel evaluation supports CPU policies only")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive")
        _preflight_pickle(policy, "policy")
        for label, factory in opps.items():
            _preflight_pickle(factory, f"opponents[{label!r}]")
        normalized_splits = {
            split_name: _as_track_list(split_val)
            for split_name, split_val in splits.items()
        }
        for label, tracks in normalized_splits.items():
            _preflight_pickle(tracks, f"splits[{label!r}]")

        specs: list[_CellSpec] = []
        for oi, opp_name in enumerate(opps):
            for si, seat_count in enumerate(seat_counts):
                for pi, split_name in enumerate(normalized_splits):
                    specs.append(
                        _CellSpec(
                            index=len(specs),
                            opponent=opp_name,
                            opponent_index=oi,
                            seat_count=seat_count,
                            seat_index=si,
                            split=split_name,
                            split_index=pi,
                            base_seed=_cell_seed(seed, oi, si, pi),
                        )
                    )
        ordered: list[EvalCell | None] = [None] * len(specs)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_eval_worker,
            initargs=(policy, opps, normalized_splits, games_per_cell),
        ) as executor:
            futures: dict[Future[tuple[int, EvalCell]], _CellSpec] = {
                executor.submit(_run_parallel_cell, spec): spec for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    index, cell = future.result()
                except BaseException as exc:
                    identity = f"{spec.opponent}/{spec.seat_count}/{spec.split}"
                    raise RuntimeError(
                        f"parallel evaluation cell {identity!r} failed"
                    ) from exc
                ordered[index] = cell
        assert all(cell is not None for cell in ordered)
        return EvalReport(
            cells=[cell for cell in ordered if cell is not None],
            games_per_cell=games_per_cell,
            seed=seed,
        )

    cells: list[EvalCell] = []
    for oi, (opp_name, opp_factory) in enumerate(opps.items()):
        for si, seat_count in enumerate(seat_counts):
            for pi, (split_name, split_val) in enumerate(splits.items()):
                tracks = _as_track_list(split_val)
                base = _cell_seed(seed, oi, si, pi)
                wins, total = _run_cell(
                    policy,
                    opp_factory,
                    seat_count,
                    tracks,
                    games_per_cell,
                    base,
                    device,
                )
                cells.append(
                    _make_cell(
                        opp_name,
                        seat_count,
                        split_name,
                        games_per_cell,
                        wins,
                        total,
                    )
                )
    return EvalReport(cells=cells, games_per_cell=games_per_cell, seed=seed)


def evaluate_vs_anchor(
    policy: PPOPolicy | BaseAgent,
    anchor: SnapshotAgent,
    *,
    num_players: int = 2,
    track: Track,
    games: int = 200,
    seed: int = 0,
    device: torch.device | None = None,
) -> EvalCell:
    """The fixed-anchor self-improvement probe (settles A5's disputed G2).

    Plays ``games`` of ``policy`` vs a **fixed** ``anchor``
    (:class:`~heat.ml.selfplay.snapshots.SnapshotAgent`, typically an early-
    training checkpoint), the policy seat rotating each game. Returns an
    :class:`EvalCell` whose ``chance`` line is ``0.5`` and whose Wilson lower
    bound is the powered verdict on "does the policy climb past its earlier
    self". At 200 games the standard error is ~0.035 (vs the A5 gate's
    underpowered 40).

    Args:
        policy: the current policy (samples via ``act``).
        anchor: the frozen earlier-self opponent.
        num_players: seats (default 2 -- the fixed-anchor probe is head-to-head).
        track: the fixed track the probe runs on.
        games: probe game count (default 200).
        seed: base seed for the probe games.
        device: torch device (defaults to CPU).
    """
    if device is None:
        device = torch.device("cpu")
    wins, total = _run_cell(
        policy, lambda: anchor, num_players, [track], games, seed, device
    )
    return _make_cell(
        "anchor", num_players, "anchor", games, wins, total, chance=0.5
    )
