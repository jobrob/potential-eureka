"""Tiny-Heat proving ground: a deliberately small Heat configuration (Sprint A1).

C6's #1 lesson is to *prove a method where iteration is cheap before paying for
scale*. Tiny-Heat is the "Connect4-equivalent" bed for Direction A: a deliberately
small Heat variant where a full method experiment is a minutes-loop, not a day. Per
the A1 design, it stays **honest on the hard axes** (stochastic card draws, hidden
opponent hands, multiplayer) and only shrinks the *size* of the game, not its
*nature*.

Key design decision (A1 design §2): **shrink the track + seat count, NOT the
engine.** The :class:`~heat.models.track.Track` is pure data, so a tiny track is
just a small ``Track``; :class:`~heat.ml.env.HeatEnv` already accepts a fixed track
or a sampler, and the A0 custom-PPO :func:`~heat.ml.selfplay.ppo.train` already
takes ``track=`` and ``num_players`` (via :class:`~heat.ml.selfplay.ppo.A0Config`).
So the tiny **track + seat count** needs essentially no new engine code. The
deck/hand (``cards.py`` / ``player_state.py`` / ``game_state.py``, hand size 7) is
deliberately left at full size -- the deck/hand fork is **deferred** (design §6) and
only triggered if A1's sanity sims show the full-deck tiny game is degenerate.

This module is purely additive: it builds a tiny track, a tiny generation
distribution, an :class:`~heat.ml.selfplay.ppo.A0Config` preset, and a convenience
env builder -- and touches nothing in the engine or the existing ml modules.
"""

from __future__ import annotations

from heat.models.track import Corner, Space, Track
from heat.ml.env import HeatEnv, TrackSource
from heat.ml.selfplay.ppo import A0Config
from heat.tracks.generator import TrackGenParams, track_sampler
from heat.tracks.validate import validate_track

#: Name carried by the fixed tiny track (so logs/eval can identify the bed).
TINY_HEAT_NAME: str = "tiny-heat"


def tiny_heat_track() -> Track:
    """Return the fixed, hand-built Tiny-Heat track (validated).

    Geometry (calibrated against ``scripts/tiny_heat_sanity.py`` so the game is
    non-degenerate, not a 2-round sprint -- design §4.2 / §5 G3):

    * **14 spaces**, **1 lap**.
    * **2 corners** (length 2 each) with speed limits in ``{3, 4}`` -- tight enough
      that a fast approach forces a real heat / gear / CARDS decision at the corner,
      so the hard axes still bite.
    * **6 start positions** on straights (count ``>= MAX_PLAYERS`` so the validator
      accepts it and any 2-6 seat game fits), laid out on the wrap-around straight
      that includes the start line, mirroring the static-track grid convention
      (start line ``0`` rotated last, as USA's ``[1, 2, 3, 4, 5, 0]``).
    * Mostly 1-lane with a single 2-lane space, keeping slipstream / overtaking
      reachable without widening the whole loop.

    The track is built directly as a :class:`~heat.models.track.Track` and passed
    through :func:`~heat.tracks.validate.validate_track` (asserted valid here -- the
    validator is the raceability contract, design §7).
    """
    # 14 spaces, indices 0..13. Two corners at 5-6 and 10-11 (length 2 each).
    # Straights: 0,1,2,3,4, 7,8,9, 12,13 -> 10 straight spaces (>= 6 starts fit).
    # A single 2-lane space (index 8) keeps slipstream / overtake reachable.
    corner_spaces = {5, 6, 10, 11}
    spaces = [
        Space(index=i, lanes=2 if i == 8 else 1) for i in range(14)
    ]
    corners = [
        Corner(start=5, end=6, speed_limit=3),
        Corner(start=10, end=11, speed_limit=4),
    ]
    # Start grid on straights around the start line, start line (0) rotated last
    # (the static-track convention). All entries are non-corner spaces.
    start_positions = [1, 2, 3, 4, 12, 0]
    assert not (set(start_positions) & corner_spaces), "start grid must avoid corners"

    track = Track(
        name=TINY_HEAT_NAME,
        spaces=spaces,
        corners=corners,
        start_positions=start_positions,
        laps=1,
    )
    # The validator is the raceability contract: assert the hand-built track is
    # valid at construction rather than hand-waving the geometry (design §7).
    validate_track(track)
    return track


def tiny_heat_params() -> TrackGenParams:
    """Return a tiny generation distribution for randomized Tiny-Heat beds.

    A narrow :class:`~heat.tracks.generator.TrackGenParams` matching the fixed
    track's scale: short single-lap tracks with 1-2 short corners. Used later
    (e.g. A8 track-distribution generalization) for a per-episode *tiny
    distribution*; it is **not** required by the A1 gate -- the fixed
    :func:`tiny_heat_track` is the A1 bed.

    Validation notes (design §4.2):

    * ``length_range[0] = 12 >= MIN_TRACK_LENGTH (8)``.
    * ``min_start_positions`` defaults to ``MAX_PLAYERS (6)``. A 12-space track
      with two length-2 corners leaves >= 8 straight spaces, so 6 contiguous
      start straights are always available -- the default is kept (verified by
      the generator drawing valid tracks in the sanity harness / tests).
    """
    return TrackGenParams(
        length_range=(12, 16),
        num_corners_range=(1, 2),
        speed_limit_choices=(2, 3, 4),
        laps=1,
        corner_len_range=(1, 2),
    )


def tiny_heat_config(**overrides: object) -> A0Config:
    """Return an :class:`~heat.ml.selfplay.ppo.A0Config` preset for Tiny-Heat.

    Preset (design §4.2): ``num_players=2`` (2 seats by default; 3 is allowed and
    fits the grid), ``randomize_seat=True``, and a smaller ``n_steps`` rollout
    sized to the short episodes (the full default 2048 would span many tiny
    episodes per update; 512 keeps updates frequent without starving GAE). The
    minibatch / epoch / PPO defaults are inherited from :class:`A0Config`.

    Any field is overridable via keyword, e.g.
    ``tiny_heat_config(num_players=3, total_timesteps=100_000)``.
    """
    defaults: dict[str, object] = {
        "num_players": 2,
        "randomize_seat": True,
        "n_steps": 512,
    }
    defaults.update(overrides)
    return A0Config(**defaults)  # type: ignore[arg-type]


def make_tiny_env(
    num_players: int = 2,
    *,
    distribution: bool = False,
    seed: int | None = None,
    opponents: object = None,
) -> HeatEnv:
    """Build a Tiny-Heat :class:`~heat.ml.env.HeatEnv` (design §4.2).

    Args:
        num_players: total seats (learner + opponents); 2 by default, 3 fits.
        distribution: when ``True`` use a per-episode tiny-track sampler
            (``track_sampler(tiny_heat_params())``); when ``False`` (default) use
            the fixed :func:`tiny_heat_track` -- the deterministic A1 bed.
        seed: unused for the fixed track; the per-episode track/seat/state are
            still a deterministic function of the *episode* seed passed to
            :meth:`HeatEnv.reset`. Accepted for call-site symmetry / future use.
        opponents: opponent spec forwarded to :class:`HeatEnv` (defaults to the
            env's own default, :class:`~heat.agents.heuristic_agent.HeuristicAgent`).

    Returns:
        A :class:`HeatEnv` with ``randomize_seat=True`` over the tiny bed.
    """
    track: TrackSource
    if distribution:
        track = track_sampler(tiny_heat_params())
    else:
        track = tiny_heat_track()

    return HeatEnv(
        track=track,
        num_players=num_players,
        opponents=opponents,  # type: ignore[arg-type]
        randomize_seat=True,
    )
