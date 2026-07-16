# Sprint A0 — Substrate decision + self-play training skeleton

> **Status:** **implemented and complete** (2026-06-24). The custom PPO substrate
> was selected, the training skeleton landed, and gate G2 passed. Retained as the
> decision record for later Direction A work.

## 1. Purpose

Direction A replaces the single-seat SB3 `MaskablePPO` stack with a loop that can later
do (A2) **N-seat shared-policy self-play**, (A3) a **variable-length legal-action
dot-product head**, and (A4) a **structured hidden-info encoder**. Before building any of
that, A0 settles one question and de-risks it with a thin spike:

> **Do we own the PPO training loop (a minimal custom PPO), or extend SB3 `MaskablePPO`?**

A0 is **plumbing + a decision**, not architecture. It deliberately reuses the *existing*
flat observation, the existing `action_codec` masking, and the existing `HeatEnv` so the
only new thing under test is **the training loop itself**. The architecture pieces
(dot-product head, hidden-info encoder, true multi-seat rollouts) are A2–A4 and are
explicitly out of scope here — but A0's interfaces must not block them.

## 2. The decision

### Criteria

| Criterion | Why it matters for Direction A |
|---|---|
| **Variable legal-action head** | A3 needs to score *only currently-legal* actions by dot-product. A fixed-width head fights this. |
| **Control of the rollout loop** | A2 needs per-seat trajectories from N shared-policy seats in one game; the trainer must own how rollouts are collected. |
| **GPU leaf-batching path** | Throughput (Direction D later) wants many positions in one forward pass; the loop must not hide the batch boundary. |
| **Debuggability / dependency weight** | Our collapse history means we must see and instrument every loss term. |
| **Maturity / risk** | SB3 is battle-tested; a custom loop is code we must get right (GAE signs, masking, advantage normalization). |

### Assessment

- **SB3 `MaskablePPO` (extend).** Its action space is a fixed `Discrete(ACTION_DIM)` with a
  boolean mask — it *cannot natively* express a variable-length legal-action dot-product
  head (A3) without subclassing the distribution and policy internals. `.learn()` owns the
  rollout loop and assumes one acting agent per env step, so true N-seat per-seat
  trajectory collection (A2) means fighting the `VecEnv`/`RolloutBuffer` contract. We would
  end up overriding so much of SB3 that we own the hard parts anyway, minus the
  transparency. **Evaluated analytically + by a small API probe — not by building a full
  parallel skeleton** (that effort is the thing we're trying to avoid committing to).
- **Minimal custom PPO (own the loop).** ~300–400 lines: rollout buffer, GAE, clipped
  surrogate + value loss + entropy bonus, Adam step, a swappable policy interface. Gives us
  the variable head (A3), per-seat rollouts (A2), and an explicit batch boundary (Direction
  D) for free, and every loss term is ours to log. Cost: we must get PPO right and keep it
  correct — mitigated by the A0 sanity gate (§5, G2) and a smoke test.

### Recommendation (provisional, confirmed by the spike)

**Build the minimal custom PPO.** The two defining Direction-A requirements (variable head,
multi-seat rollouts) are exactly where SB3 resists hardest, and the custom loop is small
and well-understood. The A0 skeleton **is** the spike that confirms this; G3 (§5) records
the final decision. *What would flip it:* if the spike shows the custom loop cannot match a
reference PPO's learning behavior on the sanity probe (G2) after reasonable debugging, fall
back to subclassing SB3's maskable distribution instead.

The existing SB3 path (`scripts/train_ml.py`, `heat.ml.model`, `heat.ml.training`) stays
**untouched and working** — A0 is purely additive so we retain a known-good baseline.

## 3. Scope

**In scope**
- A new self-play training spine package `src/heat/ml/selfplay/` (custom PPO).
- A swappable **policy interface** that A3/A4 will later re-implement without touching the
  trainer.
- Rollout collection by **driving the existing `HeatEnv`** (single learning seat,
  scripted opponents) — enough to validate GAE + a PPO update end-to-end.
- A minimal CLI to run it, and a smoke test.
- The decision recorded in this doc's §6 (filled in at the end).

**Out of scope (named so they aren't accidentally built here)**
- True N-seat shared-policy self-play and per-seat trajectories → **A2**.
- The legal-action dot-product head → **A3** (A0 uses a masked categorical placeholder
  behind the policy interface).
- The structured hidden-info / card-embedding encoder → **A4** (A0 uses the existing flat
  `encode_observation`).
- Tiny-Heat variant → **A1**. A0 runs on the existing full env at small step counts.
- Vectorization / throughput tuning, entropy/opponent-pool anti-collapse tuning, eval gate
  → later sprints. A0 wants *correctness of the loop*, not skill.

## 4. Design

### 4.1 Module layout (all new, additive)

```
src/heat/ml/selfplay/
  __init__.py
  policy.py      # HeatPolicy interface + A0 masked-categorical implementation
  buffer.py      # RolloutBuffer + GAE
  ppo.py         # PPO update (clipped surrogate, value loss, entropy) + train loop
scripts/
  train_selfplay_a0.py   # minimal CLI to run the skeleton
tests/
  test_selfplay_ppo_smoke.py
```

### 4.2 Policy interface (the contract A3/A4 must honor)

`HeatPolicy(nn.Module)` — keep the signatures below stable so later sprints swap *internals*
only:

```python
class HeatPolicy(nn.Module):
    def act(self, obs, mask):
        # obs: (B, OBS_DIM) float32; mask: (B, ACTION_DIM) bool
        # returns: action (B,) long, logp (B,), value (B,), entropy (B,)
        ...
    def evaluate(self, obs, actions, mask):
        # returns: logp (B,), value (B,), entropy (B,)  -- for the PPO update
        ...
```

A0 implementation: a small MLP trunk over the flat obs → (policy logits over `ACTION_DIM`,
value scalar). Apply the mask by `logits.masked_fill(~mask, -inf)`; build a `Categorical`;
sample in `act`, score given `actions` in `evaluate`. **No illegal action may ever receive
mass** (assert in the smoke test). When A3 lands, `mask` generalizes to "the set of legal
action feature vectors" and the head becomes a dot-product — but `act`/`evaluate` keep
these shapes/return tuples.

### 4.3 Rollout collection (A0 = single seat, reuse `HeatEnv`)

Drive the existing env directly — it already returns obs, reward, and the mask:

```
obs, info = env.reset(seed); mask = info["action_mask"]
loop for n_steps:
    action, logp, value, _ = policy.act(obs, mask)
    next_obs, reward, term, trunc, info = env.step(action)
    buffer.add(obs, action, logp, value, reward, done=term or trunc, mask)
    obs, mask = (reset if done else next_obs, info["action_mask"])
bootstrap last value; buffer.compute_gae(gamma, gae_lambda)
```

Use `randomize_seat=True` and scripted `HeuristicAgent`/`RandomAgent` opponents (existing
defaults). One env is fine; a short list of envs stepped in sequence is acceptable but not
required — vectorization is a later concern.

### 4.4 PPO update

Standard clipped PPO over the buffer: minibatch SGD for `n_epochs`; loss =
`policy_loss + vf_coef * value_loss - ent_coef * entropy`, with advantage normalization,
`clip_range`, `max_grad_norm` clipping. Reuse the hyperparameter *names/defaults* from
`heat.ml.model.PPOConfig` where sensible (gamma 0.999, gae_lambda 0.95, clip 0.2,
ent_coef 0.01, etc.) so config stays familiar; a small local `A0Config` dataclass is fine
— do **not** entangle with `PPOConfig`/SB3.

### 4.5 Device

Honor CPU/CUDA via the existing `heat.ml.model.resolve_device` (auto → CUDA if available,
else CPU; never crash on a CPU box).

## 5. Acceptance gate (exit criteria)

- **G1 — end-to-end runs.** `scripts/train_selfplay_a0.py` collects a rollout and performs
  ≥1 PPO update with no error, on CPU (and on CUDA if `torch.cuda.is_available()`). The
  smoke test asserts **no illegal action is ever sampled** (cross-check against
  `legal_action_mask`).
- **G2 — gradients flow (the real spike result).** Over a short run (e.g. ~50k steps,
  fixed short setup, `RandomAgent` opponents), **mean episode return improves beyond noise**
  vs. the initial policy (compare a frozen-policy control, or first-vs-last eval windows).
  This is the one test that catches the classic custom-PPO sign/normalization bugs — it is
  the evidence that the loop is correct, not just runnable. *Not a skill bar* (that is A5+).
- **G3 — decision recorded.** §6 below filled in: criteria table outcome, chosen substrate,
  rationale, and "what would flip it."
- **G4 — non-regression.** The full existing test suite passes and the SB3 path is
  unchanged (A0 added no edits to `model.py` / `training.py` / `env.py`). `ruff` and `mypy`
  clean on new files, matching repo config.

Run tests with the repo convention: `PYTHONPATH=src python -m pytest` (and the new smoke
test specifically). Keep the smoke test fast (tiny net, few hundred steps) so CI stays
quick; G2's longer probe can be a manual/marked run, not part of default CI.

## 6. Decision record (closed 2026-06-24)

- **Chosen substrate: minimal custom PPO** (`src/heat/ml/selfplay/`). Confirmed — the
  skeleton spike built and trained correctly end to end.
- **Criteria outcome:** decided by the two requirements where SB3-Maskable resists
  hardest — the variable legal-action head (A3) and owning the per-seat rollout loop
  (A2). The custom loop expresses both natively and keeps every loss term inspectable;
  SB3 would have meant overriding its distribution + rollout buffer to the point of
  owning the hard parts anyway. The existing SB3 path is left untouched as a baseline.
- **G2 result (PASS).** Highest-SNR setup (fixed USA track, 2 players, `RandomAgent`
  opponent, pure-sparse placement reward, single env, CPU, seed 0). Mean episode return
  (range [-1, +1]) climbed from **−0.13 at init → +0.78 by ~35k steps → ~+0.85 plateau by
  ~80k**, with entropy decaying gently (1.18→1.0, no collapse) and value loss falling
  (0.16→0.10). Unambiguous, durable improvement vs the untrained baseline = gradients flow
  correctly (GAE sign, advantage normalization, masking, clip all sound). Run stopped at
  plateau (~96k steps); full 400k unnecessary.
  - **On the step budget (re: the under-powering risk):** our prior runs put "clear
    improvement" at ~400k steps, so G2 was *planned* at ~400k on the easiest setup rather
    than the design's tentative 50k. In this maximal-SNR corner the signal actually
    surfaced by ~16–35k — but that does **not** generalize: it's fast only because beating
    a single random car on a fixed track is the easiest possible signal. For the
    sparse-reward-vs-heuristic and generated-track settings, the ~400k figure (and
    "1.6M was 5–10× too few" for generated) still governs how much compute later gates need.
- **What would flip it later:** if A3's dot-product head or A2's multi-seat rollouts prove
  hard to express cleanly on this loop, reconsider subclassing SB3's maskable distribution
  — but nothing in A0 suggests that.

## 7. Notes for the implementer

- **Additive only.** Do not modify `env.py`, `model.py`, `training.py`, `action_codec.py`,
  or `features.py`. Reuse them.
- Match existing conventions: module docstrings in the house style, `from __future__ import
  annotations`, type hints, `PYTHONPATH=src` import layout, dataclass configs.
- Do not commit unless asked; leave the work on the working tree and report test results.
- Keep it minimal — A0 is judged on a *correct, transparent loop*, not on features. Every
  line here is something A2–A4 build on, so clarity beats cleverness.
