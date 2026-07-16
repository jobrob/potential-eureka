# Option C in plain terms — what we're building and what we think will happen

> A non-technical companion to the design docs in this folder. The `README.md` and
> the `sprint-C*` files are the precise spec; this file is the "explain it to a
> human" version, plus an honest, up-front guess at how it ends. Written
> 2026-06-22, before any C code exists.

---

## The one-paragraph version

We're teaching a neural network to drive a solo (time-trial) race well by letting it
**think ahead** and then **learn from its own thinking**. The network makes a quick
suggestion about what to do; a search routine then plays out many possible futures
using the real game engine to check whether the suggestion was actually good; the
network is retrained to agree with whatever the search discovered was best; a smarter
network makes better suggestions, so the next search is better, and we repeat. The
whole point is that this can become **better than the search tool we already have**
(`LookaheadAgent`) — unlike everything we've tried so far, which could at best copy
or match it.

## Why we're doing this (the short history)

We want an agent that drives a solo race *fast and safely*. We've already tried:

- **Plain machine learning** (copy a good player's moves): hits a ceiling — it can
  imitate but never beat the teacher.
- **Option A** (learn a "how's the race going" score and search with it): built and
  run; it improved corner safety but **plateaued** and couldn't surpass the search.
- **Option B** (a shortcut abstraction of the game): tested and **rejected** — too
  lossy to be trustworthy.
- The current best driver is **`LookaheadAgent`**, a search tool that already beats
  the strong hand-written player.

Option C is the **only remaining path that can actually surpass `LookaheadAgent`**,
because the network learns from the search's *improvements over itself*, not from a
fixed teacher. There's no ceiling baked in.

## What actually gets built (in plain terms)

Think of it as two cooperating parts that keep handing work back and forth:

1. **The thinker (the search).** Given the current race situation, it explores a tree
   of "what if I do this, then that…" futures. Two things make our version honest:
   - It uses the **real game engine** to simulate each future — not a guess at the
     rules, the actual rules. We own a perfect simulator, so we use it.
   - It correctly handles the **one source of luck** in a solo race: the cards we
     draw at the end of each round. Instead of pretending we'll know our future draws
     (the shortcut the current tool takes, which quietly biases its plans), it
     averages over real sampled draws inside the tree. This is the single most
     important way C is "the real thing" and not a half-measure.

2. **The guesser (the network).** A neural net that, for any situation, instantly
   offers (a) which moves look promising and (b) how well the race is going. The
   thinker uses these guesses to search *smarter* (focus on promising moves, stop
   early with a value estimate) instead of brute-forcing everything.

The loop: the thinker produces better answers than the guesser's raw instinct → we
**train the guesser to match the thinker's answers** → the now-smarter guesser makes
the thinker even better → repeat.

We reuse what already works: the existing network design, the existing way the game
is encoded into numbers, the existing safety rule that a line which spins the car out
is never preferred, and the existing held-out test tracks. We are **not** reinventing
those.

## The plan, sprint by sprint (each one is a checkpoint we must pass)

| Sprint | In one sentence | The bar it must clear |
|---|---|---|
| **C0** | Measure whether this is even affordable — how slow is one "think," and what dial settings make it cheap enough to run thousands of times. | A clear go / no-go with real numbers. |
| **C1** | Build the thinker (search) with a *fixed*, untrained-for-this guesser. No learning yet. | **Match** today's `LookaheadAgent` on safety and speed. |
| **C2** | Turn on learning for **one** round: let the thinker generate lessons, train the guesser once. | The retrained guesser **beats** the version it learned from. |
| **C3** | Repeat the learn-and-improve loop many times, with a safety catch that refuses to promote a worse network. | **Beat** `LookaheadAgent` outright — or stop honestly at a plateau. |

Each checkpoint is judged on the same **held-out generated tracks** the network never
trained on, and on the metric that matters: **worst-case corner spin-outs** and
**rounds taken to finish** — never just "did it finish," because a car that crawls and
spins still technically finishes.

## Our honest best guess at the outcome

This is a prediction, not a promise. We'll update it as the real numbers come in.

- **C0 (affordability): likely GO, with a real cost.** The engine is cheap to clone,
  but a single "think" now runs many simulations, each doing engine work — so each
  move will cost meaningfully more than the current tool's ~4 ms. Expect a verdict of
  "affordable for generating training data over hours, not days," possibly with a note
  that we'll eventually want GPU batching or the deferred "Gumbel" speed-up. Small
  chance C0 says the simulation budget we can afford is too low to ever beat the
  current tool — in which case we stop here and fall back to the A/B/E options.

- **C1 (match the current tool): likely YES, but the most fragile step.** A correct
  search with a sensible fixed value *should* match `LookaheadAgent` — and our version
  is theoretically *less biased* about future card luck, which could even edge it out.
  The risk is entirely in correctness: getting the value-scaling, the chance-node
  averaging, and the safety floor exactly right. History says "healthy-looking but
  subtly broken" is the failure mode to fear, which is why C1's only job is to prove
  parity before any learning muddies the picture.

- **C2 (one round of learning beats itself): probably YES.** This is the
  make-or-break experiment. If the search genuinely finds better-than-instinct moves
  and the training faithfully captures them, one generation should show a real,
  measurable improvement. If it *doesn't*, the whole premise is wrong and we'd stop —
  so this is the cheapest place to find out, by design.

- **C3 (beat `LookaheadAgent` outright): genuinely uncertain — call it a coin flip,
  leaning hopeful.** This is the prize and the real unknown. Three honest outcomes:
  1. **It works** — the loop improves generation over generation and surpasses the
     current tool on safety *and* speed on unseen tracks. Best case.
  2. **It plateaus at parity** — matches but never clearly beats `LookaheadAgent`.
     This is a legitimate, honest result: we report the gap and fall back to the spine
     options. Given Option A already plateaued, this is a real possibility.
  3. **It collapses** — training quietly makes the network worse (we've seen this
     twice before: "8C" and "S3"). We *expect* this to happen at least once; the
     whole reason C3 has a promotion guard is to catch it and refuse to ship the
     regression, not to pretend it can't occur.

- **Wildcard upside:** if C3 produces a strong network, it may drive well **on its own
  without any search at all** — which would be both the fastest and the strongest
  agent we've built. That's a deferred, post-C3 question, but it's the quiet long-term
  reason this option is worth the effort.

**Bottom line:** we're fairly confident we can *match* the current best tool (C1) and
show the learning idea works for one step (C2). Whether the full loop *surpasses* it
(C3) is the open question — and we've deliberately built the project so that "it
didn't beat the tool" is an honest, early, cheap answer rather than an expensive
surprise.

## What we are *not* building yet (so expectations stay grounded)

Opponents (this is solo-only on purpose), a learned shortcut model of the game,
the fancy fast-search variant, big networks, and a long multi-day training campaign
are all **deliberately deferred** behind clean interfaces. We're proving the core idea
is sound at small scale first. Scaling up only happens if C3 shows the loop genuinely
improves.
