# Experiment harnesses

Reusable, throwaway-friendly scripts from the 2026-06 ML training investigation.
See `docs/ml-learnings-solo-pretrain.md` for the findings these produced.

Run from the repo root with `PYTHONPATH=src`, e.g.:

```bash
PYTHONPATH=src python experiments/proto_solo.py
```

| Script | What it does |
|---|---|
| `proto_solo.py` | Solo time-trial pretraining (`num_players=1`, dense progress + finish-bonus reward, speed via `gamma<1`). Reports steps-to-finish (fresh vs trained) and a cold transfer eval into a 4p race vs the weak heuristic. Saves `checkpoints/heat_ppo_solo_proto`. |
| `proto_finetune.py` | Warm-starts the solo checkpoint and fine-tunes vs weak opponents, then ramps to strong (the opponent-axis curriculum). Reports win-rate vs weak and strong after each stage. |
| `diag_ablate.py` | Toggles one Sprint-A/B feature at a time on generated tracks (seat / track-curriculum / strong opponents), trains short, evals vs the weak heuristic on held-out generated tracks. Variant name is the CLI arg. |
| `diag_canlearn.py` | Learnability control: trains on the fixed `usa` track with weak opponents and no extras. If this clears ~90% the core obs/training/eval path is healthy. |

These are diagnostics, not part of the product. Checkpoints they write land in the
git-ignored `checkpoints/` directory.
