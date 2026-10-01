# Repository Guidelines

## Project Structure & Module Organization

Production code lives under `src/heat/`. Core rules are in `engine/`, domain objects in `models/`, players in `agents/`, race orchestration in `simulation/`, and learning code in `ml/`. Track definitions are JSON files in `tracks/`. Keep tests in `tests/`, mirroring source areas with names such as `test_game.py` or `test_ml_model.py`. Training and evaluation entry points live in `scripts/` and `experiments/`; decisions and findings belong in `docs/`. Do not commit generated content from `runs/`, `checkpoints/`, `data/`, caches, or logs.

## Build, Test, and Development Commands

Install the package and developer tools with:

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH="src"; python -m pytest
```

The first command creates an editable development install. The second runs the full test suite. Use `python -m pytest tests/test_game.py -q` for a focused check, `python -m pytest -m "not slow"` to skip subprocess smoke tests, `ruff check src tests scripts experiments` for linting, and `mypy src` for strict type checking. Install `.[ml]` only when working on training features. Run the normal training workflow with `$env:PYTHONPATH="src"; python train_8c.py`; it is not part of routine verification.

## Learned Policy Generation Names

Before starting, registering, or comparing a non-smoke learned-policy campaign, read `docs/policy-generation-registry.html` and `experiments/policy_registry/index.json`. Allocate the next immutable `G####` ID for any recipe change that can affect learned weights; use `G####-R##` for independent seeded runs and `@<checkpoint>` only when identifying a non-selected checkpoint. Repeated seeds of the same resolved recipe stay in the same generation. Smoke runs use `dev-*` and are not registered. Record the complete resolved recipe, source revision/patch provenance, parent and anchor policy IDs, run seeds, selected checkpoint paths, and SHA-256 hashes in the registry. Evaluation artifacts and reports must use registered policy IDs rather than inventing experiment-local agent names. Never reuse or renumber an allocated generation.

# User communication
    - Keep generated documents for users feedback in as simple language as possible. The user has coding experiance but no game training experiance.Technical language or concepts should be used but they should always be described in simple terms. Descriptions of what has been done or designed should include all relvant details but should also include enough explainations that they can be followed by someone without lots of context
# Code Preferences
    - Focus on simple changes where possible
    - If a solution is more complicated then it seemed or a better approach is found consider feeding back and re designing
    - Avoid excessive tests or tests for the sake of it. New code should have some tests attached but only focused considered tests
    - Methods should be commented. Keep comments to a reasonable limit and dont over comment 
    - Report back after writing code with simple descriptions of what was changed.
    - Update design documents if used and mark them as implemented. If the document describes a large multi stage implementation mark the sections that have been completed and what is left outstanding.
# Design
    - Where appropriate ask the user for design directions. If there are multiple possible approaches quickly create high level descriptions with pros and cons then ask the user for feedback of which direction to design in detail
    - Create design HTML documents. Keep the documents simple in language technical subjects should be readable by someone with a basic understanding of the project. 
    - Design documents should focus on being human readable including relevant code diagrams and short code snippets where there is a relevant idea
# Experiments
    - Experiments will need to be run to evaluate the performance changes.
    - Before running experiments make sure they are well designed with a hypothesis and a estimation of our confidence levels in the changes
    - Make sure all experiments are reasonably scoped and its possible that a result will be seen in the scope planned.
    - Include in experiment scope both the ammount of code changes that will be needed as well as estimated compute time
    - Write up results in a HTML document. Focus on human readability.
    - Suggest experiments where they could be useful or give needed direction.
    - Do not start large experiments without permission from the user.
    - Before starting experiments make sure that reasonable time limits are set or logging produced so that we know if the experiment is progressing or has stalled
   

## Agent skills

### Issue tracker

Issues are tracked as local Markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The default canonical triage label vocabulary is used. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo using root `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.
