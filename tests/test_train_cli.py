"""Fast tests for the committed training CLI ``scripts/train_ml.py`` (Sprint 6C).

Per the design's test gates:

* **CLI config build** — a representative arg vector parses into a valid
  ``PPOConfig`` + ``CurriculumConfig`` (no training run; we assert the built
  config and short-circuit ``train_self_play``).
* **TB log dir** — ``--tensorboard-log`` flows onto ``PPOConfig.tensorboard_log``.

``scripts/`` is not an importable package, so the module is loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_ml.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("heat_train_ml_cli", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    return _load_cli()


def test_cli_arg_vector_builds_valid_configs(cli) -> None:
    args = cli._parse_args(
        [
            "--timesteps", "1000",
            "--phase1-steps", "400",
            "--snapshot-every", "200",
            "--n-envs", "4",
            "--vec", "subproc",
            "--device", "cpu",
            "--net-profile", "large",
            "--normalize-reward",
            "--shaping-weight-start", "0.02",
            "--shaping-weight-end", "0.0",
            "--run-name", "myrun",
            "--seed", "11",
            "--gate-games", "8",
        ]
    )
    ppo, curriculum = cli.build_configs(args)

    # Net profile applied.
    assert ppo.features_dim == 256  # the "large" profile
    assert ppo.net_arch == [512, 512]

    # Throughput / device knobs.
    assert ppo.n_envs == 4
    assert ppo.device == "cpu"
    assert ppo.seed == 11

    # Curriculum / schedule knobs.
    assert curriculum.total_timesteps == 1000
    assert curriculum.phase1_steps == 400
    assert curriculum.snapshot_every == 200
    assert curriculum.run_name == "myrun"
    assert curriculum.normalize_reward is True
    assert curriculum.normalize_obs is False
    assert curriculum.shaping_weight_start == pytest.approx(0.02)
    assert curriculum.gate_games == 8


def test_cli_tensorboard_log_flows_to_config(cli, tmp_path) -> None:
    log_dir = str(tmp_path / "tb")
    args = cli._parse_args(["--tensorboard-log", log_dir])
    ppo, _curriculum = cli.build_configs(args)
    assert ppo.tensorboard_log == log_dir


def test_cli_defaults_are_sane(cli) -> None:
    args = cli._parse_args([])
    ppo, curriculum = cli.build_configs(args)
    # CPU-safe defaults: small net, single env, auto device (CPU-fallback).
    assert ppo.features_dim == 128  # small profile default
    assert ppo.n_envs == 1
    assert ppo.device == "auto"
    assert curriculum.normalize_reward is False


def test_cli_main_invokes_training_and_prints_paths(cli, tmp_path, capsys, monkeypatch) -> None:
    """``main`` builds configs, calls train_self_play (mocked), and prints paths.

    No real training — ``train_self_play`` is monkeypatched to return a sentinel
    best path so the CLI wiring (config build -> call -> output) is exercised fast.
    """
    captured = {}

    def fake_train(ppo, curriculum, *, num_players, track):  # noqa: ANN001
        captured["ppo"] = ppo
        captured["curriculum"] = curriculum
        captured["num_players"] = num_players
        return object(), str(tmp_path / "heat_ppo")

    monkeypatch.setattr(cli, "train_self_play", fake_train)

    rc = cli.main(
        [
            "--timesteps", "10",
            "--phase1-steps", "5",
            "--players", "3",
            "--checkpoint-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert captured["num_players"] == 3

    out = capsys.readouterr().out
    assert "Best checkpoint" in out
    assert "_final" in out
