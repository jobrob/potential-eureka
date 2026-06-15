"""Tests for the scripts/run_simulation.py CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLI_PATH = _REPO_ROOT / "scripts" / "run_simulation.py"


def _load_cli_module():
    """Import scripts/run_simulation.py as a module by file path."""
    spec = importlib.util.spec_from_file_location("run_simulation_cli", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    return _load_cli_module()


class TestMainInvocation:
    def test_runs_and_prints_summary(self, cli, capsys) -> None:
        cli.main(
            ["--games", "5", "--track", "usa", "--seed", "1", "--no-parallel"]
        )
        out = capsys.readouterr().out
        assert "HEAT Simulation Summary" in out
        assert "Games:   5" in out

    def test_random_vs_heuristic_runs(self, cli, capsys) -> None:
        cli.main(
            [
                "--games", "4", "--heuristic", "1", "--random", "1",
                "--seed", "2", "--no-parallel",
            ]
        )
        out = capsys.readouterr().out
        assert "HeuristicAgent" in out
        assert "RandomAgent" in out

    def test_group_by_player_id(self, cli, capsys) -> None:
        cli.main(
            ["--games", "3", "--players", "2", "--group", "player_id",
             "--seed", "1", "--no-parallel"]
        )
        out = capsys.readouterr().out
        assert "player_0" in out


class TestBadInput:
    def test_too_many_players(self, cli) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--players", "7", "--no-parallel"])
        assert exc.value.code == 1

    def test_zero_players(self, cli) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--heuristic", "0", "--random", "0", "--no-parallel"])
        assert exc.value.code == 1

    def test_zero_games(self, cli) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--games", "0", "--no-parallel"])
        assert exc.value.code == 1

    def test_unknown_track(self, cli) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--games", "2", "--track", "nope", "--no-parallel"])
        assert exc.value.code == 1


@pytest.mark.slow
class TestSubprocessSmoke:
    def test_subprocess_runs(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_REPO_ROOT / "src")
        result = subprocess.run(
            [sys.executable, str(_CLI_PATH), "--games", "3", "--no-parallel"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "HEAT Simulation Summary" in result.stdout
