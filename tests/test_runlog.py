"""Tests for the run-lifecycle logging helper (experiments/_runlog.py, FIX 2).

The markers must be reliable: a successful run prints a DONE marker; a crashed
run prints the traceback, a single-line FAILED marker, and exits non-zero. These
are what a ``tail -f | grep '==='`` monitor relies on so a finished or crashed
run is never mistaken for a still-running one.
"""

from __future__ import annotations

import os
import sys

import pytest

_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import _runlog  # noqa: E402


def test_run_main_prints_start_and_done_on_success(capsys):
    ran = {"called": False}

    def fn():
        ran["called"] = True

    _runlog.run_main("demo", fn)  # must NOT raise / exit on success

    assert ran["called"]
    out = capsys.readouterr().out
    assert "=== demo START" in out
    assert "=== demo DONE (exit 0," in out
    assert "FAILED" not in out


def test_run_main_prints_failed_and_exits_nonzero_on_exception(capsys):
    def fn():
        raise ValueError("boom kaboom")

    with pytest.raises(SystemExit) as ei:
        _runlog.run_main("demo", fn)

    assert ei.value.code == 1
    cap = capsys.readouterr()
    out, err = cap.out, cap.err
    assert "=== demo START" in out
    # The greppable one-line FAILED marker (carrying the exception text) on
    # stdout, and the full traceback on stderr (so the cause is in the log).
    assert "=== demo FAILED (exit 1," in out
    assert "boom kaboom" in out
    assert "ValueError" in err
    # The single-line marker must not have been split across lines.
    failed_line = next(l for l in out.splitlines() if "FAILED (exit 1" in l)
    assert failed_line.endswith("===")


def test_run_main_treats_sys_exit_zero_as_success(capsys):
    def fn():
        sys.exit(0)

    # A clean sys.exit(0) inside the wrapped main is a success, not a failure.
    _runlog.run_main("demo", fn)

    out = capsys.readouterr().out
    assert "=== demo DONE (exit 0," in out
    assert "FAILED" not in out


def test_run_main_propagates_explicit_nonzero_sys_exit(capsys):
    def fn():
        sys.exit(2)

    with pytest.raises(SystemExit) as ei:
        _runlog.run_main("demo", fn)

    # An explicit non-zero exit is a failure: FAILED marker + non-zero exit.
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "=== demo FAILED (exit 1," in out
