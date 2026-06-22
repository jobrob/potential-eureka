"""Shared run-lifecycle logging for the experiment entry points.

Long-running experiment scripts (``finetune_ppo.py``, ``dagger.py``, ...) used to
produce *no* terminal signal: a finished or crashed run looked identical to one
still grinding away, so a ``tail -f | grep`` monitor (or a human) could wait on a
run that had actually exited 40 minutes ago. An uncaught exception was worse --
it could die quietly into a redirected log with no greppable marker.

:func:`run_main` wraps an entry point's ``main()`` so every run emits three
reliable, flushed, single-line markers:

    === {name} START <iso-timestamp> ===
    === {name} DONE (exit 0, {elapsed:.1f}s) ===          # on success
    === {name} FAILED (exit 1, {elapsed:.1f}s): {exc} ===  # on any exception

The markers are flushed to stdout immediately so a ``tail -f | grep '==='``
monitor reliably sees START/DONE/FAILED in real time. On failure the full
traceback is printed (so the cause is in the log) followed by the single-line
FAILED marker, then the process exits non-zero -- a crashed run can never be
mistaken for a running one.
"""

from __future__ import annotations

import datetime as _dt
import sys
import time
import traceback
from typing import Callable


def _force_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 so non-ASCII output never crashes.

    On a default Windows console ``sys.stdout`` is cp1252; printing the Greek /
    math glyphs the Option-C banners and ``argparse`` ``description=__doc__`` help
    text carry (``pi``/``tau``/``Sigma``/``-`` etc.) raises ``UnicodeEncodeError``
    and kills an otherwise-healthy run at exit 1 before it does any work. Forcing
    UTF-8 here (the Py3.7+ ``reconfigure`` hook, a no-op when already UTF-8 or when
    the stream does not support it, e.g. a plain pipe) makes every wrapped entry
    point robust on a plain console without a ``PYTHONIOENCODING`` override. ASCII
    output is unaffected; this only widens what can be encoded.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - non-reconfigurable stream
            pass


def _flush() -> None:
    """Flush both streams so a tail/grep monitor sees markers immediately."""
    try:
        sys.stdout.flush()
    except Exception:  # pragma: no cover - stdout may be closed at shutdown
        pass
    try:
        sys.stderr.flush()
    except Exception:  # pragma: no cover
        pass


def run_main(name: str, fn: Callable[[], object]) -> None:
    """Run ``fn`` with START/DONE/FAILED lifecycle markers around it.

    Prints a flushed ``START`` line, calls ``fn()``, then on success prints a
    flushed ``DONE (exit 0, ...)`` line and returns. On ANY exception it prints
    the full traceback, a single-line flushed ``FAILED (exit 1, ...)`` marker,
    and calls ``sys.exit(1)`` so the non-zero exit is unambiguous.

    Intended to wrap an entry point's ``main`` under ``if __name__ ==
    "__main__":`` -- e.g. ``run_main("train_bc", main)``.
    """
    _force_utf8_streams()
    started = _dt.datetime.now().isoformat(timespec="seconds")
    print(f"=== {name} START {started} ===")
    _flush()
    t0 = time.perf_counter()
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - we re-raise via sys.exit below
        elapsed = time.perf_counter() - t0
        # Full traceback first (so the cause is in the log), then the greppable
        # single-line marker. SystemExit(0) is treated as success below.
        if isinstance(exc, SystemExit) and (exc.code in (0, None)):
            print(f"=== {name} DONE (exit 0, {elapsed:.1f}s) ===")
            _flush()
            return
        traceback.print_exc()
        # Collapse the exception to a single line for the marker.
        summary = str(exc).replace("\n", " ").strip() or exc.__class__.__name__
        print(f"=== {name} FAILED (exit 1, {elapsed:.1f}s): {summary} ===")
        _flush()
        sys.exit(1)
    elapsed = time.perf_counter() - t0
    print(f"=== {name} DONE (exit 0, {elapsed:.1f}s) ===")
    _flush()
