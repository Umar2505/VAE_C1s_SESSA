#!/usr/bin/env python3
"""Run each generated SESSA physical case in a fresh, resumable process."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import subprocess
import threading
import time


CASE_PATTERN = re.compile(r"case_(\d+)_full")


def load_case_blocks(sessions: list[Path]) -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    for session in sessions:
        current: list[str] = []
        for raw_line in session.read_text(encoding="utf-8").splitlines():
            command = raw_line.strip()
            if not command or command.upper() == "QUIT":
                continue
            if command.upper().endswith("PROJECT RESET") and current:
                match = next((CASE_PATTERN.search(item) for item in current if CASE_PATTERN.search(item)), None)
                if match is None:
                    raise ValueError(f"Cannot determine case ID in {session}")
                blocks[int(match.group(1))] = current
                current = []
            current.append(command)
        if current:
            match = next((CASE_PATTERN.search(item) for item in current if CASE_PATTERN.search(item)), None)
            if match is None:
                raise ValueError(f"Cannot determine case ID in {session}")
            blocks[int(match.group(1))] = current
    return blocks


def run_case(
    case_id: int,
    commands: list[str],
    sessa: Path,
    env: dict[str, str],
    failure_dir: Path,
    attempts: int,
) -> None:
    last_output = ""
    for attempt in range(1, attempts + 1):
        process = subprocess.Popen(
            ["xvfb-run", "-a", str(sessa), "-c"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        transcript: list[str] = []
        try:
            for command in commands:
                process.stdin.write(command + "\n")
                process.stdin.flush()
                while True:
                    line = process.stdout.readline()
                    transcript.append(line)
                    if line == "" and process.poll() is not None:
                        raise RuntimeError(f"SESSA exited {process.returncode} during {command}")
                    if "*** Error" in line or "Unhandled unknown exception" in line:
                        raise RuntimeError(f"SESSA rejected {command}")
                    if "Done" in line:
                        break
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.stdin.close()
            transcript.append(process.stdout.read())
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(f"SESSA exited {return_code} after case completion")
            return
        except Exception as error:
            process.kill()
            process.wait()
            last_output = "".join(transcript) + f"\nAttempt {attempt}: {error}\n"
            if attempt < attempts:
                time.sleep(0.5 * attempt)
    failure_dir.mkdir(exist_ok=True)
    failure_path = failure_dir / f"case_{case_id:05d}.log"
    failure_path.write_text(last_output, encoding="utf-8")
    raise RuntimeError(f"case {case_id} failed after {attempts} attempts; see {failure_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sessa", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--pattern", default="atmospheric_c1s_batch*.ses")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run at most this many pending cases (useful for a pilot)")
    args = parser.parse_args()

    root = args.input.resolve()
    sessions = sorted(root.glob(args.pattern))
    if not sessions:
        parser.error(f"No generated session files found in {root}")
    blocks = load_case_blocks(sessions)
    spectra = root / "spectra"
    failure_dir = root / "failures"
    pending = {
        case_id: commands for case_id, commands in blocks.items()
        if not (spectra / f"case_{case_id:05d}_fullreg1.spc").exists()
        or not (spectra / f"case_{case_id:05d}_zeroreg1.spc").exists()
    }
    already_complete = len(blocks) - len(pending)
    remaining = len(pending)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        pending = dict(sorted(pending.items())[:args.limit])
    print(f"cases discovered={len(blocks):,}; already complete={already_complete:,}; "
          f"remaining={remaining:,}; selected this run={len(pending):,}")

    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(args.library_dir.resolve()) + (":" + existing if existing else "")
    started = time.monotonic()
    completed_now = 0
    completed_lock = threading.Lock()
    failures = []

    def wrapped(case_id: int, commands: list[str]) -> None:
        nonlocal completed_now
        run_case(case_id, commands, args.sessa.resolve(), env, failure_dir, args.attempts)
        with completed_lock:
            completed_now += 1
            if completed_now % 100 == 0 or completed_now == len(pending):
                elapsed = (time.monotonic() - started) / 60.0
                print(f"completed this run={completed_now:,}/{len(pending):,}; elapsed={elapsed:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(wrapped, case_id, commands): case_id
            for case_id, commands in pending.items()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append((futures[future], error))
                print(f"failed: {error}", flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} cases failed; rerun the same command to retry only missing cases")
    print(f"SESSA generation complete for this run: {len(pending):,} physical cases", flush=True)


if __name__ == "__main__":
    main()
