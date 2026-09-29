"""
Run provenance: what code, what config, which days, when.

Project rule: every run records its git commit, a dirty-tree flag, the full
config, the seeds and the date. This module is the single implementation, so a
bug fixed here is fixed for every script - the strip()/rstrip() bug below would
otherwise have needed fixing in three separate copies.
"""

import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git_information():
    """Return (commit hash, `git status --porcelain` output); None on failure."""

    def git(*arguments):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        # rstrip("\n"), not strip(): git status --porcelain encodes state in two
        # leading characters per line, so " M file" (modified, not staged) and
        # "M  file" (staged) are different states. strip() would eat the leading
        # space of the FIRST line only, silently corrupting that one entry.
        return completed.stdout.rstrip("\n") if completed.returncode == 0 else None

    return git("rev-parse", "HEAD"), git("status", "--porcelain")


def write_config_used(path, script, payload):
    """
    Write one config_used.yaml next to a run's results.

    Call this BEFORE the run starts, not after. The project rule asks for the
    dirty-tree state "when a run starts"; writing at the end also pollutes
    git_uncommitted_paths with the very result files the run is creating, which
    is what happened to results/oversell_fix_summary_config_used.yaml.

    Arguments:
        path:    where to write the YAML.
        script:  repo-relative path of the script being run, for the record.
        payload: everything script-specific (seeds, configs, criteria, ...).
                 Merged in after the shared "run:" block.
    """
    commit, porcelain = git_information()
    document = {
        "run": {
            "script": script,
            "date": date.today().isoformat(),
            "git_commit": commit,
            "git_working_tree_clean": (
                (porcelain == "") if porcelain is not None else None
            ),
            "git_uncommitted_paths": porcelain.splitlines() if porcelain else [],
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
    }
    document.update(payload)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
    return path
