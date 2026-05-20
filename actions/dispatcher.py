"""
Phase 5 — Action dispatcher.

Turns a triggered sign class name into its configured action. Today every sign
opens a YouTube video, but routing through a config file (configs/actions.yaml)
means changing which video — or adding a different action type later — is a
config edit, not a code change. That's the "store behavior in config, not in
code" convention this project follows.
"""

from __future__ import annotations

from pathlib import Path

import yaml  # ships with ultralytics; already in .venv-rocm

from actions.youtube_action import open_video

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "actions.yaml"

# Marker left in placeholder URLs so we skip (and warn) instead of opening junk.
_PLACEHOLDER_MARK = "REPLACE_ME"


class ActionDispatcher:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Actions config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config: dict = yaml.safe_load(fh) or {}

    def fire(self, class_name: str) -> None:
        """Run the action mapped to `class_name`. No-ops (with a log line) if the
        class is unmapped or its URL is still a placeholder — so the detector
        keeps running even before you've filled in real URLs."""
        entry = self.config.get(class_name)
        if not entry:
            print(f"[action] no action configured for '{class_name}' — skipping")
            return
        url = (entry.get("url") or "").strip()
        if not url or _PLACEHOLDER_MARK in url:
            print(
                f"[action] '{class_name}' URL not set in configs/actions.yaml "
                "— skipping (fill in a real YouTube URL)"
            )
            return
        print(f"[action] {class_name} -> opening YouTube: {url}")
        open_video(url)
