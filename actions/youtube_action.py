"""
Phase 5 — YouTube action.

Opens a music video in the default browser. We use the stdlib `webbrowser`
module on purpose: launching a URL needs no third-party dependency. `open()`
hands the URL to the OS default browser; navigating straight to a YouTube watch
URL autoplays the video on desktop.
"""

from __future__ import annotations

import webbrowser


def open_video(url: str) -> bool:
    """Open `url` in the default browser (new tab). Returns True on handoff."""
    return webbrowser.open(url, new=2)  # new=2 = new tab where supported
