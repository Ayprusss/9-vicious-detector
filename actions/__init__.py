"""
Phase 5 action layer.

Maps a triggered sign to a real-world side effect. Currently every sign opens a
YouTube music video in the browser (config-driven via configs/actions.yaml).
"""

from actions.dispatcher import ActionDispatcher
from actions.youtube_action import open_video

__all__ = ["ActionDispatcher", "open_video"]
