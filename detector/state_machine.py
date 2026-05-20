"""
Phase 4 — Decision layer.

The detector tells us what's in each frame; this class decides whether the user
*meant* a sign. It exists because models jitter: trusting any single frame would
make the system fire on noise. Three mechanisms, all classic systems ideas:

  * Temporal smoothing — keep the last `window` frames' votes and require a class
    to win at least `min_votes` of them. One bad frame can't trigger anything.
  * Debouncing — same principle that stops a physical button press registering
    twice. A momentary flicker to `sign_ysl` won't reach the vote threshold.
  * Cooldown / hysteresis — after firing, ignore everything for `cooldown_s`
    seconds so HOLDING a sign doesn't re-fire it every frame.

Per-frame, we reduce all detections to a single vote: the highest-confidence
detection at or above `conf_threshold`, or None if nothing clears the bar.
`other_hand` votes are counted (they crowd out real signs — exactly what the
negative class is for) but are never allowed to fire an action.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

from detector.inference import Detection

# Defaults: 15-frame window (~0.5s @ 30 FPS), 12/15 consensus, 0.7 confidence
# floor for a vote to count, 5s cooldown after a trigger.
DEFAULT_WINDOW = 15
DEFAULT_MIN_VOTES = 12
DEFAULT_CONF_THRESHOLD = 0.7
DEFAULT_COOLDOWN_S = 5.0
DEFAULT_TRIGGER_CLASSES = ("sign_nine", "sign_ysl")


@dataclass(frozen=True)
class TriggerEvent:
    """Emitted exactly once when a sign clears the vote + cooldown gates."""

    class_name: str
    timestamp: float  # time.monotonic() seconds


class SignStateMachine:
    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        min_votes: int = DEFAULT_MIN_VOTES,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        trigger_classes: tuple[str, ...] = DEFAULT_TRIGGER_CLASSES,
    ) -> None:
        self.window = window
        self.min_votes = min_votes
        self.conf_threshold = conf_threshold
        self.cooldown_s = cooldown_s
        self.trigger_classes = set(trigger_classes)
        self._history: deque[str | None] = deque(maxlen=window)
        self._cooldown_until = 0.0

    def update(self, detections: list[Detection]) -> TriggerEvent | None:
        """Feed one frame's detections. Returns a TriggerEvent at most once
        per gesture (then enters cooldown), else None."""
        self._history.append(self._frame_vote(detections))

        now = time.monotonic()
        if now < self._cooldown_until:
            return None  # in cooldown — keep recording votes, but never fire
        if len(self._history) < self.window:
            return None  # not enough history yet to make an honest decision

        winner, votes = self._leading_vote()
        if (
            winner is not None
            and votes >= self.min_votes
            and winner in self.trigger_classes
        ):
            self._cooldown_until = now + self.cooldown_s
            self._history.clear()  # reset so we don't immediately re-trigger
            return TriggerEvent(class_name=winner, timestamp=now)
        return None

    def _frame_vote(self, detections: list[Detection]) -> str | None:
        """Collapse a frame's detections to one vote: the highest-confidence
        detection at/above the threshold, or None."""
        best: Detection | None = None
        for det in detections:
            if det.confidence >= self.conf_threshold:
                if best is None or det.confidence > best.confidence:
                    best = det
        return best.class_name if best else None

    def _leading_vote(self) -> tuple[str | None, int]:
        """The class with the most votes in the current window (None excluded)."""
        counts = Counter(v for v in self._history if v is not None)
        if not counts:
            return None, 0
        cls, n = counts.most_common(1)[0]
        return cls, n

    # --- read-only helpers for the HUD ---

    def progress(self) -> tuple[str | None, int]:
        """(leading class, its vote count) — for the 'Tracking X: n/15' display."""
        return self._leading_vote()

    def in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())
