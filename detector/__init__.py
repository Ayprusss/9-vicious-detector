"""
Phase 4 inference package.

Two decoupled layers:
  inference.py     — perception: one frame in, list[Detection] out (stateless)
  state_machine.py — decision: a stream of detections in, TriggerEvent out (stateful)
"""

from detector.inference import Detection, HandSignDetector
from detector.state_machine import SignStateMachine, TriggerEvent

__all__ = [
    "Detection",
    "HandSignDetector",
    "SignStateMachine",
    "TriggerEvent",
]
