# Phase 1 — Data Collection: high-impact gaps to be mindful of

These are the issues that, if missed during capture, will silently degrade
the model's real-world performance. Most are about discipline, not code.
**Re-read this before every capture session.**

---

## 1. Capture reference frames for each sign first

**Action:** Before any real collection, capture 3–5 "definitive" reference
shots per sign — your cleanest possible execution of each. Stash them
somewhere you can glance at (e.g. `data/reference/sign_ysl/`,
`data/reference/sign_nine/`).

**Why it matters:** `sign_ysl` and `sign_nine` exist only as muscle
memory in your hand. Over 2–3 sessions your form *will* drift — tight
on day 1, sloppier on day 3, sometimes unconsciously hybriding the two
signs in transitional frames. A frame that's 70% sign_ysl / 30% sign_nine
is **label noise**: you're training the model to make a distinction you
couldn't reliably make yourself, which it'll learn to make randomly.

**Heuristic during capture:** if a frame doesn't match the reference,
retake it.

---

## 2. Spread captures across 3+ sessions on different days

**Action:** Do not shoot all ~1,200 images in one sitting. Plan for at
least 3 separate sessions on different days.

**Why it matters:** Different days = different lighting, different
outfits, different camera white-balance state, different hand fatigue,
different mood. This kind of variation is genuinely hard to fake
deliberately — you get it for free if you just give it time. Compounds
with the variation matrix in `HANDOFF.md`.

---

## 3. Reserve a held-out test set under DIFFERENT conditions

**Action:** Dedicate one short session (~50–80 images total, ~15–25 per
class) to deliberately novel conditions — different room, different
lighting, possibly different shirt, possibly a different time of day.
In Phase 2, **manually assign these images to the test split** in
Roboflow rather than letting the random 70/20/10 splitter handle them.

**Why it matters:** Roboflow's default split is random across uploaded
images. If every image was captured under similar conditions, the test
set is "more of the same" — you'll see a great mAP that doesn't predict
real-world performance. The honest measure of generalization requires
the test set to look *different* from the training set, the same way
deployment will look different.

This pattern is called a **held-out distribution test set**. It's the
only way to catch overfitting to your capture environment.

---

## 4. Back up `data/raw/` after every session

**Action:** After each session, copy `data/raw/` to OneDrive / Google
Drive / an external disk.

**Why it matters:** `data/raw/` is gitignored — it lives only on this
laptop. Cost of forgetting: 6–10 hours of capture work lost to an SSD
failure, accidental delete, or Explorer misclick.

---

## 5. QC pause after the first ~30 frames per class

**Action:** After roughly 30 images per class on your first session,
**stop and skim them** with `scripts/browse.py` before going further.

**Why it matters:** Catch systematic issues early — cheaper to fix a
habit at 30 frames than at 300. Look for:
- Hand consistently cut off at an edge of frame
- Focus locked on background instead of hand
- Autoexposure crushing details on dark backgrounds
- Sign form drifting away from the reference frames
- Lighting hotspots / glare on the hand

---

## 6. Hand-size-in-frame floor

**Rule of thumb:** the hand should occupy at least ~5% of the image
area — roughly a 200×200 region at our 1280×720 capture resolution.

**Why it matters:** YOLO downsamples the image significantly through
its conv stack. A hand at 50×50 pixels in the source image has
essentially no signal left after downsampling. Frames where the hand
is way off in the distance are mostly throwaway.

Doesn't need to be enforced by the script — just stay conscious of it.

---

## 7. Commit to a concrete plan for friend's-hand frames

**Action:** Target ~40 of the 400 per positive class (sign_ysl,
sign_nine) from a friend's hand. `other_hand` is more forgiving and
doesn't need this.

**Why it matters:** Without a concrete commitment, this gets
deprioritized and never happens. The model will then overfit to your
specific hand geometry, skin tone, and proportions — and silently fail
on anyone else.

---

## Pre-capture checklist (re-read before each session)

- [ ] Reviewed reference frames so I know exactly what I'm targeting
- [ ] Today's session is materially different from prior sessions
      (lighting, location, outfit, time of day, etc.)
- [ ] If this is the "held-out test set" session, mentally tag those frames
- [ ] Will back up `data/raw/` to external storage after this session
- [ ] Will pause at ~30 frames per class and skim with `scripts/browse.py`
- [ ] Hand fills at least ~5% of frame in shots I'm keeping

---

## Reminders on dataset composition (from the broader conversation)

**`other_hand` mix target:**
| Category | Target % | Why |
|---|---|---|
| Similar/competing hand signs (peace, OK, other rap signs) | ~50% | Hard negatives — highest-value frames |
| Transitional / mid-motion poses | ~20% | Defeats trigger flicker during sign formation |
| Random gestures (pointing, waving) | ~15% | General coverage |
| Relaxed / mundane (typing, resting) | ~15% | Cheap base rate |

**Framing distribution should be identical across classes.** Don't shoot
sign_ysl always face+shoulders and other_hand always hand-only — the
model will learn "face visible == sign_ysl" as a shortcut. Pick ~3
framing styles (close-up hand-only / mid-shot / wide off-center) and
apply each style in similar proportions to *all three* classes.

**Pose variation within a class is the goal, not a problem.** Different
angles/tilts/orientations teach the model pose invariance. Filter rule:
if a stranger looking at the photo couldn't tell which sign it is, the
frame is label noise — skip it.

**Don't waste captures on horizontal mirrors.** Roboflow's Phase 2
augmentation flips every training image horizontally. If you have your
right hand tilted left, the flip produces the left-hand-tilted-right
version for free. Out-of-plane rotations (palm vs. back of hand) you
*do* have to capture explicitly — augmentation can't synthesize those.
