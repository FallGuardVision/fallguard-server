"""
Posture verification layer for the fall detection backend.

Purpose
-------
The Bi-LSTM's FALL/IMMOBILE labels come from a heuristic (build_urfd_labels)
that only looks at hip-Y velocity. Bending/crouching produces the same
velocity signature as a real fall, causing false FALL alerts that either
revert to STABLE or stick as IMMOBILE.

This module sits BETWEEN run_inference() and log_event()/websocket send.
It does not retrain or touch the model. It:

  1. Computes independent posture features from the raw pose landmarks
     (torso verticality, body aspect ratio, hip height relative to ankles).
  2. Rejects a FALL/IMMOBILE prediction outright if posture reads upright --
     this is what kills the bending false positive.
  3. Confirms a genuine fall two ways:
       a) the model holds "FALL" steady (with fallen posture) for
          CONFIRMATION_SECONDS, or
       b) the model transitions to "IMMOBILE" while posture is fallen --
          treated as corroboration, not cancellation, since real falls are
          fast and the "FALL" class is often too brief (or entirely
          skipped) to survive a multi-second hold requirement. A short
          IMMOBILE_CONFIRM_SECONDS debounce filters a single noisy frame
          without waiting out the whole fall.
  4. Once confirmed, the rest of that fall episode (e.g. ongoing IMMOBILE)
     passes through and logs normally. The episode resets the moment the
     model returns to STABLE/HIGH_RISK, so the next fall can be detected
     fresh.

Everything else (patient scoping, WebSocket, event history) is untouched --
this module only decides whether/when a FALL should be treated as
confirmed, and exposes a monkeypatch-free set of calls you drop into
your existing inference path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (33 landmarks, x/y/z/visibility each)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# ---------------------------------------------------------------------------
# Tunable thresholds -- reverted to the original, stricter values.
#
# Validated against a real fall clip (fast, realistic motion, not a staged
# Le2i-style fall): hip-ankle gap stayed >=0.19 throughout normal standing/
# bending/walking and only collapsed to <0.13 once the person was actually
# on the ground; torso angle peaked at ~36 deg during a mid-clip lean and
# only exceeded 45 deg once genuinely fallen. The looser values tried
# previously weren't actually the cause of the missed-FALL issue (see
# process() below) and give up real margin against bending false positives,
# so they're reverted here.
# ---------------------------------------------------------------------------
TORSO_ANGLE_FALL_DEG = 55.0
HEIGHT_RATIO_FALL = 1.0
HIP_ANKLE_GAP_FALL = 0.15
CONFIRMATION_SECONDS = 1.75      # FALL-hold path: for models that DO sustain the FALL label
IMMOBILE_CONFIRM_SECONDS = 0.5   # IMMOBILE-corroboration path: debounce only, not a hold-out
MIN_VOTES_TO_FLAG_FALLEN = 2     # 2 of 3 posture signals must agree


def _landmark(flat_frame: List[float], idx: int) -> tuple:
    """flat_frame is a length-132 list/array: 33 landmarks * (x,y,z,visibility)."""
    base = idx * 4
    return flat_frame[base], flat_frame[base + 1], flat_frame[base + 2], flat_frame[base + 3]


def compute_posture_features(flat_frame: List[float]) -> Dict[str, float]:
    """
    Compute the three posture signals from a single frame's flat (132,) landmarks.
    Uses normalized MediaPipe coordinates (y increases downward).
    """
    lsx, lsy, _, _ = _landmark(flat_frame, LEFT_SHOULDER)
    rsx, rsy, _, _ = _landmark(flat_frame, RIGHT_SHOULDER)
    lhx, lhy, _, _ = _landmark(flat_frame, LEFT_HIP)
    rhx, rhy, _, _ = _landmark(flat_frame, RIGHT_HIP)
    lax_, lay, _, _ = _landmark(flat_frame, LEFT_ANKLE)
    rax_, ray, _, _ = _landmark(flat_frame, RIGHT_ANKLE)
    nx, ny, _, _ = _landmark(flat_frame, NOSE)

    shoulder_mid = ((lsx + rsx) / 2.0, (lsy + rsy) / 2.0)
    hip_mid = ((lhx + rhx) / 2.0, (lhy + rhy) / 2.0)
    ankle_mid = ((lax_ + rax_) / 2.0, (lay + ray) / 2.0)

    # 1. Torso verticality: angle between shoulder->hip vector and vertical axis
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    torso_angle_deg = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    # 2. Body aspect ratio: vertical span (head to ankle) vs horizontal span
    xs = [nx, lsx, rsx, lhx, rhx, lax_, rax_]
    ys = [ny, lsy, rsy, lhy, rhy, lay, ray]
    vertical_span = max(ys) - min(ys)
    horizontal_span = max(xs) - min(xs) + 1e-6
    height_ratio = vertical_span / horizontal_span

    # 3. Hip height relative to ankles (small gap = hips near floor level)
    hip_ankle_gap = abs(hip_mid[1] - ankle_mid[1])

    return {
        "torso_angle_deg": torso_angle_deg,
        "height_ratio": height_ratio,
        "hip_ankle_gap": hip_ankle_gap,
    }


def is_fallen_posture(features: Dict[str, float]) -> Tuple[bool, Dict[str, object]]:
    """
    Majority vote across the three independent posture signals.
    Returns: (is_fallen, debug_info)
    """
    torso_ok = features["torso_angle_deg"] >= TORSO_ANGLE_FALL_DEG
    height_ok = features["height_ratio"] <= HEIGHT_RATIO_FALL
    hip_ok = features["hip_ankle_gap"] <= HIP_ANKLE_GAP_FALL

    votes = int(torso_ok) + int(height_ok) + int(hip_ok)

    debug = {
        "torso_angle": round(features["torso_angle_deg"], 2),
        "torso_pass": torso_ok,
        "height_ratio": round(features["height_ratio"], 3),
        "height_pass": height_ok,
        "hip_ankle_gap": round(features["hip_ankle_gap"], 4),
        "hip_pass": hip_ok,
        "votes": votes,
        "min_votes_needed": MIN_VOTES_TO_FLAG_FALLEN,
        "is_fallen": votes >= MIN_VOTES_TO_FLAG_FALLEN,
    }
    return votes >= MIN_VOTES_TO_FLAG_FALLEN, debug


@dataclass
class _PendingState:
    pending_since: Optional[float] = None            # FALL-hold timer
    immobile_pending_since: Optional[float] = None    # IMMOBILE-corroboration debounce timer
    last_confirmed_label: str = "STABLE"
    fall_confirmed_this_episode: bool = False         # avoids re-confirming every tick while down


@dataclass
class PostureVerifier:
    """
    Stateful, per-patient verifier. Instantiate once (module-level singleton
    in your backend) and call process() for every inference result.
    """
    _states: Dict[str, _PendingState] = field(default_factory=dict)
    _debug_enabled: bool = True  # Set to False to disable debug prints

    def _state_for(self, patient_id: str) -> _PendingState:
        if patient_id not in self._states:
            self._states[patient_id] = _PendingState()
        return self._states[patient_id]

    def process(
        self,
        patient_id: str,
        raw_label: str,
        flat_frame: List[float],
        now: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Call this once per inference tick, right after run_inference().

        Returns a dict:
          {
            "label": <str>           -- the label to actually log/broadcast
            "should_log_event": bool  -- whether to write to the events table
            "is_pending": bool        -- true while a fall is being verified
            "posture_debug": dict     -- debug info for tuning (empty when not computed)
          }
        """
        now = now if now is not None else time.time()
        state = self._state_for(patient_id)

        if raw_label in ("STABLE", "HIGH_RISK"):
            # Recovery / normal activity -- reset the episode so the next
            # fall (if any) is detected fresh.
            state.pending_since = None
            state.immobile_pending_since = None
            state.fall_confirmed_this_episode = False
            state.last_confirmed_label = raw_label
            return {"label": raw_label, "should_log_event": True, "is_pending": False, "posture_debug": {}}

        # raw_label is FALL or IMMOBILE from here on -- posture always decides
        features = compute_posture_features(flat_frame)
        fallen, debug_info = is_fallen_posture(features)

        if self._debug_enabled and patient_id is not None:
            status = "FALLEN [pass]" if fallen else "UPRIGHT [reject]"
            print(f"[PostureVerifier] Patient {patient_id}: raw={raw_label} {status} | "
                  f"torso={debug_info['torso_angle']} deg (need >={TORSO_ANGLE_FALL_DEG}), "
                  f"height={debug_info['height_ratio']} (need <={HEIGHT_RATIO_FALL}), "
                  f"hip_gap={debug_info['hip_ankle_gap']} (need <={HIP_ANKLE_GAP_FALL}), "
                  f"votes={debug_info['votes']}/{debug_info['min_votes_needed']}")

        if not fallen:
            # Model says FALL/IMMOBILE but posture says upright -- reject outright.
            # Covers bending (misread as FALL) and ordinary quiet sitting/standing
            # (misread as IMMOBILE).
            state.pending_since = None
            state.immobile_pending_since = None
            return {
                "label": state.last_confirmed_label,
                "should_log_event": False,
                "is_pending": False,
                "posture_debug": debug_info,
            }

        # Posture agrees this looks fallen.
        if state.fall_confirmed_this_episode:
            # Already confirmed earlier in this same fall episode -- just pass
            # the model's current label through (typically IMMOBILE) so the UI
            # reflects ongoing post-fall stillness, and keep logging it.
            state.last_confirmed_label = raw_label
            state.pending_since = None
            state.immobile_pending_since = None
            return {"label": raw_label, "should_log_event": True, "is_pending": False, "posture_debug": debug_info}

        if raw_label == "IMMOBILE":
            # The model has already concluded "not moving" while posture reads
            # as fallen -- strong corroboration of a real fall, even if the
            # transient FALL class was too brief (or skipped entirely) to
            # satisfy the hold-timer below. Debounce briefly to filter a
            # single noisy frame, not to wait out the fall.
            if state.immobile_pending_since is None:
                state.immobile_pending_since = now
                if self._debug_enabled:
                    print(f"[PostureVerifier] Patient {patient_id}: IMMOBILE+fallen, debounce started")
                return {
                    "label": state.last_confirmed_label,
                    "should_log_event": False,
                    "is_pending": True,
                    "posture_debug": debug_info,
                }

            if now - state.immobile_pending_since >= IMMOBILE_CONFIRM_SECONDS:
                state.fall_confirmed_this_episode = True
                state.last_confirmed_label = "FALL"
                state.pending_since = None
                state.immobile_pending_since = None
                if self._debug_enabled:
                    print(f"[PostureVerifier] Patient {patient_id}: FALL CONFIRMED via IMMOBILE corroboration")
                return {"label": "FALL", "should_log_event": True, "is_pending": False, "posture_debug": debug_info}

            return {
                "label": state.last_confirmed_label,
                "should_log_event": False,
                "is_pending": True,
                "posture_debug": debug_info,
            }

        # raw_label == "FALL": original sustained-hold path, for models that
        # do hold the FALL label steady for a while.
        if state.pending_since is None:
            state.pending_since = now
            if self._debug_enabled:
                print(f"[PostureVerifier] Patient {patient_id}: FALL+fallen, hold-timer started")
            return {
                "label": state.last_confirmed_label,
                "should_log_event": False,
                "is_pending": True,
                "posture_debug": debug_info,
            }

        elapsed = now - state.pending_since
        if elapsed >= CONFIRMATION_SECONDS:
            state.fall_confirmed_this_episode = True
            state.last_confirmed_label = "FALL"
            state.pending_since = None
            if self._debug_enabled:
                print(f"[PostureVerifier] Patient {patient_id}: FALL CONFIRMED after {elapsed:.2f}s hold")
            return {"label": "FALL", "should_log_event": True, "is_pending": False, "posture_debug": debug_info}

        return {
            "label": state.last_confirmed_label,
            "should_log_event": False,
            "is_pending": True,
            "posture_debug": debug_info,
        }

    def reset(self, patient_id: str) -> None:
        """Call when a monitoring session ends/switches patient."""
        self._states.pop(patient_id, None)
        if self._debug_enabled:
            print(f"[PostureVerifier] Patient {patient_id}: session reset")



# """
# Posture verification layer for the fall detection backend.

# Purpose
# -------
# The Bi-LSTM's FALL/IMMOBILE labels come from a heuristic (build_urfd_labels)
# that only looks at hip-Y velocity. Bending/crouching produces the same
# velocity signature as a real fall, causing false FALL alerts that either
# revert to STABLE or stick as IMMOBILE.

# This module sits BETWEEN run_inference() and log_event()/websocket send.
# It does not retrain or touch the model. It:

#   1. Computes independent posture features from the raw pose landmarks
#      (torso verticality, body aspect ratio, hip height relative to ankles).
#   2. Requires temporal confirmation: a FALL prediction must be accompanied
#      by a "fallen posture" for ~1.0-1.5s before it is confirmed and alerted.
#   3. If posture returns upright before the timer elapses, the pending
#      fall is cancelled silently -- nothing is logged, no alert fires.
#   4. Real falls (sustained low, horizontal posture) still confirm within
#      the window and pass through unchanged.

# Everything else (patient scoping, WebSocket, event history) is untouched --
# this module only decides whether/when a FALL should be treated as
# confirmed, and exposes a monkeypatch-free set of calls you drop into
# your existing inference path.
# """

# from __future__ import annotations

# import math
# import time
# from dataclasses import dataclass, field
# from typing import Optional, List, Dict


# # ---------------------------------------------------------------------------
# # MediaPipe Pose landmark indices (33 landmarks, x/y/z/visibility each)
# # ---------------------------------------------------------------------------
# NOSE = 0
# LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
# LEFT_HIP, RIGHT_HIP = 23, 24
# LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# # ---------------------------------------------------------------------------
# # Tunable thresholds -- adjusted for real-world testing
# # Reduced thresholds to be more tolerant of different fall postures
# # ---------------------------------------------------------------------------
# TORSO_ANGLE_FALL_DEG = 38.0      # Reduced from 55° (more tolerant of sideways/backward falls)
# HEIGHT_RATIO_FALL = 1.8          # Increased from 1.0 (camera angle can affect this)
# HIP_ANKLE_GAP_FALL = 0.30        # Increased from 0.15 (more tolerant of camera angles)
# CONFIRMATION_SECONDS = 1.25      # Reduced from 1.75 (faster alert for testing)
# MIN_VOTES_TO_FLAG_FALLEN = 2     # 2 out of 3 signals must agree


# def _landmark(flat_frame: List[float], idx: int) -> tuple:
#     """flat_frame is a length-132 list/array: 33 landmarks * (x,y,z,visibility)."""
#     base = idx * 4
#     return flat_frame[base], flat_frame[base + 1], flat_frame[base + 2], flat_frame[base + 3]


# def compute_posture_features(flat_frame: List[float]) -> Dict[str, float]:
#     """
#     Compute the three posture signals from a single frame's flat (132,) landmarks.
#     Uses normalized MediaPipe coordinates (y increases downward).
#     """
#     lsx, lsy, _, _ = _landmark(flat_frame, LEFT_SHOULDER)
#     rsx, rsy, _, _ = _landmark(flat_frame, RIGHT_SHOULDER)
#     lhx, lhy, _, _ = _landmark(flat_frame, LEFT_HIP)
#     rhx, rhy, _, _ = _landmark(flat_frame, RIGHT_HIP)
#     lax_, lay, _, _ = _landmark(flat_frame, LEFT_ANKLE)
#     rax_, ray, _, _ = _landmark(flat_frame, RIGHT_ANKLE)
#     nx, ny, _, _ = _landmark(flat_frame, NOSE)

#     shoulder_mid = ((lsx + rsx) / 2.0, (lsy + rsy) / 2.0)
#     hip_mid = ((lhx + rhx) / 2.0, (lhy + rhy) / 2.0)
#     ankle_mid = ((lax_ + rax_) / 2.0, (lay + ray) / 2.0)

#     # 1. Torso verticality: angle between shoulder->hip vector and vertical axis
#     dx = hip_mid[0] - shoulder_mid[0]
#     dy = hip_mid[1] - shoulder_mid[1]
#     torso_angle_deg = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

#     # 2. Body aspect ratio: vertical span (head to ankle) vs horizontal span
#     xs = [nx, lsx, rsx, lhx, rhx, lax_, rax_]
#     ys = [ny, lsy, rsy, lhy, rhy, lay, ray]
#     vertical_span = max(ys) - min(ys)
#     horizontal_span = max(xs) - min(xs) + 1e-6
#     height_ratio = vertical_span / horizontal_span

#     # 3. Hip height relative to ankles (small gap = hips near floor level)
#     hip_ankle_gap = abs(hip_mid[1] - ankle_mid[1])

#     return {
#         "torso_angle_deg": torso_angle_deg,
#         "height_ratio": height_ratio,
#         "hip_ankle_gap": hip_ankle_gap,
#     }


# def is_fallen_posture(features: Dict[str, float]) -> tuple[bool, Dict[str, object]]:
#     """
#     Majority vote across the three independent posture signals.
#     Returns: (is_fallen, debug_info)
#     """
#     votes = 0
#     debug = {}
    
#     torso_ok = features["torso_angle_deg"] >= TORSO_ANGLE_FALL_DEG
#     height_ok = features["height_ratio"] <= HEIGHT_RATIO_FALL
#     hip_ok = features["hip_ankle_gap"] <= HIP_ANKLE_GAP_FALL
    
#     debug["torso_angle"] = round(features["torso_angle_deg"], 2)
#     debug["torso_pass"] = torso_ok
#     debug["height_ratio"] = round(features["height_ratio"], 3)
#     debug["height_pass"] = height_ok
#     debug["hip_ankle_gap"] = round(features["hip_ankle_gap"], 4)
#     debug["hip_pass"] = hip_ok
    
#     if torso_ok:
#         votes += 1
#     if height_ok:
#         votes += 1
#     if hip_ok:
#         votes += 1
    
#     debug["votes"] = votes
#     debug["min_votes_needed"] = MIN_VOTES_TO_FLAG_FALLEN
#     debug["is_fallen"] = votes >= MIN_VOTES_TO_FLAG_FALLEN
    
#     return votes >= MIN_VOTES_TO_FLAG_FALLEN, debug


# @dataclass
# class _PendingState:
#     pending_since: Optional[float] = None   # wall-clock time.time() when FALL prediction started
#     last_confirmed_label: str = "STABLE"
#     pending_start_frame: Optional[int] = None  # For tracking frame count


# @dataclass
# class PostureVerifier:
#     """
#     Stateful, per-patient verifier. Instantiate once (module-level singleton
#     in your backend) and call process() for every inference result.
#     """
#     _states: Dict[str, _PendingState] = field(default_factory=dict)
#     _debug_enabled: bool = True  # Set to False to disable debug prints

#     def _state_for(self, patient_id: str) -> _PendingState:
#         if patient_id not in self._states:
#             self._states[patient_id] = _PendingState()
#         return self._states[patient_id]

#     def process(
#         self,
#         patient_id: str,
#         raw_label: str,
#         flat_frame: List[float],
#         now: Optional[float] = None,
#         frame_idx: Optional[int] = None,
#     ) -> Dict[str, object]:
#         """
#         Call this once per inference tick, right after run_inference().

#         Returns a dict:
#           {
#             "label": <str>          -- the label to actually log/broadcast
#             "should_log_event": bool -- whether to write to the events table
#             "is_pending": bool        -- true while a FALL is being verified
#             "posture_debug": dict     -- debug info for tuning
#           }

#         Non-FALL labels (STABLE, HIGH_RISK, IMMOBILE) pass straight through
#         unchanged -- this layer only intervenes on FALL.
#         """
#         now = now if now is not None else time.time()
#         state = self._state_for(patient_id)

#         if raw_label != "FALL":
#             # Any non-FALL prediction cancels a pending fall outright.
#             state.pending_since = None
#             state.pending_start_frame = None
#             state.last_confirmed_label = raw_label
#             return {
#                 "label": raw_label, 
#                 "should_log_event": True, 
#                 "is_pending": False,
#                 "posture_debug": {}
#             }

#         # raw_label == "FALL" from the model
#         features = compute_posture_features(flat_frame)
#         fallen, debug_info = is_fallen_posture(features)

#         # Debug logging
#         if self._debug_enabled and patient_id is not None:
#             status = "FALLEN ✓" if fallen else "UPRIGHT ✗"
#             print(f"[PostureVerifier] Patient {patient_id}: {status} | "
#                   f"torso={debug_info['torso_angle']}° (need ≥{TORSO_ANGLE_FALL_DEG}), "
#                   f"height={debug_info['height_ratio']} (need ≤{HEIGHT_RATIO_FALL}), "
#                   f"hip_gap={debug_info['hip_ankle_gap']} (need ≤{HIP_ANKLE_GAP_FALL}), "
#                   f"votes={debug_info['votes']}/{debug_info['min_votes_needed']}")

#         if not fallen:
#             # Model says FALL but posture says upright/bending -> reject outright,
#             # don't even start a pending timer. This is the core bending fix.
#             state.pending_since = None
#             state.pending_start_frame = None
#             return {
#                 "label": state.last_confirmed_label,
#                 "should_log_event": False,
#                 "is_pending": False,
#                 "posture_debug": debug_info
#             }

#         # Posture agrees this looks like a fall -- start or continue the timer
#         if state.pending_since is None:
#             state.pending_since = now
#             state.pending_start_frame = frame_idx if frame_idx is not None else 0
#             if self._debug_enabled:
#                 print(f"[PostureVerifier] Patient {patient_id}: ⏱️ FALL pending, timer started")
#             return {
#                 "label": state.last_confirmed_label,  # hold previous state in UI
#                 "should_log_event": False,
#                 "is_pending": True,
#                 "posture_debug": debug_info
#             }

#         elapsed = now - state.pending_since
#         if elapsed >= CONFIRMATION_SECONDS:
#             # Confirmed: sustained fallen posture for the full window
#             state.last_confirmed_label = "FALL"
#             state.pending_since = None
#             state.pending_start_frame = None
#             if self._debug_enabled:
#                 print(f"[PostureVerifier] Patient {patient_id}: ✅ FALL CONFIRMED after {elapsed:.2f}s")
#             return {
#                 "label": "FALL", 
#                 "should_log_event": True, 
#                 "is_pending": False,
#                 "posture_debug": debug_info
#             }

#         # Still within the confirmation window, still looks fallen -- keep waiting
#         if self._debug_enabled and int(elapsed) % 2 == 0:  # Print less frequently
#             print(f"[PostureVerifier] Patient {patient_id}: ⏳ Pending fall: {elapsed:.2f}s / {CONFIRMATION_SECONDS}s")
        
#         return {
#             "label": state.last_confirmed_label,
#             "should_log_event": False,
#             "is_pending": True,
#             "posture_debug": debug_info
#         }

#     def reset(self, patient_id: str) -> None:
#         """Call when a monitoring session ends/switches patient."""
#         self._states.pop(patient_id, None)
#         if self._debug_enabled:
#             print(f"[PostureVerifier] Patient {patient_id}: Session reset")