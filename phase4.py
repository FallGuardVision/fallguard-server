# """
# Phase 4 -- Rolling buffer for live windows.

# Accumulates incoming per-frame keypoints (each frame: 132 flat values,
# i.e. 33 landmarks x 4) into a sliding window of 30 frames, matching
# training's create_sliding_windows(window_size=30, stride=5).

# Two stride options were considered:
#   - stride=5 (matches training exactly): emit a new window every 5 frames
#   - stride=1: emit a new window every frame, more responsive but the model
#     never saw this density of overlapping windows during training

# We use stride=5 by default to match training distribution. Change STRIDE
# below if you want tighter/looser responsiveness -- flag it here so we
# discuss before locking in for the live demo.
# """

# import numpy as np

# WINDOW_SIZE = 30
# STRIDE = 5


# class RollingBuffer:
#     def __init__(self, window_size=WINDOW_SIZE, stride=STRIDE):
#         self.window_size = window_size
#         self.stride = stride
#         self.buffer = []  # list of (132,) frames
#         self.frames_since_last_window = 0

#     def add_frame(self, frame_132):
#         """
#         frame_132: array-like, shape (132,) -- one frame of flattened keypoints
#         Returns: np.ndarray shape (window_size, 132) if a new window is ready,
#                  else None
#         """
#         frame_132 = np.asarray(frame_132, dtype=np.float32)
#         assert frame_132.shape == (132,), f"expected (132,), got {frame_132.shape}"

#         self.buffer.append(frame_132)

#         # keep buffer capped at window_size (drop oldest)
#         if len(self.buffer) > self.window_size:
#             self.buffer.pop(0)

#         # not enough frames yet
#         if len(self.buffer) < self.window_size:
#             return None

#         self.frames_since_last_window += 1

#         # only emit a window every `stride` frames once buffer is full
#         if self.frames_since_last_window >= self.stride:
#             self.frames_since_last_window = 0
#             return np.array(self.buffer, dtype=np.float32)  # (30, 132)

#         return None

#     def reset(self):
#         self.buffer = []
#         self.frames_since_last_window = 0


# if __name__ == "__main__":
#     # Simulate streaming frames from a real sequence file to sanity check
#     # window count and shape.
#     TEST_NPY_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\upfall"  # EDIT: point to a real per-frame sequence npy if you have one, else this uses random data below

#     print("=== Simulating live stream with random frames ===")
#     buf = RollingBuffer()
#     n_frames_to_simulate = 200
#     windows_emitted = 0

#     for i in range(n_frames_to_simulate):
#         fake_frame = np.random.rand(132).astype(np.float32)
#         window = buf.add_frame(fake_frame)
#         if window is not None:
#             windows_emitted += 1
#             assert window.shape == (30, 132)

#     print(f"Frames streamed: {n_frames_to_simulate}")
#     print(f"Windows emitted: {windows_emitted}")

#     # expected count: after first 30 frames fill buffer, one window every
#     # `stride` frames after that
#     expected = 1 + (n_frames_to_simulate - WINDOW_SIZE) // STRIDE
#     print(f"Expected windows (approx): {expected}")

#     print("\n=== DONE ===")

"""
Phase 4 -- Rolling buffer for live windows.

Accumulates incoming per-frame keypoints (each frame: 132 flat values,
i.e. 33 landmarks x 4) into a sliding window of 30 frames, matching
training's create_sliding_windows(window_size=30, stride=5).

Two stride options were considered:
  - stride=5 (matches training exactly): emit a new window every 5 frames
  - stride=1: emit a new window every frame, more responsive but the model
    never saw this density of overlapping windows during training

We use stride=5 by default to match training distribution. Change STRIDE
below if you want tighter/looser responsiveness -- flag it here so we
discuss before locking in for the live demo.
"""

import numpy as np

WINDOW_SIZE = 30
STRIDE = 5


class RollingBuffer:
    def __init__(self, window_size=WINDOW_SIZE, stride=STRIDE):
        self.window_size = window_size
        self.stride = stride
        self.buffer = []  # list of (132,) frames
        self.frames_since_last_window = 0

    def add_frame(self, frame_132):
        """
        frame_132: array-like, shape (132,) -- one frame of flattened keypoints
        Returns: np.ndarray shape (window_size, 132) if a new window is ready,
                 else None
        """
        frame_132 = np.asarray(frame_132, dtype=np.float32)
        assert frame_132.shape == (132,), f"expected (132,), got {frame_132.shape}"

        self.buffer.append(frame_132)

        # keep buffer capped at window_size (drop oldest)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # not enough frames yet
        if len(self.buffer) < self.window_size:
            return None

        # emit immediately the moment buffer first fills (matches training's
        # start=0 window), then every `stride` frames after that
        should_emit = self.frames_since_last_window == 0
        self.frames_since_last_window = (self.frames_since_last_window + 1) % self.stride

        if should_emit:
            return np.array(self.buffer, dtype=np.float32)  # (30, 132)

        return None

    def reset(self):
        self.buffer = []
        self.frames_since_last_window = 0


if __name__ == "__main__":
    # Simulate streaming frames from a real sequence file to sanity check
    # window count and shape.
    TEST_NPY_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\upfall"  # EDIT: point to a real per-frame sequence npy if you have one, else this uses random data below

    print("=== Simulating live stream with random frames ===")
    buf = RollingBuffer()
    n_frames_to_simulate = 200
    windows_emitted = 0

    for i in range(n_frames_to_simulate):
        fake_frame = np.random.rand(132).astype(np.float32)
        window = buf.add_frame(fake_frame)
        if window is not None:
            windows_emitted += 1
            assert window.shape == (30, 132)

    print(f"Frames streamed: {n_frames_to_simulate}")
    print(f"Windows emitted: {windows_emitted}")

    # expected count: after first 30 frames fill buffer, one window every
    # `stride` frames after that
    expected = 1 + (n_frames_to_simulate - WINDOW_SIZE) // STRIDE
    print(f"Expected windows (approx): {expected}")

    print("\n=== DONE ===")