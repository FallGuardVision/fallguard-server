"""
Phase 5 -- Wire rolling buffer to inference.

Streams a real per-video keypoint sequence frame-by-frame through the
RollingBuffer, runs each emitted window through the model, and logs
predictions over "time" so we can sanity-check the state transitions
(ideally: mostly STABLE, with a HIGH_RISK -> FALL -> IMMOBILE stretch
somewhere if this clip actually contains a fall).
"""

import numpy as np
import tensorflow as tf

MODEL_PATH = r"Z:\College\FallDetectionFYP\models\bilstm_v2\fall_detection_bilstm_v2.keras"
MEAN_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_mean_v2.npy"
STD_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_std_v2.npy"
SEQUENCE_PATH = r"Z:\College\FallDetectionFYP\processed_data\skeletons_resampled\le2i\Coffee_room_01__video (1).npy"
LABELS_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels_resampled\le2i\Coffee_room_01__video (1).npy"

CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]
WINDOW_SIZE = 30
STRIDE = 5


class RollingBuffer:
    def __init__(self, window_size=WINDOW_SIZE, stride=STRIDE):
        self.window_size = window_size
        self.stride = stride
        self.buffer = []
        self.frames_since_last_window = 0

    def add_frame(self, frame_132):
        frame_132 = np.asarray(frame_132, dtype=np.float32)
        assert frame_132.shape == (132,), f"expected (132,), got {frame_132.shape}"

        self.buffer.append(frame_132)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) < self.window_size:
            return None

        should_emit = self.frames_since_last_window == 0
        self.frames_since_last_window = (self.frames_since_last_window + 1) % self.stride

        if should_emit:
            return np.array(self.buffer, dtype=np.float32)
        return None


def preprocess_window(window_flat_132, feat_mean, feat_std):
    T = window_flat_132.shape[0]
    reshaped = window_flat_132.reshape(T, 33, 4)
    normalized = (reshaped - feat_mean) / feat_std
    return normalized.reshape(T, 132)


if __name__ == "__main__":
    print("=== Loading model + normalization stats ===")
    model = tf.keras.models.load_model(MODEL_PATH)
    feat_mean = np.load(MEAN_PATH)
    feat_std = np.load(STD_PATH)

    print("\n=== Loading test sequence ===")
    sequence = np.load(SEQUENCE_PATH)
    print("Loaded shape:", sequence.shape, "dtype:", sequence.dtype)

    # normalize shape to (T, 132) regardless of whether it's stored as
    # (T,33,4) or already flat
    if sequence.ndim == 3:
        T = sequence.shape[0]
        sequence = sequence.reshape(T, 132)
        print("Reshaped to:", sequence.shape)
    elif sequence.ndim == 2 and sequence.shape[1] == 132:
        pass
    else:
        raise ValueError(f"Unexpected sequence shape: {sequence.shape}")

    print("\n=== Loading ground truth labels ===")
    labels = np.load(LABELS_PATH)
    print("Labels shape:", labels.shape)

    print(f"\n=== Streaming {sequence.shape[0]} frames through buffer + model ===")
    buf = RollingBuffer()
    predictions_log = []  # (frame_index, predicted_class, confidence, true_class)

    for frame_idx in range(sequence.shape[0]):
        frame = sequence[frame_idx]
        window = buf.add_frame(frame)

        if window is not None:
            normalized = preprocess_window(window, feat_mean, feat_std)
            batch = np.expand_dims(normalized, axis=0)
            pred = model.predict(batch, verbose=0)[0]
            predicted_idx = int(np.argmax(pred))
            true_idx = int(labels[frame_idx])  # label = last frame in window, matches training
            predictions_log.append(
                (frame_idx, CLASS_NAMES[predicted_idx], float(pred[predicted_idx]), CLASS_NAMES[true_idx])
            )

    print(f"\nTotal windows evaluated: {len(predictions_log)}")
    print("\n=== Prediction timeline (frame_idx, predicted, confidence, TRUE) ===")
    correct = 0
    for entry in predictions_log:
        match = "OK" if entry[1] == entry[3] else "XX"
        if entry[1] == entry[3]:
            correct += 1
        print(f"frame {entry[0]:4d}  ->  pred={entry[1]:10s} ({entry[2]:.3f})  true={entry[3]:10s}  {match}")

    print(f"\nAccuracy on this clip: {correct}/{len(predictions_log)} = {correct/len(predictions_log):.2%}")

    print("\n=== Predicted class distribution ===")
    from collections import Counter
    counts = Counter(p[1] for p in predictions_log)
    for cls in CLASS_NAMES:
        print(f"{cls:10s}: {counts.get(cls, 0)}")

    print("\n=== DONE. Paste this whole output back. ===")