"""
Phase 2 — Preprocessing function.

Confirmed from Phase 1:
  - Model expects flat input (30, 132)
  - feat_mean_v2 / feat_std_v2 have shape (1, 1, 4) -> normalization was
    computed per-feature (x, y, z, visibility), broadcasting across all
    33 landmarks and all frames, on the (T, 33, 4) shaped data --
    BEFORE flattening to (T, 132).

So correct order for a live window:
  (30, 132) --reshape--> (30, 33, 4) --normalize--> (30, 33, 4) --flatten--> (30, 132)
"""

import numpy as np
import tensorflow as tf

MODEL_PATH = r"Z:\College\FallDetectionFYP\models\bilstm_v2\fall_detection_bilstm_v2.keras"
MEAN_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_mean_v2.npy"
STD_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_std_v2.npy"

CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]

feat_mean = np.load(MEAN_PATH)  # shape (1,1,4)
feat_std = np.load(STD_PATH)    # shape (1,1,4)


def preprocess_window(window_flat_132):
    """
    window_flat_132: np.ndarray, shape (30, 132), raw (unnormalized) keypoints
    Returns: np.ndarray, shape (30, 132), normalized, ready for model input
    """
    T = window_flat_132.shape[0]
    reshaped = window_flat_132.reshape(T, 33, 4)          # (30, 33, 4)
    normalized = (reshaped - feat_mean) / feat_std          # broadcast (1,1,4)
    flattened = normalized.reshape(T, 132)                  # back to (30, 132)
    return flattened


if __name__ == "__main__":
    print("=== Loading model ===")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n=== Testing preprocessing shapes ===")
    dummy_raw = np.random.rand(30, 132).astype(np.float32)
    dummy_norm = preprocess_window(dummy_raw)
    print("Raw shape:", dummy_raw.shape)
    print("Normalized shape:", dummy_norm.shape)
    print("Normalized sample (first row, first 8 vals):", dummy_norm[0][:8])

    print("\n=== Running inference on properly normalized dummy window ===")
    batch_input = np.expand_dims(dummy_norm, axis=0)  # (1, 30, 132)
    pred = model.predict(batch_input, verbose=0)
    print("Raw prediction output:", pred)
    print("Predicted class:", CLASS_NAMES[np.argmax(pred)])
    print("Confidence:", float(np.max(pred)))

    print("\n=== Sanity check: does prediction change with normalization? ===")
    pred_unnorm = model.predict(np.expand_dims(dummy_raw, axis=0), verbose=0)
    print("Unnormalized pred:", pred_unnorm, "->", CLASS_NAMES[np.argmax(pred_unnorm)])
    print("Normalized pred  :", pred, "->", CLASS_NAMES[np.argmax(pred)])
    if not np.allclose(pred, pred_unnorm):
        print("Good -- normalization changes the output as expected.")
    else:
        print("WARNING: predictions identical -- check normalization is actually applying.")

    print("\n=== DONE. Paste this whole output back. ===")