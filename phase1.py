"""
Phase 1 — Model loading & sanity check.

Run this in venv_training (has TensorFlow).
Confirms:
  - model loads and its expected input shape
  - feat_mean_v2 / feat_std_v2 shapes (to know exactly how normalization
    was applied: per-landmark-feature over flattened 132-dim, or (33,4))
  - a dummy forward pass works end-to-end
"""

import numpy as np
import tensorflow as tf

MODEL_PATH = r"Z:\College\FallDetectionFYP\models\bilstm_v2\fall_detection_bilstm_v2.keras"
MEAN_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_mean_v2.npy"
STD_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_std_v2.npy"

CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]

print("=== Loading model ===")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

print("\n=== Model input/output shape ===")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

print("\n=== Loading normalization stats ===")
feat_mean = np.load(MEAN_PATH)
feat_std = np.load(STD_PATH)
print("feat_mean shape:", feat_mean.shape, "dtype:", feat_mean.dtype)
print("feat_std  shape:", feat_std.shape, "dtype:", feat_std.dtype)
print("feat_mean sample:", feat_mean.flatten()[:5])
print("feat_std  sample:", feat_std.flatten()[:5])

print("\n=== Building dummy input to match model's expected shape ===")
# model.input_shape will look like (None, 30, 33, 4) or (None, 30, 132)
input_shape = model.input_shape
window_shape = input_shape[1:]  # drop batch dim
print("Window shape expected by model:", window_shape)

dummy_raw = np.random.rand(*window_shape).astype(np.float32)
print("Dummy raw window shape:", dummy_raw.shape)

print("\n=== Applying normalization ===")
# Try direct broadcast; if shapes don't align, this will error and tell us
# exactly what needs reshaping before we write the real preprocessing fn.
try:
    dummy_norm = (dummy_raw - feat_mean) / feat_std
    print("Normalization broadcast succeeded. Normalized shape:", dummy_norm.shape)
except Exception as e:
    print("Normalization broadcast FAILED:", e)
    print("--> feat_mean/feat_std shape doesn't directly match window shape.")
    print("--> Paste this output back so we fix the reshape logic.")
    dummy_norm = dummy_raw  # fall back so script still runs to completion

print("\n=== Running dummy inference ===")
batch_input = np.expand_dims(dummy_norm, axis=0)  # add batch dim
print("Batch input shape:", batch_input.shape)

pred = model.predict(batch_input, verbose=0)
print("Raw prediction output:", pred)
print("Predicted class:", CLASS_NAMES[np.argmax(pred)])
print("Confidence:", float(np.max(pred)))

print("\n=== DONE. Paste this whole output back. ===")