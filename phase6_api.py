"""
Backend API -- single-window /predict (Phase 3, corrected) + live WebSocket
streaming endpoint (Phase 6), now with a posture-verification layer to fix
bending-vs-falling false positives.

Run: uvicorn phase3_api:app --reload
"""

import numpy as np
import tensorflow as tf
import sqlite3
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from posture_verification import PostureVerifier

import os
import tempfile
import zipfile
from pathlib import Path

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "fall_detection_bilstm_v2_fixed.keras",
)
DB_PATH = os.environ.get("DB_PATH", "events.db")

CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]
ALERT_CLASSES = {"HIGH_RISK", "FALL", "IMMOBILE"}  # states worth logging as events
WINDOW_SIZE = 30
STRIDE = 5

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,https://fallguard-ui.vercel.app",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singleton: holds per-session pending-fall state across
# all WebSocket connections. Keyed by patient_id (or connection id when
# no patient_id is provided -- see live_session below).
posture_verifier = PostureVerifier()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            timestamp TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_patient(name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO patients (name, created_at) VALUES (?, ?)",
        (name, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, name, created_at FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_event(predicted_class: str, confidence: float, patient_id: int = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (patient_id, timestamp, predicted_class, confidence) VALUES (?, ?, ?, ?)",
        (patient_id, datetime.now(timezone.utc).isoformat(), predicted_class, confidence),
    )
    conn.commit()
    conn.close()


def get_recent_events(limit: int = 50, patient_id: int = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if patient_id is not None:
        rows = conn.execute(
            """
            SELECT events.id, events.timestamp, events.predicted_class, events.confidence,
                   events.patient_id, patients.name AS patient_name
            FROM events LEFT JOIN patients ON events.patient_id = patients.id
            WHERE events.patient_id = ?
            ORDER BY events.id DESC LIMIT ?
            """,
            (patient_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT events.id, events.timestamp, events.predicted_class, events.confidence,
                   events.patient_id, patients.name AS patient_name
            FROM events LEFT JOIN patients ON events.patient_id = patients.id
            ORDER BY events.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


init_db()

print("Loading model...")
model_path = Path(MODEL_PATH)
if model_path.is_dir():
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as archive_file:
        model_archive_path = Path(archive_file.name)

    try:
        with zipfile.ZipFile(model_archive_path, "w") as model_archive:
            for model_file in model_path.iterdir():
                if model_file.is_file():
                    model_archive.write(model_file, arcname=model_file.name)
        model = tf.keras.models.load_model(str(model_archive_path))
    finally:
        model_archive_path.unlink(missing_ok=True)
else:
    model = tf.keras.models.load_model(str(model_path))
print("Model loaded.")


def preprocess_window(window_flat_132: np.ndarray) -> np.ndarray:
    # Bi-LSTM v2 was trained on raw, unnormalized flattened landmarks.
    return window_flat_132


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

    def reset(self):
        self.buffer = []
        self.frames_since_last_window = 0


def run_inference(window_flat_132: np.ndarray) -> dict:
    normalized = preprocess_window(window_flat_132)
    batch = np.expand_dims(normalized, axis=0)
    pred = model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(pred))
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": float(pred[predicted_idx]),
        "class_probabilities": {CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))},
    }


class WindowRequest(BaseModel):
    window: List[List[float]]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict


@app.get("/")
def health_check():
    return {"status": "ok", "model_input_shape": str(model.input_shape)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: WindowRequest):
    """
    Stateless single-window classification (no patient session, no history) --
    posture verification's temporal confirmation needs a session to track,
    so it intentionally does NOT apply here. Use /ws/session for verified,
    alert-worthy predictions.
    """
    window = np.array(req.window, dtype=np.float32)
    if window.shape != (30, 132):
        return {
            "predicted_class": "ERROR",
            "confidence": 0.0,
            "class_probabilities": {"error": f"expected shape (30,132), got {window.shape}"},
        }
    return run_inference(window)


class PatientCreateRequest(BaseModel):
    name: str


@app.post("/patients")
def add_patient(req: PatientCreateRequest):
    new_id = create_patient(req.name)
    return {"id": new_id, "name": req.name}


@app.get("/patients")
def list_patients():
    return get_patients()


@app.get("/events")
def list_events(limit: int = 50, patient_id: int = None):
    return get_recent_events(limit, patient_id)


@app.websocket("/ws/session")
async def live_session(websocket: WebSocket, patient_id: int = None):
    """
    Client connects (optionally with ?patient_id=<id> in the URL), then
    sends one frame at a time as JSON: {"frame": [132 floats]}.
    Server internally buffers frames into 30-length windows (stride=5,
    matching training) and pushes a prediction back over the same socket
    every time a window completes.

    Raw model predictions of FALL are passed through the posture verifier
    before anything is logged or alerted: a FALL with upright/bending
    posture is rejected outright, and a FALL with genuinely fallen posture
    must hold for ~1.75s before being confirmed. HIGH_RISK, STABLE, and
    IMMOBILE pass through unchanged -- only FALL is intercepted.

    Logs an event to the DB (tagged with patient_id, if provided) whenever
    the *verified* predicted class transitions into HIGH_RISK, FALL, or
    IMMOBILE.
    """
    await websocket.accept()
    buffer = RollingBuffer()
    last_class = None

    # Key the verifier's per-session state by patient_id when available;
    # fall back to the connection identity so two anonymous sessions
    # (no patient_id) don't share a pending-fall timer.
    session_key = patient_id if patient_id is not None else id(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            frame = data.get("frame")

            if frame is None or len(frame) != 132:
                await websocket.send_json({"error": f"expected 'frame' with 132 values, got {frame}"})
                continue

            window = buffer.add_frame(frame)
            if window is not None:
                result = run_inference(window)
                raw_predicted = result["predicted_class"]

                verified = posture_verifier.process(
                    patient_id=session_key,
                    raw_label=raw_predicted,
                    flat_frame=window[-1],  # most recent frame in the window
                )
                effective_class = verified["label"]

                # Send the verified label to the frontend (not the raw one),
                # plus a flag the UI can use to show "checking..." during
                # the confirmation window if desired.
                payload = dict(result)
                payload["predicted_class"] = effective_class
                payload["pending_fall_check"] = verified["is_pending"]
                await websocket.send_json(payload)

                if (
                    verified["should_log_event"]
                    and effective_class in ALERT_CLASSES
                    and effective_class != last_class
                ):
                    log_event(effective_class, result["confidence"], patient_id)
                last_class = effective_class

    except WebSocketDisconnect:
        posture_verifier.reset(session_key)
        print("Client disconnected, session buffer and posture state discarded.")





# """
# Backend API -- single-window /predict (Phase 3, corrected) + live WebSocket
# streaming endpoint (Phase 6), now with a posture-verification layer to fix
# bending-vs-falling false positives.

# Run: uvicorn phase6_api:app --reload
# """

# import numpy as np
# import tensorflow as tf
# import sqlite3
# from datetime import datetime, timezone
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List

# from posture_verification import PostureVerifier

# MODEL_PATH = r"Z:\College\FallDetectionFYP\models\bilstm_v2\fall_detection_bilstm_v2.keras"
# DB_PATH = r"Z:\College\FallDetectionFYP\events.db"

# CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]
# ALERT_CLASSES = {"HIGH_RISK", "FALL", "IMMOBILE"}  # states worth logging as events
# WINDOW_SIZE = 30
# STRIDE = 5

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Module-level singleton: holds per-session pending-fall state across
# # all WebSocket connections. Keyed by patient_id (or connection id when
# # no patient_id is provided -- see live_session below).
# posture_verifier = PostureVerifier()


# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         """
#         CREATE TABLE IF NOT EXISTS patients (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             name TEXT NOT NULL,
#             created_at TEXT NOT NULL
#         )
#         """
#     )
#     conn.execute(
#         """
#         CREATE TABLE IF NOT EXISTS events (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             patient_id INTEGER,
#             timestamp TEXT NOT NULL,
#             predicted_class TEXT NOT NULL,
#             confidence REAL NOT NULL,
#             FOREIGN KEY (patient_id) REFERENCES patients (id)
#         )
#         """
#     )
#     conn.commit()
#     conn.close()


# def create_patient(name: str):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.execute(
#         "INSERT INTO patients (name, created_at) VALUES (?, ?)",
#         (name, datetime.now(timezone.utc).isoformat()),
#     )
#     conn.commit()
#     new_id = cursor.lastrowid
#     conn.close()
#     return new_id


# def get_patients():
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     rows = conn.execute("SELECT id, name, created_at FROM patients ORDER BY id DESC").fetchall()
#     conn.close()
#     return [dict(row) for row in rows]


# def log_event(predicted_class: str, confidence: float, patient_id: int = None):
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         "INSERT INTO events (patient_id, timestamp, predicted_class, confidence) VALUES (?, ?, ?, ?)",
#         (patient_id, datetime.now(timezone.utc).isoformat(), predicted_class, confidence),
#     )
#     conn.commit()
#     conn.close()


# def get_recent_events(limit: int = 50, patient_id: int = None):
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     if patient_id is not None:
#         rows = conn.execute(
#             """
#             SELECT events.id, events.timestamp, events.predicted_class, events.confidence,
#                    events.patient_id, patients.name AS patient_name
#             FROM events LEFT JOIN patients ON events.patient_id = patients.id
#             WHERE events.patient_id = ?
#             ORDER BY events.id DESC LIMIT ?
#             """,
#             (patient_id, limit),
#         ).fetchall()
#     else:
#         rows = conn.execute(
#             """
#             SELECT events.id, events.timestamp, events.predicted_class, events.confidence,
#                    events.patient_id, patients.name AS patient_name
#             FROM events LEFT JOIN patients ON events.patient_id = patients.id
#             ORDER BY events.id DESC LIMIT ?
#             """,
#             (limit,),
#         ).fetchall()
#     conn.close()
#     return [dict(row) for row in rows]


# init_db()

# print("Loading model...")
# model = tf.keras.models.load_model(MODEL_PATH)
# print("Model loaded.")
# print(f"Model input shape: {model.input_shape}")
# print(f"Model output shape: {model.output_shape}")


# def preprocess_window(window_flat_132: np.ndarray) -> np.ndarray:
#     """
#     Preprocess window for model inference.
#     Bi-LSTM v2 was trained on raw, unnormalized flattened landmarks.
#     """
#     return window_flat_132


# class RollingBuffer:
#     """Rolling buffer for streaming frames into windows."""
    
#     def __init__(self, window_size=WINDOW_SIZE, stride=STRIDE):
#         self.window_size = window_size
#         self.stride = stride
#         self.buffer = []
#         self.frames_since_last_window = 0
#         self.frame_count = 0

#     def add_frame(self, frame_132):
#         """
#         Add a frame to the buffer.
#         Returns: np.ndarray of shape (window_size, 132) if a window is ready,
#                  else None.
#         """
#         frame_132 = np.asarray(frame_132, dtype=np.float32)
#         assert frame_132.shape == (132,), f"expected (132,), got {frame_132.shape}"

#         self.buffer.append(frame_132)
#         self.frame_count += 1
        
#         # Keep buffer capped at window_size
#         if len(self.buffer) > self.window_size:
#             self.buffer.pop(0)

#         # Not enough frames yet
#         if len(self.buffer) < self.window_size:
#             return None

#         # Emit window every `stride` frames
#         should_emit = self.frames_since_last_window == 0
#         self.frames_since_last_window = (self.frames_since_last_window + 1) % self.stride

#         if should_emit:
#             return np.array(self.buffer, dtype=np.float32)
#         return None

#     def reset(self):
#         """Reset the buffer state."""
#         self.buffer = []
#         self.frames_since_last_window = 0
#         self.frame_count = 0


# def run_inference(window_flat_132: np.ndarray) -> dict:
#     """
#     Run model inference on a single window.
    
#     Args:
#         window_flat_132: np.ndarray of shape (30, 132)
        
#     Returns:
#         dict with predicted_class, confidence, and class_probabilities
#     """
#     normalized = preprocess_window(window_flat_132)
#     batch = np.expand_dims(normalized, axis=0)
#     pred = model.predict(batch, verbose=0)[0]
#     predicted_idx = int(np.argmax(pred))
#     return {
#         "predicted_class": CLASS_NAMES[predicted_idx],
#         "confidence": float(pred[predicted_idx]),
#         "class_probabilities": {CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))},
#     }


# class WindowRequest(BaseModel):
#     """Request model for /predict endpoint."""
#     window: List[List[float]]  # 30 frames x 132 values


# class PredictionResponse(BaseModel):
#     """Response model for /predict endpoint."""
#     predicted_class: str
#     confidence: float
#     class_probabilities: dict


# @app.get("/")
# def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "ok",
#         "model_input_shape": str(model.input_shape),
#         "model_output_shape": str(model.output_shape),
#         "classes": CLASS_NAMES,
#         "window_size": WINDOW_SIZE,
#         "stride": STRIDE,
#     }


# @app.post("/predict", response_model=PredictionResponse)
# def predict(req: WindowRequest):
#     """
#     Stateless single-window classification.
    
#     Note: Posture verification is NOT applied here because it requires
#     temporal context. Use /ws/session for verified predictions.
#     """
#     window = np.array(req.window, dtype=np.float32)
#     if window.shape != (30, 132):
#         return {
#             "predicted_class": "ERROR",
#             "confidence": 0.0,
#             "class_probabilities": {"error": f"expected shape (30,132), got {window.shape}"},
#         }
#     return run_inference(window)


# class PatientCreateRequest(BaseModel):
#     """Request model for creating a patient."""
#     name: str


# @app.post("/patients")
# def add_patient(req: PatientCreateRequest):
#     """Create a new patient."""
#     new_id = create_patient(req.name)
#     return {"id": new_id, "name": req.name}


# @app.get("/patients")
# def list_patients():
#     """List all patients."""
#     return get_patients()


# @app.get("/events")
# def list_events(limit: int = 50, patient_id: int = None):
#     """List recent events, optionally filtered by patient."""
#     return get_recent_events(limit, patient_id)


# @app.websocket("/ws/session")
# async def live_session(websocket: WebSocket, patient_id: int = None):
#     """
#     WebSocket endpoint for live streaming inference.
    
#     Client sends frames as JSON: {"frame": [132 floats]}
#     Server responds with predictions when windows are complete.
    
#     Query parameter: ?patient_id=<id> (optional)
    
#     Raw model predictions of FALL are passed through the posture verifier:
#     - FALL + upright/bending posture → rejected (STABLE)
#     - FALL + fallen posture → confirmation timer starts
#     - FALL + sustained fallen posture for 1.25s → confirmed FALL alert
    
#     Logs events to DB when alert-worthy states are confirmed.
#     """
#     await websocket.accept()
#     buffer = RollingBuffer()
#     last_class = None

#     # Key verifier state by patient_id (or connection id if no patient_id)
#     session_key = str(patient_id) if patient_id is not None else str(id(websocket))

#     print(f"[WebSocket] Client connected: session_key={session_key}, patient_id={patient_id}")

#     try:
#         while True:
#             # Receive frame from client
#             data = await websocket.receive_json()
#             frame = data.get("frame")

#             # Validate frame
#             if frame is None or len(frame) != 132:
#                 await websocket.send_json({
#                     "error": f"expected 'frame' with 132 values, got {frame}"
#                 })
#                 continue

#             # Add frame to buffer and get window if ready
#             window = buffer.add_frame(frame)
            
#             if window is not None:
#                 # Run inference
#                 result = run_inference(window)
#                 raw_predicted = result["predicted_class"]

#                 # Apply posture verification
#                 verified = posture_verifier.process(
#                     patient_id=session_key,
#                     raw_label=raw_predicted,
#                     flat_frame=window[-1].tolist(),  # most recent frame
#                     frame_idx=buffer.frame_count,
#                 )
                
#                 effective_class = verified["label"]

#                 # Prepare response payload
#                 payload = {
#                     "predicted_class": effective_class,
#                     "confidence": result["confidence"],
#                     "class_probabilities": result["class_probabilities"],
#                     "raw_predicted": raw_predicted,  # For debugging
#                     "pending_fall_check": verified["is_pending"],
#                     "posture_debug": verified.get("posture_debug", {}),
#                 }
                
#                 await websocket.send_json(payload)

#                 # Log event on state change to an alert class
#                 if (
#                     verified["should_log_event"]
#                     and effective_class in ALERT_CLASSES
#                     and effective_class != last_class
#                 ):
#                     log_event(effective_class, result["confidence"], patient_id)
#                     print(f"[WebSocket] Event logged: {effective_class} for patient {patient_id}")
                
#                 last_class = effective_class

#     except WebSocketDisconnect:
#         # Clean up on disconnect
#         posture_verifier.reset(session_key)
#         print(f"[WebSocket] Client disconnected: session_key={session_key}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)