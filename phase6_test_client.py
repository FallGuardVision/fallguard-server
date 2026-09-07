"""
Test client for the /ws/session WebSocket endpoint.
Streams a real sequence frame-by-frame, prints predictions as they arrive.

Run phase6_api.py first (uvicorn phase6_api:app --reload), then run this.
Needs: pip install websockets
"""

import asyncio
import json
import numpy as np
import websockets

SEQUENCE_PATH = r"Z:\College\FallDetectionFYP\processed_data\skeletons_resampled\le2i\Coffee_room_01__video (1).npy"
LABELS_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels_resampled\le2i\Coffee_room_01__video (1).npy"
CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]

WS_URL = "ws://127.0.0.1:8000/ws/session"


async def stream_sequence():
    sequence = np.load(SEQUENCE_PATH)
    labels = np.load(LABELS_PATH)

    if sequence.ndim == 3:
        T = sequence.shape[0]
        sequence = sequence.reshape(T, 132)

    print(f"Streaming {sequence.shape[0]} frames over WebSocket...\n")

    async with websockets.connect(WS_URL) as ws:
        correct = 0
        total = 0
        for frame_idx in range(sequence.shape[0]):
            frame = sequence[frame_idx].tolist()
            await ws.send(json.dumps({"frame": frame}))

            # server only replies once a window completes; peek non-blocking
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.05)
                result = json.loads(response)
                if "predicted_class" in result:
                    true_class = CLASS_NAMES[int(labels[frame_idx])]
                    match = "OK" if result["predicted_class"] == true_class else "XX"
                    if match == "OK":
                        correct += 1
                    total += 1
                    print(
                        f"frame {frame_idx:4d}  pred={result['predicted_class']:10s} "
                        f"({result['confidence']:.3f})  true={true_class:10s}  {match}"
                    )
            except asyncio.TimeoutError:
                pass  # no window completed on this frame yet

        print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}" if total else "No predictions received.")


if __name__ == "__main__":
    asyncio.run(stream_sequence())