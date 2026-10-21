"""Semantic end-of-turn detection with Smart Turn v3 (pipecat-ai, BSD-2): a Whisper-tiny-encoder classifier
that judges from the audio itself (words and intonation) whether the speaker has finished. Runs on the CPU
in a few ms, so it can be consulted during the end-of-turn pause."""
import os
from pathlib import Path

import numpy as np

MODEL = Path(os.environ.get("MING_TURN_MODEL", Path(__file__).parent / "model_extra" / "smart-turn-v3.2-cpu.onnx"))
URL = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
N = 8 * 16000  # the model looks at the last 8 s


class TurnDetector:
    def __init__(self, path=MODEL, threads=2):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)

    def prob(self, audio16k: np.ndarray) -> float:
        """Probability that the turn is complete, for float32 16 kHz audio ending at the current moment."""
        x = np.asarray(audio16k, np.float32)[-N:]
        x = np.pad(x, (N - len(x), 0))  # the model expects the speech at the end, zeros in front
        f = self.fe(x,sampling_rate=16000, return_tensors="np", padding="max_length", max_length=N,
                    truncation=True, do_normalize=True).input_features.astype(np.float32)
        return float(self.sess.run(None, {"input_features": f})[0].reshape(-1)[0])


def available():
    return MODEL.exists()
