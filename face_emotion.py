import os
 
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
 
import tensorflow as tf  # noqa: F401
from deepface import DeepFace
 
MIN_CONFIDENCE = 0.30   # dominant emotion must exceed 30% to be considered reliable
 
 
class FaceEmotion:
    def __init__(self):
        pass
 
    def predict_from_frame(self, frame):
        """
        frame: BGR image (numpy array) from OpenCV.
        Returns: {"label": str, "scores": {emotion: float}} 
                 or {"label": None, "scores": {}} if confidence is too low
                 (no face detected / poor quality image).
        """
        res = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if isinstance(res, list):
            res = res[0]
 
        dominant = res.get("dominant_emotion", "neutral")
        emotions = res.get("emotion", {})
        scores = {k: float(v) for k, v in emotions.items()}
 
        # Normalize scores to probabilities (DeepFace returns percentages)
        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}
 
        top_score = scores.get(dominant, 0.0)
        if top_score < MIN_CONFIDENCE:
            # Not confident enough — signal to caller that face wasn't useful
            return {"label": None, "scores": scores}
 
        return {"label": dominant, "scores": scores}
 