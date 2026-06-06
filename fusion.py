from typing import Dict, Iterable, Optional, Set
 
DEFAULT_WEIGHTS = {"face": 0.4, "voice": 0.3, "text": 0.3}
CANONICAL_EMOTIONS = (
    "angry",
    "anxious",
    "disgust",
    "fear",
    "happy",
    "joy",
    "neutral",
    "sad",
    "surprise",
)
 
 
def normalize_scores(
    scores: Dict[str, float],
    canonical_emotions: Iterable[str] = CANONICAL_EMOTIONS,
) -> Dict[str, float]:
    normalized = {emotion: 0.0 for emotion in canonical_emotions}
 
    for label, raw_score in (scores or {}).items():
        key = (label or "").strip().lower()
        if key in normalized:
            try:
                normalized[key] += max(float(raw_score), 0.0)
            except (TypeError, ValueError):
                continue
 
    total = sum(normalized.values())
    if total <= 0:
        normalized["neutral"] = 1.0
        return normalized
 
    return {key: value / total for key, value in normalized.items()}
 
 
def _redistribute_weights(
    weights: Dict[str, float],
    active: Set[str],
) -> Dict[str, float]:
    """
    Return a new weight dict where inactive modalities have their weight
    distributed proportionally among the active ones.
    If no modality is active, return equal weights for all.
    """
    if not active:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
 
    inactive_total = sum(v for k, v in weights.items() if k not in active)
    active_total   = sum(v for k, v in weights.items() if k in active) or 1.0
 
    new_weights: Dict[str, float] = {}
    for k, v in weights.items():
        if k in active:
            # Add a proportional share of inactive weight
            new_weights[k] = v + inactive_total * (v / active_total)
        else:
            new_weights[k] = 0.0
 
    # Re-normalize to sum=1
    total = sum(new_weights.values()) or 1.0
    return {k: v / total for k, v in new_weights.items()}
 
 
class FusionEngine:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
 
    def fuse(
        self,
        face_res: dict,
        voice_res: dict,
        text_res: dict,
        active_modalities: Optional[Set[str]] = None,
    ) -> dict:
        """
        Each res is dict: {"label": label, "scores": {label: prob, ...}}
 
        active_modalities: set of modality names that actually received real input
            e.g. {"text"} when user typed text only, {"face","text"} for image+text.
            When None, all three are treated as active (original behaviour).
 
        Inactive modalities have their weights redistributed to active ones so that
        missing inputs cannot drag the result toward neutral.
        """
        if active_modalities is None:
            effective_weights = self.weights
        else:
            effective_weights = _redistribute_weights(self.weights, active_modalities)
 
        face_scores  = normalize_scores(face_res.get("scores", {}))
        voice_scores = normalize_scores(voice_res.get("scores", {}))
        text_scores  = normalize_scores(text_res.get("scores", {}))
 
        final_scores = {emotion: 0.0 for emotion in CANONICAL_EMOTIONS}
 
        for emotion in CANONICAL_EMOTIONS:
            final_scores[emotion] += effective_weights.get("face",  0.0) * face_scores[emotion]
            final_scores[emotion] += effective_weights.get("voice", 0.0) * voice_scores[emotion]
            final_scores[emotion] += effective_weights.get("text",  0.0) * text_scores[emotion]
 
        total = sum(final_scores.values()) or 1.0
        final_scores = {key: value / total for key, value in final_scores.items()}
        top_label = max(final_scores, key=final_scores.get)
 
        return {"label": top_label, "scores": final_scores}
 