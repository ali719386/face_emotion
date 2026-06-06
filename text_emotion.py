from transformers import pipeline

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

LABEL_MAP = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprise",
}

# Negation words that flip positive emotions to sad/neutral
NEGATION_WORDS = ("not", "never", "no", "don't", "doesn't", "didn't", "can't",
                  "cannot", "won't", "wasn't", "aren't", "isn't", "hardly",
                  "barely", "nothing", "nowhere", "nobody", "neither")

POSITIVE_EMOTIONS = {"happy", "joy"}

EMOTION_KEYWORDS = {
    "angry":    ("furious", "enraged", "livid", "fuming", "outraged", "infuriated", "rage"),
    "anxious":  ("anxious", "anxiety", "panicking", "dreading", "apprehensive"),
    "disgust":  ("disgusted", "revolted", "repulsed", "sickened", "nauseated"),
    "fear":     ("terrified", "frightened", "petrified", "horrified"),
    "happy":    ("joyful", "ecstatic", "thrilled", "elated", "overjoyed", "euphoric", "wonderful"),
    "joy":      ("jubilant", "gleeful", "exhilarated", "blissful"),
    "neutral":  ("indifferent", "unbothered", "apathetic"),
    "sad":      ("heartbroken", "devastated", "miserable", "grieving", "despairing",
                 "depressed", "hopeless", "worthless", "empty", "broken"),
    "surprise": ("astonished", "astounded", "flabbergasted", "dumbfounded"),
}

KEYWORD_BOOST = 0.15


def has_negation(text: str) -> bool:
    """Check if text contains negation words."""
    words = text.lower().split()
    return any(w in NEGATION_WORDS for w in words)


class TextEmotion:
    def __init__(self, model_name: str = EMOTION_MODEL):
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
        )

    def predict(self, text: str):
        if not text or not text.strip():
            return {"label": "neutral", "scores": {"neutral": 1.0}}

        content = text[:512]
        lowered = content.lower()

        # ---- Model predictions ------------------------------------------------
        raw = self.pipe(content)

        # top_k=None returns [[{label, score}, ...]] — safely unwrap
        if isinstance(raw, list) and len(raw) > 0:
            scores_list = raw[0] if isinstance(raw[0], list) else raw
        else:
            scores_list = raw

        # Map model labels → canonical labels
        emotion_scores: dict[str, float] = {}
        for item in scores_list:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            canonical = LABEL_MAP.get(label, label)
            emotion_scores[canonical] = emotion_scores.get(canonical, 0.0) + score

        # Ensure all canonical emotions exist
        all_emotions = set(LABEL_MAP.values()) | {"anxious", "joy"}
        for e in all_emotions:
            emotion_scores.setdefault(e, 0.0)

        # ---- Negation handling ------------------------------------------------
        # If text has negation AND model predicted a positive emotion with high
        # confidence, redistribute that score to sad
        if has_negation(lowered):
            for pos_emotion in POSITIVE_EMOTIONS:
                pos_score = emotion_scores.get(pos_emotion, 0.0)
                if pos_score > 0.4:  # model is confidently positive but negation present
                    # Transfer 80% of positive score to sad
                    transfer = pos_score * 0.80
                    emotion_scores[pos_emotion] -= transfer
                    emotion_scores["sad"] = emotion_scores.get("sad", 0.0) + transfer

        # ---- Keyword boosting -------------------------------------------------
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0.0) + KEYWORD_BOOST
                    break

        # ---- Normalize --------------------------------------------------------
        total = sum(emotion_scores.values()) or 1.0
        scores = {label: val / total for label, val in emotion_scores.items()}
        top_label = max(scores, key=scores.get)

        return {"label": top_label, "scores": scores}
