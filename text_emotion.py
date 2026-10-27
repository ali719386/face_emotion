import re

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

POSITIVE_EMOTIONS = {"happy"}

# ---------------------------------------------------------------------------
# Only canonical emotions (values of LABEL_MAP, plus "anxious") get boosted.
# Each emotion appears ONCE here — no duplicate keys, so nothing gets
# silently overwritten. Motivational / achievement / success / discipline
# words are folded into "happy" since they carry a positive tone.
# ---------------------------------------------------------------------------
EMOTION_KEYWORDS = {
    "angry": (
        "furious", "enraged", "livid", "fuming", "outraged", "infuriated",
        "rage", "angry", "mad", "annoyed", "hate", "hated", "frustrated",
        "upset",
    ),
    "disgust": (
        "disgusted", "revolted", "repulsed", "sickened", "nauseated",
    ),
    "fear": (
        "terrified", "frightened", "petrified", "horrified", "fear",
        "afraid", "panic", "panicking", "anxious", "worry", "worried",
        "nervous", "dreading", "apprehensive",
    ),
    "neutral": (
        "indifferent", "unbothered", "apathetic",
    ),
    "sad": (
        "cry", "crying", "tears", "tear", "alone", "lonely",
        "heartbroken", "broken", "pain", "hurt", "hurts",
        "miss", "missing", "goodbye", "lost", "loss",
        "grief", "empty", "hopeless", "worthless", "depressed",
        "devastated", "miserable", "suffer", "suffering",
        "nobody", "silent", "silence", "regret", "memories",
        "memory", "dark",
    ),
    "surprise": (
        "wow", "unexpected", "surprised", "astonished", "unbelievable",
    ),
    "happy": (
        "happy", "joy", "joyful", "smile", "smiling", "grateful",
        "gratitude", "kindness", "love", "hope", "hopeful",
        "blessed", "celebrate", "celebration", "excited",
        "wonderful", "amazing", "beautiful", "peace", "success",
        "achievement", "win", "winning", "proud",
        # motivational words merged in (positive tone, no separate label)
        "keep going", "never give up", "believe", "dream", "dreams",
        "future", "discipline", "strong", "strength", "improve",
        "improvement", "hard work", "motivation", "focus", "goal",
        "achieve", "comeback", "begin", "tomorrow", "better days",
        "keep faith", "battle", "hardest", "hard",
    ),
}

KEYWORD_BOOST = 0.3
# When the text contains a non-neutral keyword, neutral's score is scaled
# down by this factor so it can't dominate purely because the sentence
# has no emotionally "loud" wording that the transformer model recognizes.
NEUTRAL_DAMPEN = 0.5
# Cap on how many keyword hits count per emotion, to avoid runaway scores
# on lines that repeat many keywords from the same category.
MAX_KEYWORD_HITS = 2


# Pre-compiled whole-word/phrase patterns so "hope" doesn't match "hopeless",
# "mad" doesn't match "made", "win" doesn't match "window", etc.
KEYWORD_PATTERNS = {
    emotion: [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]
    for emotion, keywords in EMOTION_KEYWORDS.items()
}


def has_negation(text: str) -> bool:
    """Check if text contains negation words (ignores punctuation, handles curly apostrophes)."""
    words = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
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
        all_emotions = set(LABEL_MAP.values()) | {"anxious"}
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

        # ---- Keyword boosting (canonical emotions only) -----------------------
        matched_emotions = set()
        for emotion, patterns in KEYWORD_PATTERNS.items():
            if emotion not in all_emotions:
                continue  # safety net: never create non-canonical labels
            hits = sum(1 for pattern in patterns if pattern.search(lowered))
            if hits:
                boost = KEYWORD_BOOST * min(hits, MAX_KEYWORD_HITS)
                emotion_scores[emotion] = emotion_scores.get(emotion, 0.0) + boost
                matched_emotions.add(emotion)

        # If any real emotion keyword matched, neutral shouldn't be allowed
        # to dominate just because the transformer model defaulted to it.
        if matched_emotions and "neutral" not in matched_emotions:
            emotion_scores["neutral"] *= NEUTRAL_DAMPEN

        # ---- Normalize --------------------------------------------------------
        total = sum(emotion_scores.values()) or 1.0
        scores = {label: val / total for label, val in emotion_scores.items()}
        top_label = max(scores, key=scores.get)

        return {"label": top_label, "scores": scores}