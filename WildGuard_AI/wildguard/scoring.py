"""Turn behaviour probabilities (+ movement) into an arousal / stress index and a plain-language summary."""
import numpy as np

from .behavior import AROUSAL, BEHAVIOUR_NAMES

W_BEHAVIOUR, W_MOTION = 0.7, 0.3
CALM_BELOW, STRESSED_FROM = 30, 60


def stress_series(probs, motion=None):
    """probs (n, n_behaviours); motion (n,) in 0..1 or None  ->  index 0..100 per frame"""
    b = probs @ AROUSAL
    if motion is None:
        return b * 100.0
    return (W_BEHAVIOUR * b + W_MOTION * np.asarray(motion)) * 100.0


def level(index):
    if index < CALM_BELOW:
        return "Calm"
    if index < STRESSED_FROM:
        return "Alert"
    return "Agitated / stressed"


def top_behaviours(probs, k=3):
    mean = probs.mean(axis=0)
    idx = np.argsort(-mean)[:k]
    return [(BEHAVIOUR_NAMES[i], float(mean[i])) for i in idx]


def advice(lvl, status_code):
    parts = []
    if lvl == "Agitated / stressed":
        parts.append("The animal looks agitated - keep a safe distance, avoid noise and flash, and do not approach.")
    elif lvl == "Alert":
        parts.append("The animal is watchful - move slowly and give it space.")
    else:
        parts.append("The animal looks relaxed.")
    if status_code in ("CR", "EN", "VU"):
        parts.append("This species is threatened: consider reporting the sighting to local wildlife authorities or a citizen-science project.")
    return " ".join(parts)
