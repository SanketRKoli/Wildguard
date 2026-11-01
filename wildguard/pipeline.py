"""End-to-end analysis used by the web app and by the tests."""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from . import knowledge, scoring
from .behavior import BEHAVIOUR_NAMES, BehaviourModel
from .recognizer import Candidate, SpeciesRecognizer
from .video import read_video

FAKE = os.environ.get("WG_FAKE") == "1"
RECOGNIZER = SpeciesRecognizer("fake" if FAKE else None)
BEHAVIOUR = BehaviourModel(fake=FAKE)
LOW_CONFIDENCE = 0.20       # below this the species answer is not trusted
CLIP_ACCEPT = 0.45          # CLIP's guess among well-known animals must reach this
SPECIES_FRAMES = 5          # frames of a video given to the species model (it is the slow part)


@dataclass
class Analysis:
    kind: str                                  # image | video
    animal_present: bool = True
    presence: float = 1.0
    candidates: List[Candidate] = field(default_factory=list)
    higher_rank: Optional[str] = None
    info: dict = field(default_factory=dict)
    behaviour_mean: dict = field(default_factory=dict)
    top: list = field(default_factory=list)
    timeline: Optional[pd.DataFrame] = None
    stress_mean: float = 0.0
    stress_peak: float = 0.0
    level: str = "Calm"
    motion_mean: Optional[float] = None
    seconds: Optional[float] = None
    backend: str = ""
    advice: str = ""
    message: str = ""
    uncertain: bool = False
    source: str = "bioclip"


def _evenly(items, n):
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def _analyse_frames(kind, frames, times, motion):
    a = Analysis(kind=kind)
    presence = BEHAVIOUR.animal_presence(frames)
    a.presence = float(np.mean(presence))
    a.backend = RECOGNIZER.backend
    if a.presence < 0.5:
        a.animal_present = False
        a.message = "No animal was detected in this picture. Try a clearer photo where the animal fills more of the frame."
        return a

    cands, higher = RECOGNIZER.identify(_evenly(frames, SPECIES_FRAMES), top_k=3)
    a.backend = RECOGNIZER.active or RECOGNIZER.backend
    a.candidates, a.higher_rank = cands, higher
    best = cands[0] if cands else None
    if best is None or best.score < LOW_CONFIDENCE:
        # BioCLIP is unsure (typical for pets / farm animals): ask the general model about well-known animals
        alt = BEHAVIOUR.common_animal(frames)
        if alt and alt[0][1] >= CLIP_ACCEPT:
            name, prob = alt[0]
            c = Candidate(knowledge.resolve_scientific(name), name.title(), prob, extra={"source": "clip"})
            a.candidates = [c] + [x for x in cands if x.score >= 0.05]
            a.source = "clip"
            best = c
        else:
            a.uncertain = True
            best = None
    if best is not None:
        a.info = knowledge.species_info(best.scientific, best.common)
        subject = best.common or best.scientific
    else:
        subject = "animal"

    probs = BEHAVIOUR.behaviour(frames, subject)
    s = scoring.stress_series(probs, motion if kind == "video" else None)
    a.top = scoring.top_behaviours(probs, 3)
    a.behaviour_mean = {n: float(v) for n, v in zip(BEHAVIOUR_NAMES, probs.mean(axis=0))}
    a.stress_mean, a.stress_peak = float(s.mean()), float(s.max())
    a.level = scoring.level(a.stress_mean)
    a.advice = scoring.advice(a.level, (a.info.get("status") or {}).get("code"))

    rows = {"time_s": np.round(times, 2), "stress_index": np.round(s, 1)}
    if kind == "video":
        rows["movement"] = np.round(np.asarray(motion), 3)
        a.motion_mean = float(np.mean(motion))
    for j, n in enumerate(BEHAVIOUR_NAMES):
        rows[n] = np.round(probs[:, j], 3)
    a.timeline = pd.DataFrame(rows)
    a.timeline.insert(1, "top_behaviour", [BEHAVIOUR_NAMES[i] for i in probs.argmax(axis=1)])
    return a


def analyse_image(img):
    return _analyse_frames("image", [img.convert("RGB")], [0.0], None)


def analyse_video(path):
    v = read_video(path)
    a = _analyse_frames("video", v["frames"], list(v["times"]), v["motion"])
    a.seconds = v["seconds"]
    return a
