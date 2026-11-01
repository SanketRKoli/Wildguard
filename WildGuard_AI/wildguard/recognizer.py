"""Worldwide species recognition.
Primary: BioCLIP (via `pybioclip`) - a vision-language model trained on millions of images covering ~450,000 taxa.
Fallback: general CLIP over a list of well-known animals (used only if BioCLIP cannot load)."""
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .animals_list import COMMON_ANIMALS


@dataclass
class Candidate:
    scientific: str
    common: str
    score: float
    family: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def display(self):
        return self.common or self.scientific


def _clean_sci(d):
    sp = (d.get("species") or "").strip()
    genus = (d.get("genus") or "").strip()
    epi = (d.get("species_epithet") or "").strip()
    if sp and " " not in sp and genus:
        sp = f"{genus} {sp}"
    if not sp and genus and epi:
        sp = f"{genus} {epi}"
    return sp


class SpeciesRecognizer:
    def __init__(self, backend=None):
        self.backend = backend or os.environ.get("WG_BACKEND", "bioclip")   # bioclip | clip | fake
        self.active = None
        self._lock = threading.Lock()
        self._clf = None
        self._clip = None
        self._proc = None

    # ---------------- loading ----------------
    def _load(self):
        if self.active:
            return
        with self._lock:
            if self.active:
                return
            if self.backend == "fake":
                self.active = "fake"
                return
            if self.backend == "bioclip":
                try:
                    from bioclip import TreeOfLifeClassifier
                    self._clf = TreeOfLifeClassifier()
                    self.active = "bioclip"
                    return
                except Exception as e:                       # fall back instead of crashing the app
                    print("[WildGuard] BioCLIP unavailable, using CLIP fallback:", repr(e))
            import torch
            from transformers import CLIPModel, CLIPProcessor
            name = os.environ.get("WG_CLIP", "openai/clip-vit-base-patch32")
            self._clip = CLIPModel.from_pretrained(name).eval()
            self._proc = CLIPProcessor.from_pretrained(name)
            self.active = "clip"

    # ---------------- bioclip ----------------
    def _bioclip_predict(self, path, rank, k):
        try:
            return self._clf.predict(path, rank, k=k)
        except TypeError:
            return self._clf.predict(path, rank)

    def _identify_bioclip(self, frames, top_k) -> Tuple[List[Candidate], Optional[str]]:
        from bioclip import Rank
        acc, meta = {}, {}
        paths = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, im in enumerate(frames):
                im = im.convert("RGB")
                im.thumbnail((768, 768))
                p = os.path.join(tmp, f"f{i}.jpg")
                im.save(p, quality=90)
                paths.append(p)
            for p in paths:
                for d in self._bioclip_predict(p, Rank.SPECIES, 25):
                    if d.get("kingdom") and d["kingdom"] != "Animalia":
                        continue
                    sci = _clean_sci(d)
                    if not sci:
                        continue
                    acc[sci] = acc.get(sci, 0.0) + float(d.get("score", 0.0))
                    meta[sci] = (d.get("common_name") or "", d.get("family") or "")
            cands = [Candidate(s, meta[s][0], v / len(paths), meta[s][1]) for s, v in acc.items()]
            cands.sort(key=lambda c: -c.score)
            cands = cands[:top_k]
            higher = None
            if not cands or cands[0].score < 0.25:           # unsure at species level -> try the family level
                try:
                    mid = paths[len(paths) // 2]
                    fam = [d for d in self._bioclip_predict(mid, Rank.FAMILY, 5) if d.get("kingdom", "Animalia") == "Animalia"]
                    if fam:
                        higher = f"{fam[0].get('family', '')} family (score {float(fam[0].get('score', 0)):.2f})"
                except Exception:
                    pass
        return cands, higher

    # ---------------- clip fallback ----------------
    def _identify_clip(self, frames, top_k):
        import torch
        prompts = [f"a photo of a {n}" for n in COMMON_ANIMALS]
        inputs = self._proc(text=prompts, images=list(frames), return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self._clip(**inputs).logits_per_image.cpu().numpy()
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = (p / p.sum(axis=1, keepdims=True)).mean(axis=0)
        idx = np.argsort(-p)[:top_k]
        return [Candidate("", COMMON_ANIMALS[i].title(), float(p[i])) for i in idx], None

    # ---------------- public ----------------
    def identify(self, frames, top_k=3):
        self._load()
        if self.active == "fake":
            seed = int(np.asarray(frames[0].resize((8, 8))).mean())
            names = [("Panthera tigris", "Tiger", "Felidae"), ("Elephas maximus", "Asian Elephant", "Elephantidae"),
                     ("Pavo cristatus", "Indian Peafowl", "Phasianidae"), ("Passer domesticus", "House Sparrow", "Passeridae")]
            order = np.random.default_rng(seed).permutation(len(names))[:top_k]
            scores = np.sort(np.random.default_rng(seed + 1).dirichlet(np.ones(len(order)) * 0.5))[::-1]
            return [Candidate(names[i][0], names[i][1], float(s), names[i][2]) for i, s in zip(order, scores)], None
        if self.active == "bioclip":
            return self._identify_bioclip(frames, top_k)
        return self._identify_clip(frames, top_k)
