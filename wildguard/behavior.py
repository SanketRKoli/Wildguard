"""Behaviour + 'is there an animal?' analysis with a general CLIP model (zero-shot, no training needed)."""
import os
import threading
import numpy as np

# name -> (phrase used in the prompt, arousal weight 0..1)
BEHAVIOURS = {
    "resting":    ("resting or sleeping", 0.00),
    "grooming":   ("grooming itself", 0.05),
    "drinking":   ("drinking water", 0.10),
    "feeding":    ("eating or feeding", 0.10),
    "with young": ("with its young", 0.20),
    "walking":    ("walking calmly", 0.25),
    "playing":    ("playing", 0.30),
    "swimming":   ("swimming", 0.30),
    "flying":     ("flying", 0.35),
    "calling":    ("calling with its mouth open", 0.55),
    "alert":      ("standing alert and watching", 0.60),
    "stalking":   ("stalking or hunting prey", 0.60),
    "fleeing":    ("running away fast", 0.90),
    "aggressive": ("showing aggression and baring its teeth", 0.90),
}
BEHAVIOUR_NAMES = list(BEHAVIOURS)
AROUSAL = np.array([BEHAVIOURS[b][1] for b in BEHAVIOUR_NAMES], dtype="float32")

ANIMAL_PROMPTS = ["a photo of an animal", "a photo of a wild animal", "a photo of a bird", "a photo of a fish",
                  "a photo of an insect", "a photo of a reptile", "a photo of a mammal in nature"]
OTHER_PROMPTS = ["a photo of a person", "a photo of a group of people", "a photo of a car", "a photo of a building",
                 "a photo of a landscape with no animals", "a photo of food", "a photo of a plant or flower",
                 "a cartoon or drawing", "a screenshot or text", "a photo of furniture or an object"]

CLIP_NAME = os.environ.get("WG_CLIP", "openai/clip-vit-base-patch32")


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class BehaviourModel:
    def __init__(self, fake=False):
        self.fake = fake
        self._lock = threading.Lock()
        self._model = None
        self._proc = None

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from transformers import CLIPModel, CLIPProcessor
                    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                    self._model = CLIPModel.from_pretrained(CLIP_NAME).eval()
                    self._proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    def _logits(self, frames, prompts):
        import torch
        self._load()
        inputs = self._proc(text=prompts, images=list(frames), return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.logits_per_image.cpu().numpy()          # (n_frames, n_prompts)

    @staticmethod
    def _fake_rows(frames, n, salt):
        rows = []
        for f in frames:
            seed = int(np.asarray(f.resize((8, 8))).mean() * 1000) + salt
            rows.append(np.random.default_rng(seed).dirichlet(np.ones(n) * 0.6))
        return np.stack(rows)

    def behaviour(self, frames, name):
        """probabilities (n_frames, n_behaviours) that sum to 1 per frame."""
        if self.fake:
            return self._fake_rows(frames, len(BEHAVIOUR_NAMES), 7)
        subject = name.strip() or "animal"
        article = "an" if subject[:1].lower() in "aeiou" else "a"
        prompts = [f"a photo of {article} {subject}, {BEHAVIOURS[b][0]}" for b in BEHAVIOUR_NAMES]
        return _softmax(self._logits(frames, prompts))

    def common_animal(self, frames, k=3):
        """CLIP guess among well-known animals (pets, farm animals, zoo animals) -> [(name, prob), ...]"""
        from .animals_list import COMMON_ANIMALS
        if self.fake:
            return [("dog", 0.9)]
        prompts = [f"a photo of a {n}" for n in COMMON_ANIMALS]
        p = _softmax(self._logits(frames, prompts)).mean(axis=0)
        idx = np.argsort(-p)[:k]
        return [(COMMON_ANIMALS[i], float(p[i])) for i in idx]

    def animal_presence(self, frames):
        """probability (n_frames,) that the frame contains an animal."""
        if self.fake:
            return np.full(len(frames), 0.95, dtype="float32")
        prompts = ANIMAL_PROMPTS + OTHER_PROMPTS
        p = _softmax(self._logits(frames, prompts))
        return p[:, :len(ANIMAL_PROMPTS)].sum(axis=1)
