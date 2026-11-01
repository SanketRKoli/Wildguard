import os
os.environ["WG_FAKE"] = "1"
os.environ["WG_OFFLINE"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import pytest
from PIL import Image

from wildguard import knowledge, pipeline, scoring
from wildguard.behavior import BEHAVIOUR_NAMES


def make_video(path, speed):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for k in range(100):
        f = np.full((240, 320, 3), 90, "uint8")
        x = int((k * speed) % 280)
        cv2.rectangle(f, (x, 80), (x + 60, 160), (40, 200, 40), -1)
        vw.write(f)
    vw.release()


def test_status_names():
    assert knowledge.normalise_status("LEAST_CONCERN") == "LC"
    assert knowledge.normalise_status("endangered species") == "EN"
    assert knowledge.normalise_status("critically endangered") == "CR"
    assert knowledge.normalise_status("Near Threatened") == "NT"
    assert knowledge.normalise_status("vulnerable species") == "VU"
    assert knowledge.normalise_status("extinct in the wild") == "EW"
    assert knowledge.normalise_status("nonsense") is None


def test_scoring():
    calm = np.zeros((1, len(BEHAVIOUR_NAMES))); calm[0, BEHAVIOUR_NAMES.index("resting")] = 1
    wild = np.zeros((1, len(BEHAVIOUR_NAMES))); wild[0, BEHAVIOUR_NAMES.index("fleeing")] = 1
    assert scoring.level(scoring.stress_series(calm)[0]) == "Calm"
    assert scoring.level(scoring.stress_series(wild)[0]) == "Agitated / stressed"
    assert scoring.stress_series(wild, np.array([1.0]))[0] > scoring.stress_series(wild, np.array([0.0]))[0]


def test_image_pipeline():
    img = Image.fromarray((np.random.rand(240, 320, 3) * 255).astype("uint8"))
    a = pipeline.analyse_image(img)
    assert a.animal_present and a.candidates and len(a.timeline) == 1
    assert abs(sum(a.behaviour_mean.values()) - 1) < 1e-3


def test_video_pipeline_movement(tmp_path):
    slow, fast = str(tmp_path / "s.mp4"), str(tmp_path / "f.mp4")
    make_video(slow, 0); make_video(fast, 14)
    a_slow, a_fast = pipeline.analyse_video(slow), pipeline.analyse_video(fast)
    assert a_slow.motion_mean < 0.05 < a_fast.motion_mean
    assert len(a_fast.timeline) >= 8 and "movement" in a_fast.timeline


def test_no_animal(monkeypatch):
    monkeypatch.setattr(pipeline.BEHAVIOUR, "animal_presence", lambda frames: np.array([0.1]))
    img = Image.fromarray(np.zeros((64, 64, 3), "uint8"))
    a = pipeline.analyse_image(img)
    assert not a.animal_present and "No animal" in a.message


def test_app_renders(tmp_path):
    import app
    img = Image.fromarray((np.random.rand(120, 160, 3) * 255).astype("uint8"))
    summary, species, fig1, fig2, csv = app.run_image(img)
    assert "Behaviour" in summary and len(species) >= 1 and fig1 is not None and fig2 is None and os.path.exists(csv)
    vid = str(tmp_path / "v.mp4"); make_video(vid, 8)
    summary, species, fig1, fig2, csv = app.run_video(vid)
    assert fig2 is not None and os.path.getsize(csv) > 50
    with pytest.raises(Exception):
        app.run_image(None)


def _frames():
    return Image.fromarray((np.random.rand(96, 96, 3) * 255).astype("uint8"))


def test_low_confidence_uses_common_animal(monkeypatch):
    from wildguard.recognizer import Candidate
    monkeypatch.setattr(pipeline.RECOGNIZER, "identify",
                        lambda frames, top_k=3: ([Candidate("Demodex folliculorum", "", 0.01)], None))
    monkeypatch.setattr(pipeline.BEHAVIOUR, "common_animal", lambda frames, k=3: [("dog", 0.8)])
    a = pipeline.analyse_image(_frames())
    assert not a.uncertain and a.source == "clip" and a.candidates[0].common == "Dog"


def test_low_confidence_and_unknown_is_uncertain(monkeypatch):
    from wildguard.recognizer import Candidate
    monkeypatch.setattr(pipeline.RECOGNIZER, "identify",
                        lambda frames, top_k=3: ([Candidate("Demodex folliculorum", "", 0.01)], "Canidae family (score 0.50)"))
    monkeypatch.setattr(pipeline.BEHAVIOUR, "common_animal", lambda frames, k=3: [("dog", 0.2)])
    a = pipeline.analyse_image(_frames())
    assert a.uncertain and a.info == {}
    import app
    assert "Not sure which animal" in app.summary_html(a) and "Canidae" in app.summary_html(a)


def test_summary_text_is_readable():
    import app
    a = pipeline.analyse_image(_frames())
    h = app.summary_html(a)
    assert "wg-card" in h and ".wg-card *" in app.CSS
