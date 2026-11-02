"""WildGuard - worldwide animal identification + behaviour analysis. Run:  python app.py"""
import html
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd

from wildguard import knowledge, pipeline
from wildguard.behavior import AROUSAL, BEHAVIOUR_NAMES

try:                                   # Hugging Face ZeroGPU Spaces refuse to start unless one @spaces.GPU function exists.
    import spaces                      # This placeholder is never called, so analyses use the CPU and NO GPU quota.

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():
        return None
except Exception:                      # normal computers / CPU Spaces: nothing to do
    pass


def warm_up():
    """Download + load the models in the background right after start, so the first user does not wait."""
    if os.environ.get("WG_FAKE") == "1":
        return
    try:
        pipeline.RECOGNIZER._load()
        pipeline.BEHAVIOUR._load()
        print("[WildGuard] models ready:", pipeline.RECOGNIZER.active)
    except Exception as e:
        print("[WildGuard] warm-up failed (will retry on first request):", repr(e))

HEADER = """
# 🐾 WildGuard
**Identify almost any animal in the world and read its behaviour - from a photo or a short video.**
Upload a picture or a clip, press *Analyse*, and get the species, its conservation status, what it is doing and how calm or stressed it looks.
"""
FOOTER = ("<small>Species: BioCLIP (trained on millions of images, ~450,000 taxa) - Behaviour: zero-shot CLIP + movement analysis - "
          "Conservation data: IUCN Red List via GBIF / Wikidata, text from Wikipedia. "
          "The behaviour and stress values are an <b>estimate</b>, not a veterinary diagnosis.</small>")
CSS = """
.wg-card, .wg-card * { color: #1b2a1b !important; }
.wg-card .wg-muted { color: #555555 !important; }
.wg-card a { color: #1565c0 !important; }
"""
EMPTY_SPECIES = pd.DataFrame(columns=["Species", "Scientific name", "Confidence %"])


def _bar(value, colour):
    return (f"<div style='background:#e0e0e0;border-radius:8px;height:14px;width:100%'>"
            f"<div style='background:{colour};width:{max(2, min(100, value)):.0f}%;height:14px;border-radius:8px'></div></div>")


def summary_html(a):
    if not a.animal_present:
        return f"<div class='wg-card' style='padding:16px;border-radius:12px;background:#fff3e0'><h3>🔍 No animal found</h3><p>{html.escape(a.message)}</p></div>"
    best = a.candidates[0] if a.candidates else None
    wiki, status = a.info.get("wiki", {}), a.info.get("status", {})
    if a.uncertain:
        title, sci, conf = "Not sure which animal this is", "", ""
        guess = (f" Best guess: {html.escape(best.display)} ({best.score * 100:.0f}%)." if best else "")
        higher = html.escape(a.higher_rank) if a.higher_rank else "unknown"
        badge = "<span class='wg-muted'>Conservation status not shown because the species is uncertain</span>"
        note = (f"<p><i>The model is not confident about the exact species.{guess} Closest group: {higher}. "
                f"Try a clearer, closer picture where the animal fills the frame.</i></p>")
    else:
        title = html.escape(best.display) if best else "Unknown animal"
        sci = html.escape(best.scientific) if best and best.scientific else ""
        conf = f"{best.score * 100:.0f}% sure" if best else ""
        if status:
            colour = knowledge.STATUS_COLOURS.get(status["code"], "#555")
            badge = (f"<span class='wg-badge' style='background:{colour};color:#fff !important;padding:3px 12px;border-radius:999px;font-weight:700'>"
                     f"{html.escape(status['label'])}</span> <small class='wg-muted'>{html.escape(status['source'])}</small>")
        else:
            badge = "<span class='wg-muted'>Conservation status not available</span>"
        note = ""
        if a.source == "clip":
            note += "<p class='wg-muted'><i>Recognised by the general model (common / domestic animal).</i></p>"
        if a.higher_rank and best and best.score < 0.25:
            note += f"<p><i>Not fully sure about the exact species - closest group: {html.escape(a.higher_rank)}</i></p>"
    lvl_colour = {"Calm": "#2e7d32", "Alert": "#e08e00"}.get(a.level, "#c62828")
    img = (f"<img src='{html.escape(wiki['image'])}' style='width:200px;border-radius:10px;margin-right:16px;float:left'>"
           if wiki.get("image") and not a.uncertain else "")
    extract = html.escape((wiki.get("extract") or "")[:650]) if not a.uncertain else ""
    more = f" <a href='{html.escape(wiki['url'])}' target='_blank'>Read more</a>" if wiki.get("url") and not a.uncertain else ""
    beh = ", ".join(f"<b>{html.escape(n)}</b> ({p * 100:.0f}%)" for n, p in a.top)
    mov = f" &middot; movement {a.motion_mean * 100:.0f}%" if a.motion_mean is not None else ""
    secs = f" &middot; {a.seconds:.0f}s clip" if a.seconds else ""
    return f"""
<div class='wg-card' style='padding:18px;border-radius:14px;background:#f1f8e9;overflow:auto'>
  {img}
  <h2 style='margin:0'>{title} <small class='wg-muted'>{conf}</small></h2>
  <div class='wg-muted'><i>{sci}</i></div>
  <p>{badge}</p>{note}
  <p>{extract}{more}</p>
  <h3 style='margin-bottom:4px'>Behaviour</h3>
  <p>Mostly: {beh}{mov}{secs}</p>
  <b style='color:{lvl_colour} !important'>{html.escape(a.level)}</b> - arousal / stress index {a.stress_mean:.0f}/100 (peak {a.stress_peak:.0f})
  {_bar(a.stress_mean, lvl_colour)}
  <p>{html.escape(a.advice)}</p>
</div>"""


def behaviour_figure(a):
    names = list(a.behaviour_mean)
    vals = np.array([a.behaviour_mean[n] for n in names]) * 100
    order = np.argsort(vals)
    colours = plt.cm.RdYlGn_r(AROUSAL[[BEHAVIOUR_NAMES.index(names[i]) for i in order]])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh([names[i] for i in order], vals[order], color=colours)
    ax.set_xlabel("likelihood %"); ax.set_title("What is the animal doing?  (green calm - red agitated)")
    ax.set_xlim(0, max(100, vals.max() * 1.05)); fig.tight_layout()
    return fig


def timeline_figure(a):
    t = a.timeline
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axhspan(0, 30, color="#c8e6c9", alpha=.5); ax.axhspan(30, 60, color="#fff9c4", alpha=.6); ax.axhspan(60, 100, color="#ffcdd2", alpha=.5)
    ax.plot(t["time_s"], t["stress_index"], lw=2.5, color="#c62828", label="stress index")
    if "movement" in t:
        ax.plot(t["time_s"], t["movement"] * 100, lw=1.5, ls="--", color="#1565c0", label="movement %")
    ax.set_ylim(0, 100); ax.set_xlabel("time (s)"); ax.set_title("Behaviour over time"); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def render(a):
    summary = summary_html(a)
    if not a.animal_present:
        return summary, EMPTY_SPECIES, None, None, None
    species = pd.DataFrame([{"Species": c.display, "Scientific name": c.scientific, "Confidence %": round(c.score * 100, 1)}
                            for c in a.candidates]) if a.candidates else EMPTY_SPECIES
    path = os.path.join(tempfile.mkdtemp(), "wildguard_report.csv")
    a.timeline.to_csv(path, index=False)
    return summary, species, behaviour_figure(a), (timeline_figure(a) if a.kind == "video" else None), path


def run_image(img):
    if img is None:
        raise gr.Error("Please upload or capture a photo first.")
    return render(pipeline.analyse_image(img))


def run_video(path):
    if not path:
        raise gr.Error("Please upload or record a video first.")
    try:
        return render(pipeline.analyse_video(path))
    except ValueError as e:
        raise gr.Error(str(e))


with gr.Blocks(title="WildGuard - animal identification and behaviour", theme=gr.themes.Soft(primary_hue="green"), css=CSS) as demo:
    gr.Markdown(HEADER)
    with gr.Tabs():
        with gr.Tab("📷 Photo"):
            img_in = gr.Image(type="pil", label="Animal photo", sources=["upload", "webcam", "clipboard"], height=320)
            btn_img = gr.Button("Analyse photo", variant="primary")
        with gr.Tab("🎬 Video (up to ~30 s)"):
            vid_in = gr.Video(label="Animal video", sources=["upload", "webcam"], height=320)
            btn_vid = gr.Button("Analyse video", variant="primary")
    out_sum = gr.HTML()
    with gr.Row():
        out_beh = gr.Plot(label="Behaviour")
        out_time = gr.Plot(label="Timeline (video)")
    out_tab = gr.Dataframe(headers=list(EMPTY_SPECIES.columns), label="Top species guesses", interactive=False)
    out_csv = gr.File(label="Download report (CSV)")
    gr.HTML(FOOTER)
    outs = [out_sum, out_tab, out_beh, out_time, out_csv]
    btn_img.click(run_image, img_in, outs)
    btn_vid.click(run_video, vid_in, outs)

demo.queue(max_size=10, default_concurrency_limit=1)

import threading
threading.Thread(target=warm_up, daemon=True).start()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
