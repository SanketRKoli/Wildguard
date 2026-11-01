---
title: WildGuard
emoji: 🐾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 🐾 WildGuard - worldwide animal identification and behaviour analysis

Open one link, upload a **photo or a short video** of almost any animal and get:

- **Species** (common + scientific name, top-3 guesses) - recognised by **BioCLIP**, a model trained on millions of images covering ~450,000 species
- **Conservation status** from the IUCN Red List (Endangered, Vulnerable, Least Concern ...) and a short description with photo from Wikipedia
- **Behaviour** (resting, feeding, alert, fleeing, aggressive, playing ...) using zero-shot vision-language analysis
- For videos: a **timeline** of behaviour and **movement**, an arousal / stress index, and a CSV report

Everything runs in the cloud, so it works on any phone or laptop - nothing to install.

## How it works
`photo/video -> frames -> (animal present?) -> BioCLIP species -> Wikipedia + IUCN lookup -> CLIP behaviour per frame -> + movement -> stress index`

Behaviour and stress are **estimates** (an arousal weight per behaviour + movement energy), not a veterinary diagnosis.

## Run / develop
```bash
# light mode for any laptop or GitHub Codespaces (fake models, no big downloads)
pip install -r requirements-dev.txt
WG_FAKE=1 WG_OFFLINE=1 pytest -q
WG_FAKE=1 python app.py

# real models (needs ~4 GB RAM, downloads ~2 GB on first run)
pip install -r requirements.txt gradio==5.50.0
python app.py
```

## Deploy: one public link (free)
1. Create a free account on huggingface.co and a new **Space** (SDK: Gradio, hardware: CPU basic).
2. Push this repo to GitHub, then add to the GitHub repo: secret `HF_TOKEN` (a *write* token) and variables `HF_USERNAME`, `HF_SPACE`.
3. Every `git push` to `main` runs `.github/workflows/sync-to-hf.yml` and updates the Space. Your link: `https://huggingface.co/spaces/<HF_USERNAME>/<HF_SPACE>`.

## Environment switches
`WG_FAKE=1` fake models (tests) - `WG_OFFLINE=1` skip web lookups - `WG_BACKEND=clip` skip BioCLIP and use the small fallback.
