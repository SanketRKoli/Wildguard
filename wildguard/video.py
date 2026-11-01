"""Read a photo or a short video as a few frames + a movement signal."""
import cv2
import numpy as np
from PIL import Image

MAX_SECONDS = 30
N_FRAMES = 12
MOTION_STEP_FPS = 5
MOTION_NORM = 0.08          # share of changed pixels that counts as 'maximum movement'


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _shrink(bgr, width=640):
    h, w = bgr.shape[:2]
    if w <= width:
        return bgr
    return cv2.resize(bgr, (width, int(h * width / w)))


def read_video(path, max_seconds=MAX_SECONDS, n_frames=N_FRAMES):
    """returns dict(frames=[PIL], times=[s], motion=[0..1 per frame], fps, seconds)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open the video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(fps * max_seconds)
    last = min(total, limit) if total > 0 else limit
    chosen = set(np.linspace(0, max(last - 1, 0), num=min(n_frames, max(last, 1))).astype(int).tolist())
    stride = max(1, int(round(fps / MOTION_STEP_FPS)))

    frames, times, motion_t, motion_v = [], [], [], []
    prev, i = None, 0
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            g = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
            if prev is not None:
                motion_t.append(i / fps)
                motion_v.append(float(np.mean(np.abs(g - prev) > 0.10)))
            prev = g
        if i in chosen:
            frames.append(to_pil(_shrink(frame)))
            times.append(i / fps)
        i += 1
    cap.release()
    if not frames:
        raise ValueError("The video has no readable frames")
    if motion_v:
        motion = np.interp(times, motion_t, motion_v)
    else:
        motion = np.zeros(len(times))
    motion = np.clip(np.asarray(motion) / MOTION_NORM, 0.0, 1.0)
    return {"frames": frames, "times": times, "motion": motion, "fps": fps, "seconds": i / fps}
