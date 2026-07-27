"""
Core processing pipeline for OpenShorts-Lite.

Flow:
  1. download_video()      -> gets a local video file (YouTube URL or already-uploaded file)
  2. transcribe()          -> faster-whisper, runs fully locally, free
  3. select_highlights()   -> Gemini free-tier API picks the best moments from the transcript
  4. crop_segment()        -> ffmpeg cuts the segment, then OpenCV/MediaPipe face-tracks the crop

IMPORTANT LEGAL NOTE:
Downloading YouTube videos via yt-dlp is a grey area under YouTube's Terms of
Service, regardless of who runs it. This is the same approach every open-source
"auto-clip" tool (OpenShorts, AI-Youtube-Shorts-Generator, etc.) uses. You are
taking on that risk yourself if you deploy this for others to use, especially
once you're charging money for it. Consider this a personal/testing tool until
you've thought that trade-off through.
"""

import os
import json
import subprocess
import uuid
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import mediapipe as mp
import google.generativeai as genai
from faster_whisper import WhisperModel
import yt_dlp

WORKDIR = os.environ.get("OPENSHORTS_WORKDIR", "./jobs")
os.makedirs(WORKDIR, exist_ok=True)

# Configure Gemini once at import time. Get a free key at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Loaded once and reused across requests -- loading the model is the slow part.
_whisper_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        # "base" is a good speed/accuracy tradeoff for a first version.
        # Use "small" or "medium" if you have a GPU and want better accuracy.
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


@dataclass
class Highlight:
    start: float
    end: float
    title: str
    reason: str


def download_video(job_dir: str, youtube_url: Optional[str], uploaded_path: Optional[str]) -> str:
    """Returns the path to a local mp4 to process."""
    if uploaded_path:
        return uploaded_path

    if not youtube_url:
        raise ValueError("Provide either a youtube_url or an uploaded file.")

    out_path = os.path.join(job_dir, "source.mp4")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return out_path


def transcribe(video_path: str) -> list[dict]:
    """Returns a list of {start, end, text} segments."""
    model = get_whisper_model()
    segments, _info = model.transcribe(video_path, beam_size=5, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def select_highlights(transcript: list[dict], video_duration: float, max_clips: int = 5, api_key: Optional[str] = None) -> list[Highlight]:
    """Feeds the transcript to Gemini and asks it to pick the best short-form moments."""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise RuntimeError(
            "No Gemini API key provided. Get a free key (no card required) at "
            "https://aistudio.google.com/apikey and enter it in Settings, or set GEMINI_API_KEY."
        )
    genai.configure(api_key=key)

    transcript_text = "\n".join(f"[{t['start']:.1f}-{t['end']:.1f}] {t['text']}" for t in transcript)

    prompt = f"""You are selecting the best short-form clips from a video transcript for
social media (Reels/Shorts/TikTok). The full video is {video_duration:.0f} seconds long.

Pick up to {max_clips} non-overlapping segments, each between 15 and 60 seconds, that would
work best as standalone short clips -- look for hooks, punchlines, strong opinions, emotional
peaks, or self-contained stories/explanations.

Transcript (format is [start-end] text):
{transcript_text}

Respond ONLY with a JSON array, no markdown fences, no other text, in this exact format:
[{{"start": 12.4, "end": 45.0, "title": "short catchy title", "reason": "why this works as a clip"}}]
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    raw = response.text.strip()
    # Strip accidental markdown fences if the model adds them anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    data = json.loads(raw)
    return [Highlight(**item) for item in data]


def _detect_face_boxes(frame, face_detector):
    """Returns list of (x, y, w, h) in pixel coords for detected faces in a frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    boxes = []
    h, w, _ = frame.shape
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            boxes.append((x, y, bw, bh))
    return boxes


def crop_segment(source_path: str, start: float, end: float, out_path: str, target_ratio: float = 9 / 16):
    """
    Cuts [start, end] from source_path, tracks the largest face across sampled frames,
    and renders a smoothly-following vertical crop via ffmpeg.
    """
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_w / src_h > target_ratio:
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(src_w / target_ratio)

    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

    start_frame = int(start * fps)
    end_frame = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Sample every ~0.3s to find where the subject is, then smooth between samples.
    sample_stride = max(1, int(fps * 0.3))
    center_x = src_w / 2
    smoothed_cx = center_x
    crop_centers = []  # (frame_index, cx)

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % sample_stride == 0:
            boxes = _detect_face_boxes(frame, face_detector)
            if boxes:
                largest = max(boxes, key=lambda b: b[2] * b[3])
                target_cx = largest[0] + largest[2] / 2
                smoothed_cx = smoothed_cx + (target_cx - smoothed_cx) * 0.35
            crop_centers.append((frame_idx, smoothed_cx))
        frame_idx += 1

    cap.release()
    face_detector.close()

    if not crop_centers:
        crop_centers = [(start_frame, center_x)]

    # Use the median center as a simple, stable single crop-x for this segment.
    # (A per-frame dynamic crop is possible but needs a more complex ffmpeg filter graph
    # or a full frame-by-frame re-render -- this median approach is the practical v1.)
    xs = sorted(c[1] for c in crop_centers)
    median_cx = xs[len(xs) // 2]
    crop_x = int(min(max(median_cx - crop_w / 2, 0), src_w - crop_w))
    crop_y = int((src_h - crop_h) / 2)

    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", source_path, "-t", str(duration),
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def process_job(job_id: str, youtube_url: Optional[str], uploaded_path: Optional[str], max_clips: int = 5, api_key: Optional[str] = None) -> dict:
    job_dir = os.path.join(WORKDIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    source = download_video(job_dir, youtube_url, uploaded_path)

    cap = cv2.VideoCapture(source)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    transcript = transcribe(source)
    highlights = select_highlights(transcript, duration, max_clips=max_clips, api_key=api_key)

    clips = []
    for i, h in enumerate(highlights):
        out_path = os.path.join(job_dir, f"clip_{i+1}.mp4")
        crop_segment(source, h.start, h.end, out_path)
        clips.append({**asdict(h), "file": out_path})

    return {"job_id": job_id, "source": source, "clips": clips}
