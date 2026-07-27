# OpenShorts-Lite

A minimal, self-hostable version of the "paste a video, get AI-picked short
clips, auto-cropped" pipeline. Built to run on your own machine first.

## What it does

1. Takes a YouTube URL (via `yt-dlp`) or an uploaded video file
2. Transcribes it locally with `faster-whisper` (free, no API key needed)
3. Sends the transcript to Gemini (free tier) to pick the best 15-60s moments
4. Face-tracks and crops each moment to 9:16 vertical using OpenCV + MediaPipe + ffmpeg
5. Serves the resulting clips back over a simple HTTP API

## Before you run anything: read this

- **YouTube downloads via `yt-dlp` are a grey area under YouTube's Terms of
  Service.** This is genuinely how every open-source tool in this space
  (OpenShorts, AI-Youtube-Shorts-Generator, etc.) works — but the risk sits
  with whoever runs the server, especially if other people start using it or
  you start charging for it. Treat this as a personal testing tool for now.
- This needs **real compute** — transcription and video processing are CPU/GPU
  heavy. Your laptop can run this for testing; a real product needs a paid
  server eventually (Railway, Render, a VPS, etc. — all typically require a
  card once you're on a real plan).

## Setup

### 1. Install system dependencies
You need **ffmpeg** installed and on your PATH.
- Mac: `brew install ffmpeg`
- Windows: download from ffmpeg.org and add to PATH, or `choco install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### 2. Install Python dependencies
Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get a free Gemini API key
Go to https://aistudio.google.com/apikey — sign in with a Google account,
no card required for the free tier (has generous daily limits for testing).

Set it as an environment variable:
```bash
export GEMINI_API_KEY="your-key-here"   # Windows: set GEMINI_API_KEY=your-key-here
```

### 4. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

You should see it running at http://localhost:8000 — check http://localhost:8000/health

## Using it

Submit a job (YouTube link):
```bash
curl -X POST http://localhost:8000/process -F "youtube_url=https://youtube.com/watch?v=..."
```

Or with a file:
```bash
curl -X POST http://localhost:8000/process -F "file=@myvideo.mp4"
```

This returns a `job_id`. Check progress:
```bash
curl http://localhost:8000/status/<job_id>
```

Once `status` is `"done"`, download a clip:
```bash
curl http://localhost:8000/clip/<job_id>/0 -o clip1.mp4
```

## Known v1 limitations (worth knowing, not necessarily fixing today)

- **Crop is a single fixed position per clip**, chosen from the median face
  position across the segment — it doesn't dynamically follow movement
  frame-by-frame like the browser tool's live tracking did. Good enough for
  someone mostly stationary (talking-head content); less good for a subject
  moving a lot within a clip.
- **First run will be slow** — `faster-whisper`'s model downloads on first use,
  and CPU transcription of a long video can take a few minutes.
- **No auth, no rate limiting, no cleanup of old job files** — fine for
  testing on your own machine, not fine for a public server yet.
- **In-memory job store** — restarting the server loses job status (not the
  files themselves, those stay on disk in `./jobs`).

## Suggested next steps once this runs locally

1. Test it end-to-end on one of your own short videos first
2. If the AI highlight selection feels off, tweak the prompt in
   `app/pipeline.py` (`select_highlights`) — this is the part worth iterating
   on most, since it's the actual value-add over a plain crop tool
3. Only then think about hosting it somewhere real
