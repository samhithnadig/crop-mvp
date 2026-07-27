import os
import uuid
import shutil
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import pipeline

app = FastAPI(title="OpenShorts-Lite")

# Wide open for now since this is a personal MVP -- tighten this before sharing widely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. Fine for a single-process MVP; swap for Redis/DB once this
# needs to survive restarts or run across multiple server processes.
JOBS: dict[str, dict] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued" | "processing" | "done" | "error"
    error: Optional[str] = None
    clips: Optional[list] = None


def run_job(job_id: str, youtube_url: Optional[str], uploaded_path: Optional[str], max_clips: int):
    JOBS[job_id]["status"] = "processing"
    try:
        result = pipeline.process_job(job_id, youtube_url, uploaded_path, max_clips)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["clips"] = result["clips"]
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"{e}\n{traceback.format_exc()}"


@app.post("/process", response_model=JobStatus)
async def process(
    background_tasks: BackgroundTasks,
    youtube_url: Optional[str] = Form(None),
    max_clips: int = Form(5),
    file: Optional[UploadFile] = File(None),
):
    if not youtube_url and not file:
        raise HTTPException(400, "Provide either youtube_url or a file upload.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(pipeline.WORKDIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    uploaded_path = None
    if file:
        uploaded_path = os.path.join(job_dir, file.filename)
        with open(uploaded_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"job_id": job_id, "status": "queued", "error": None, "clips": None}
    background_tasks.add_task(run_job, job_id, youtube_url, uploaded_path, max_clips)

    return JOBS[job_id]


@app.get("/status/{job_id}", response_model=JobStatus)
async def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job_id")
    return JOBS[job_id]


@app.get("/clip/{job_id}/{clip_index}")
async def get_clip(job_id: str, clip_index: int):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Clip not ready or job not found")
    clips = job["clips"]
    if clip_index < 0 or clip_index >= len(clips):
        raise HTTPException(404, "Clip index out of range")
    path = clips[clip_index]["file"]
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))


@app.get("/health")
async def health():
    return {"status": "ok"}
