import os
import base64
import logging
import time
from io import BytesIO
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Optional

from cv_model import _get_model
from orchestrator import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading CV model on startup...")
    model, loaded = _get_model()
    if loaded:
        logger.info("CV model loaded and ready")
    else:
        logger.warning("CV model not loaded — fallback mode active")
    yield


app = FastAPI(title="MediScan AI — Multi-Agent Radiology API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 15


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "architecture": "multi-agent"}


@app.post("/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    patient_age:  Optional[str] = Form(None),
    patient_sex:  Optional[str] = Form(None),
    ref_doctor:   Optional[str] = Form(None),
    history:      Optional[str] = Form(None),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    raw = await file.read()

    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    try:
        img = Image.open(BytesIO(raw))
        img.verify()
        img = Image.open(BytesIO(raw))
        width, height = img.size
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    logger.info(f"Starting multi-agent pipeline: {file.filename} | {width}x{height} | patient: {patient_name or 'anonymous'}")

    image_b64 = base64.b64encode(raw).decode("utf-8")

    patient = {
        "name":    patient_name,
        "age":     patient_age,
        "sex":     patient_sex,
        "ref_doc": ref_doctor,
        "history": history,
    }

    image_meta = {
        "filename": file.filename,
        "width":    width,
        "height":   height,
        "size_kb":  round(len(raw) / 1024, 1),
    }

    result = await run_pipeline(
        raw_bytes=raw,
        image_b64=image_b64,
        mime_type=file.content_type,
        patient=patient,
        image_meta=image_meta,
    )

    return JSONResponse(result)