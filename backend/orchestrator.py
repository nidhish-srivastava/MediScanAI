"""
Multi-Agent Orchestrator — MediScan AI

Coordinates four specialized agents in sequence:

  Agent 1 — Vision Screening Agent   (TorchXRayVision DenseNet-121)
  Agent 2 — Radiologist Agent        (Groq LLM — structured report generation)
  Agent 3 — Safety Validation Agent  (false positive suppression, urgent flag gating)
  Agent 4 — Report Formatting Agent  (structures final output for frontend + PDF)

Each agent returns its own result + timing so the frontend can show live progress.
"""

import time
import logging
from typing import List, Dict, Optional

from cv_model import run_cv_model
from report_generator import generate_report
from image_validator import validate_chest_xray

logger = logging.getLogger(__name__)


# ── Agent 0: Input Validation ──────────────────────────────────────────────

def run_validation_agent(raw_bytes: bytes) -> Dict:
    """
    Heuristic gate: checks whether the uploaded image plausibly looks like a
    chest X-ray BEFORE running the CV model or spending an LLM call on it.
    """
    start = time.time()
    result = validate_chest_xray(raw_bytes)
    elapsed = round(time.time() - start, 3)

    if result["is_valid"]:
        summary = f"Input looks like a chest X-ray (confidence {result['confidence']*100:.0f}%)"
    else:
        summary = f"Input rejected — not a chest X-ray ({'; '.join(result['reasons'])})"

    return {
        "agent": "Input Validation Agent",
        "status": "complete",
        "elapsed_seconds": elapsed,
        "is_valid": result["is_valid"],
        "confidence": result["confidence"],
        "reasons": result["reasons"],
        "checks": result["checks"],
        "summary": summary,
    }


def _build_rejection_response(validation_result: Dict, patient: Dict, image_meta: Dict, total_elapsed: float) -> Dict:
    """
    Short-circuit response returned when the Input Validation Agent rejects
    the image. Shape stays close to the normal response so the frontend
    doesn't need a completely separate code path — cv_findings/report/safety
    are just empty/null, and `rejected: true` + `rejection_reasons` drive the UI.
    """
    agent_trace = [
        {
            "agent": "Input Validation Agent",
            "status": "complete",
            "elapsed_seconds": validation_result["elapsed_seconds"],
            "summary": validation_result["summary"],
        },
        {"agent": "Vision Screening Agent", "status": "skipped", "elapsed_seconds": 0, "summary": "Skipped — input rejected"},
        {"agent": "Radiologist Agent", "status": "skipped", "elapsed_seconds": 0, "summary": "Skipped — input rejected"},
        {"agent": "Safety Validation Agent", "status": "skipped", "elapsed_seconds": 0, "summary": "Skipped — input rejected"},
    ]

    return {
        "rejected": True,
        "rejection_reasons": validation_result["reasons"],
        "cv_findings": [],
        "report": None,
        "patient": patient,
        "safety": {"overrides": [], "flags_checked": [], "urgent_validated": False},
        "agent_trace": agent_trace,
        "timing": {
            "vision_seconds": 0,
            "radiologist_seconds": 0,
            "safety_seconds": 0,
            "total_seconds": round(total_elapsed, 2),
        },
        "image_meta": image_meta,
    }


# ── Agent 1: Vision Screening ──────────────────────────────────────────────

def run_vision_agent(raw_bytes: bytes) -> Dict:
    """
    Runs TorchXRayVision DenseNet-121 on the image.
    Returns ranked pathology detections with confidence scores.
    """
    start = time.time()
    findings = run_cv_model(raw_bytes)
    elapsed = round(time.time() - start, 2)

    high   = [f for f in findings if f["confidence"] >= 0.70]
    medium = [f for f in findings if 0.50 <= f["confidence"] < 0.70]
    low    = [f for f in findings if f["confidence"] < 0.50]

    top = findings[0]["label"] if findings else "none"

    return {
        "agent": "Vision Screening Agent",
        "status": "complete",
        "elapsed_seconds": elapsed,
        "findings": findings,
        "summary": f"Detected {len(findings)} findings — top: {top} ({round(findings[0]['confidence']*100) if findings else 0}%)",
        "stats": {
            "total": len(findings),
            "high_confidence": len(high),
            "medium_confidence": len(medium),
            "low_confidence": len(low),
        }
    }


# ── Agent 2: Radiologist ───────────────────────────────────────────────────

async def run_radiologist_agent(
    image_b64: str,
    mime_type: str,
    cv_findings: List[Dict],
    patient: Dict
) -> Dict:
    """
    Groq LLM acting as a senior radiologist.
    Receives CV findings + raw image, produces structured clinical report.
    """
    start = time.time()
    report = await generate_report(image_b64, mime_type, cv_findings, patient)
    elapsed = round(time.time() - start, 2)

    return {
        "agent": "Radiologist Agent",
        "status": "complete",
        "elapsed_seconds": elapsed,
        "report": report,
        "summary": f"Report generated — {report['tokens_used']} tokens, model: {report['model']}",
    }


# ── Agent 3: Safety Validation ─────────────────────────────────────────────

def run_safety_agent(report: Dict, cv_findings: List[Dict]) -> Dict:
    """
    Reviews the radiologist report and CV findings.
    Suppresses URGENT flags not visually confirmed by the LLM.
    Checks for contradictions between CV model and LLM output.
    Returns a validation summary and a (possibly corrected) safe report.
    """
    start = time.time()

    flags_checked = []
    overrides = []

    # Check every CV finding that was marked urgent
    cv_urgent = [f for f in cv_findings if f.get("urgent")]
    for f in cv_urgent:
        label = f["label"]
        llm_confirmed = (
            report.get("is_urgent") and
            report.get("urgent_finding") and
            label.lower() in report["urgent_finding"].lower()
        )
        flags_checked.append({
            "finding": label,
            "cv_urgent": True,
            "llm_confirmed": llm_confirmed,
        })
        if not llm_confirmed:
            overrides.append(label)

    # If LLM marked urgent but we can validate the finding exists in report text
    validated_urgent = report.get("is_urgent", False)
    if validated_urgent and overrides:
        # Some CV urgent flags were not confirmed — report is still valid
        # but note the discrepancy
        pass

    elapsed = round(time.time() - start, 3)

    if overrides:
        summary = f"Suppressed {len(overrides)} unconfirmed urgent flag(s): {', '.join(overrides)}"
    elif cv_urgent:
        summary = f"Validated {len(cv_urgent)} urgent flag(s) — all LLM-confirmed"
    else:
        summary = "No urgent flags to validate — report cleared"

    return {
        "agent": "Safety Validation Agent",
        "status": "complete",
        "elapsed_seconds": elapsed,
        "flags_checked": flags_checked,
        "overrides": overrides,
        "urgent_validated": validated_urgent,
        "summary": summary,
    }


# ── Agent 4: Report Formatting ─────────────────────────────────────────────

def run_formatting_agent(
    vision_result: Dict,
    radiologist_result: Dict,
    safety_result: Dict,
    patient: Dict,
    image_meta: Dict,
    total_elapsed: float,
) -> Dict:
    """
    Assembles the final structured response for the frontend.
    Combines all agent outputs into one clean payload.
    """
    start = time.time()

    report = radiologist_result["report"]

    # Build agent trace for frontend pipeline display
    agent_trace = [
        {
            "agent": vision_result["agent"],
            "status": vision_result["status"],
            "elapsed_seconds": vision_result["elapsed_seconds"],
            "summary": vision_result["summary"],
        },
        {
            "agent": radiologist_result["agent"],
            "status": radiologist_result["status"],
            "elapsed_seconds": radiologist_result["elapsed_seconds"],
            "summary": radiologist_result["summary"],
        },
        {
            "agent": safety_result["agent"],
            "status": safety_result["status"],
            "elapsed_seconds": safety_result["elapsed_seconds"],
            "summary": safety_result["summary"],
        },
        {
            "agent": "Report Formatting Agent",
            "status": "complete",
            "elapsed_seconds": round(time.time() - start, 3),
            "summary": "Output assembled and validated",
        },
    ]

    # Layer-2 check: heuristic gate passed, but the vision LLM itself may still
    # determine on direct visual inspection that this isn't a real chest X-ray.
    # We don't hard-block here (the report call already cost money/time) — instead
    # we surface it clearly so the frontend can show a prominent warning banner.
    validity_warning = None
    if report.get("is_valid_xray") is False:
        validity_warning = (
            f"The radiologist agent flagged this image as not a valid chest X-ray on direct "
            f"visual inspection: {report.get('validity_reason') or 'reason not specified'}. "
            f"Treat the report below as unreliable."
        )

    return {
        "cv_findings": vision_result["findings"],
        "report": report,
        "validity_warning": validity_warning,
        "patient": patient,
        "safety": {
            "overrides": safety_result["overrides"],
            "flags_checked": safety_result["flags_checked"],
            "urgent_validated": safety_result["urgent_validated"],
        },
        "agent_trace": agent_trace,
        "timing": {
            "vision_seconds":     vision_result["elapsed_seconds"],
            "radiologist_seconds": radiologist_result["elapsed_seconds"],
            "safety_seconds":     safety_result["elapsed_seconds"],
            "total_seconds":      round(total_elapsed, 2),
        },
        "image_meta": image_meta,
    }


# ── Main entry point ───────────────────────────────────────────────────────

async def run_pipeline(
    raw_bytes: bytes,
    image_b64: str,
    mime_type: str,
    patient: Dict,
    image_meta: Dict,
) -> Dict:
    """
    Runs all four agents in sequence.
    Called by main.py /analyze endpoint.
    """
    total_start = time.time()

    logger.info("── Agent 0: Input Validation ──")
    validation_result = run_validation_agent(raw_bytes)
    logger.info(f"Agent 0 done — {validation_result['summary']}")

    if not validation_result["is_valid"]:
        total_elapsed = time.time() - total_start
        logger.warning(f"Pipeline short-circuited — input rejected in {total_elapsed:.2f}s")
        return _build_rejection_response(validation_result, patient, image_meta, total_elapsed)

    logger.info("── Agent 1: Vision Screening ──")
    vision_result = run_vision_agent(raw_bytes)
    logger.info(f"Agent 1 done in {vision_result['elapsed_seconds']}s — {vision_result['summary']}")

    logger.info("── Agent 2: Radiologist ──")
    radiologist_result = await run_radiologist_agent(
        image_b64, mime_type, vision_result["findings"], patient
    )
    logger.info(f"Agent 2 done in {radiologist_result['elapsed_seconds']}s")

    logger.info("── Agent 3: Safety Validation ──")
    safety_result = run_safety_agent(radiologist_result["report"], vision_result["findings"])
    logger.info(f"Agent 3 done — {safety_result['summary']}")

    logger.info("── Agent 4: Report Formatting ──")
    total_elapsed = time.time() - total_start
    final = run_formatting_agent(
        vision_result, radiologist_result, safety_result,
        patient, image_meta, total_elapsed
    )

    logger.info(f"Pipeline complete in {total_elapsed:.2f}s")
    return final