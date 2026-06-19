"""
Report Generator — Groq (free tier) with improved prompt engineering

# ── CLAUDE USAGE (commented out — re-enable when budget allows) ──────────────
# Switch via LLM_PROVIDER env variable:
#   LLM_PROVIDER=groq    → uses Groq free tier (current)
#   LLM_PROVIDER=claude  → uses Claude claude-sonnet-4-6 (better quality, costs $)
#
# To re-enable Claude:
#   1. Uncomment the CLAUDE_* constants below
#   2. Uncomment _generate_claude() function
#   3. Uncomment the `if LLM_PROVIDER == "claude":` branch in generate_report()
#   4. Set ANTHROPIC_API_KEY in your .env
#   5. Change CLAUDE_MODEL to "claude-sonnet-4-6" (updated model name)
# ─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from typing import List, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# ── provider config ────────────────────────────────────────────
LLM_PROVIDER   = "groq"  # hardcoded; Claude commented out

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")

# ── CLAUDE CONFIG (commented out) ─────────────────────────────
# CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
# CLAUDE_MODEL   = "claude-sonnet-4-6"   # NOTE: updated from claude-sonnet-4-5
# CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# ──────────────────────────────────────────────────────────────


# ── IMPROVEMENT 1: Stronger false-positive suppression ────────
# Key changes vs original:
#   - Explicit "innocent until proven guilty" rule for URGENT flags
#   - CV model confidence < 70% must NOT drive URGENT on its own
#   - Pneumothorax requires a visible pleural line, not just lucency
#   - Pediatric thymus warning added (common false positive for mass/widened mediastinum)
#   - Explicit "if you are not sure, say so" instruction instead of forcing a finding
SYSTEM_PROMPT = """You are an expert radiologist with 20+ years of experience reading chest X-rays.
You produce structured, clinically accurate radiology reports used by doctors in clinical settings.
Your most important job is to NOT cause harm through false urgent alerts.

════════════════════════════════════════════════
CRITICAL RULES — follow every single one
════════════════════════════════════════════════

1. PATIENT TYPE DETECTION (do this first)
   - Assess bone density, rib spacing, body habitus, frame size
   - State explicitly: PEDIATRIC (under 18) or ADULT
   - PEDIATRIC normals differ from adults:
       • Normal CTR < 0.55
       • Thymus creates wide mediastinum in infants — this is NORMAL, not a mass
       • AP supine is the standard projection for neonates/infants
       • Suboptimal inspiration is common and inflates apparent cardiac size

2. PROJECTION (critical — affects everything)
   - PA: patient standing, scapulae outside lung fields
   - AP: patient supine, scapulae over lungs, heart MAGNIFIED
   - If monitoring equipment is present → it is AP. Always.
   - On AP films: NEVER diagnose cardiomegaly without a PA film for confirmation

3. MEDICAL DEVICES — identify ALL visible:
   - ETT: tip should be 2–3 cm above carina. Flag if < 1 cm or in mainstem bronchus
   - NG tube: confirm tip below diaphragm
   - Central line: confirm tip in SVC, not right atrium or ventricle
   - ECG leads, drains, catheters
   - A malpositioned ETT is a life-threatening emergency — flag immediately

4. TRACHEAL DEVIATION
   - Always state: midline / deviated LEFT / deviated RIGHT
   - Compare tracheal air column to spinous processes

5. ════ URGENT FLAG RULES — READ CAREFULLY ════
   The ⚠️ URGENT flag causes clinical teams to act immediately.
   A false URGENT can lead to unnecessary invasive procedures on sick patients.

   ONLY flag ⚠️ URGENT if ALL of the following are true:
     a) You can see the finding directly in the image (not just from CV model)
     b) The finding is unambiguous — you are confident, not just suspicious
     c) The finding poses immediate life threat if missed

   SPECIFIC RULES:
   - PNEUMOTHORAX: requires a VISIBLE pleural line separating lung from chest wall.
     A hyperlucent area alone is NOT sufficient — it may be rotation artifact,
     patient positioning, or soft tissue. If no pleural line → do NOT flag as urgent.
   - PLEURAL EFFUSION: only urgent if massive (> 2/3 hemithorax opacified)
   - CARDIOMEGALY: NEVER urgent on AP film alone
   - MEDIASTINAL WIDENING in pediatric patient: likely thymus — do NOT flag urgent
     unless tracheal deviation or clinical aortic injury context is present

6. CV MODEL GUIDANCE
   - The CV model assists you but does NOT decide findings
   - Confidence < 70%: treat as "possible, needs visual confirmation"
   - Confidence < 50%: treat as background noise — do NOT report unless you see it
   - Known high false-positive labels: Pneumothorax, Mass, Enlarged Cardiomediastinum
   - If CV model says URGENT but you cannot visually confirm → explicitly state
     "CV model flagged X but not visually confirmed on this image"

7. UNCERTAINTY IS ACCEPTABLE
   - If you cannot distinguish consolidation from atelectasis → say so
   - If image quality limits assessment → say so
   - Do not manufacture findings to fill report sections
   - "Cannot be assessed on this projection" is a valid and honest answer
════════════════════════════════════════════════
Always use proper radiology terminology. Be concise but clinically precise."""


def _build_user_prompt(cv_findings: List[Dict], patient: Dict) -> str:
    # Patient context block
    patient_ctx = ""
    if any(patient.values()):
        parts = []
        if patient.get("name"):    parts.append(f"Name: {patient['name']}")
        if patient.get("age"):     parts.append(f"Age: {patient['age']}")
        if patient.get("sex"):     parts.append(f"Sex: {patient['sex']}")
        if patient.get("ref_doc"): parts.append(f"Referring Doctor: {patient['ref_doc']}")
        if patient.get("history"): parts.append(f"Clinical History: {patient['history']}")
        patient_ctx = "PATIENT INFORMATION:\n" + "\n".join(parts) + "\n\n"

    # ── IMPROVEMENT 2: Separate high-confidence from low-confidence CV findings ──
    # Original prompt lumped all findings together, encouraging the LLM to
    # treat 52% confidence the same as 85% confidence.
    if cv_findings:
        high_conf = [f for f in cv_findings if f["confidence"] >= 0.70]
        mid_conf  = [f for f in cv_findings if 0.50 <= f["confidence"] < 0.70]
        low_conf  = [f for f in cv_findings if f["confidence"] < 0.50]

        def fmt(findings):
            return "\n".join(
                f"  - {f['label']}: {round(f['confidence'] * 100)}%"
                + (" ← VISUALLY VERIFY BEFORE FLAGGING URGENT" if f.get("urgent") else "")
                for f in findings
            )

        cv_ctx = "CV MODEL PRE-SCREENING (TorchXRayVision DenseNet-121):\n\n"

        if high_conf:
            cv_ctx += f"HIGH CONFIDENCE (≥70%) — likely real, still verify visually:\n{fmt(high_conf)}\n\n"
        if mid_conf:
            cv_ctx += f"MODERATE CONFIDENCE (50–69%) — possible findings, require clear visual confirmation:\n{fmt(mid_conf)}\n\n"
        if low_conf:
            cv_ctx += f"LOW CONFIDENCE (<50%) — likely noise, report ONLY if you independently see it:\n{fmt(low_conf)}\n\n"

        cv_ctx += (
            "IMPORTANT: Do not report a finding solely because the CV model detected it.\n"
            "For Pneumothorax specifically: only flag if you see a pleural line.\n\n"
        )
    else:
        cv_ctx = "No CV model pre-screening available. Examine the image independently.\n\n"

    # ── IMPROVEMENT 3: Add explicit "what not to do" examples in the prompt ──
    # LLMs respond well to negative examples alongside positive instructions.
    return f"""{patient_ctx}{cv_ctx}Produce a structured radiology report with EXACTLY these sections:

**PATIENT & TECHNICAL**
Patient type (PEDIATRIC/ADULT), projection (PA/AP — be definitive), image quality,
rotation, inspiration. List ALL medical devices with positions and flag any malpositioned.

**LUNG FIELDS**
Right and left separately. Opacities, consolidation, infiltrates, atelectasis,
hyperinflation. If you cannot distinguish consolidation from atelectasis, say so.

**CARDIAC SILHOUETTE**
CTR estimate. If AP: state "true size cannot be determined on AP projection."
For pediatric: note if thymus may be contributing to mediastinal width.

**MEDIASTINUM & HILUM**
Width. Tracheal position — EXPLICITLY state midline / deviated left / deviated right.
Hilar size bilaterally.

**PLEURAL SPACES**
Each costophrenic angle (sharp/blunted). Pleural effusion assessment per side.
Pneumothorax assessment — ONLY flag if pleural line is visible.

**BONY STRUCTURES**
Ribs, spine, clavicles, scapulae. Fractures, deformities, lesions.

**IMPRESSION**
Numbered list, most urgent first.
If CV model flagged something you cannot visually confirm, state:
  "X flagged by CV model — not visually confirmed, clinical correlation advised."
If normal: "No acute cardiopulmonary abnormality detected."

**RECOMMENDATIONS**
Specific next steps with urgency. Tailor to pediatric pathways if applicable.

WHAT NOT TO DO:
- Do not flag pneumothorax urgent without a visible pleural line
- Do not call cardiomegaly on an AP film
- Do not treat all CV model detections as equally reliable
- Do not fill sections with speculation to appear thorough"""


async def generate_report(
    image_b64: str,
    mime_type: str,
    cv_findings: List[Dict],
    patient: Optional[Dict] = None
) -> Dict:
    if patient is None:
        patient = {}

    user_prompt = _build_user_prompt(cv_findings, patient)
    return await _generate_groq(image_b64, mime_type, user_prompt)

    # ── CLAUDE CALL (commented out) ───────────────────────────
    # if LLM_PROVIDER == "claude":
    #     return await _generate_claude(image_b64, mime_type, user_prompt)
    # else:
    #     return await _generate_groq(image_b64, mime_type, user_prompt)
    # ──────────────────────────────────────────────────────────


async def _generate_groq(image_b64: str, mime_type: str, user_prompt: str) -> Dict:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set")

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 2200,        # IMPROVEMENT 4: increased from 1800 — reports were getting cut off
        "temperature": 0.10,       # IMPROVEMENT 5: lowered from 0.15 — less hallucination on edge cases
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload
        )

    if response.status_code != 200:
        logger.error(f"Groq error {response.status_code}: {response.text}")
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

    data = response.json()
    report_text = data["choices"][0]["message"]["content"]
    return _parse_report(report_text, GROQ_MODEL, data.get("usage", {}).get("completion_tokens", 0))


# ── CLAUDE FUNCTION (commented out) ───────────────────────────────────────────
# async def _generate_claude(image_b64: str, mime_type: str, user_prompt: str) -> Dict:
#     if not CLAUDE_API_KEY:
#         raise ValueError("ANTHROPIC_API_KEY environment variable not set")
#
#     payload = {
#         "model": CLAUDE_MODEL,
#         "max_tokens": 2200,
#         "system": SYSTEM_PROMPT,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
#                     {"type": "text", "text": user_prompt}
#                 ]
#             }
#         ]
#     }
#
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         response = await client.post(
#             CLAUDE_API_URL,
#             headers={
#                 "x-api-key": CLAUDE_API_KEY,
#                 "anthropic-version": "2023-06-01",
#                 "Content-Type": "application/json"
#             },
#             json=payload
#         )
#
#     if response.status_code != 200:
#         logger.error(f"Claude error {response.status_code}: {response.text}")
#         raise Exception(f"Claude API error {response.status_code}: {response.text}")
#
#     data = response.json()
#     report_text = data["content"][0]["text"]
#     return _parse_report(report_text, CLAUDE_MODEL, data.get("usage", {}).get("output_tokens", 0))
# ──────────────────────────────────────────────────────────────────────────────


def _parse_report(report_text: str, model: str, tokens: int) -> Dict:
    is_urgent = "⚠️ URGENT" in report_text
    urgent_finding = None
    if is_urgent:
        for line in report_text.split("\n"):
            if "⚠️ URGENT" in line:
                urgent_finding = line.replace("⚠️ URGENT:", "").strip()
                break
    return {
        "text": report_text,
        "is_urgent": is_urgent,
        "urgent_finding": urgent_finding,
        "model": model,
        "provider": "groq",
        "tokens_used": tokens
    }