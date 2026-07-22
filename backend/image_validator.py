"""
Input Validation Agent — heuristic chest X-ray gate

Runs BEFORE the vision/radiologist agents to catch non-radiograph images
(screenshots, photos, slides, memes, etc.) before they get a full clinical
report treatment. Real chest X-rays reliably share a few properties that
most non-radiograph images don't:

  1. Near-grayscale color profile (X-rays are single-channel, even when
     exported/saved as RGB JPEG/PNG)
  2. Roughly portrait-to-square aspect ratio (typical CR/DICOM export)
  3. Broad, continuous-tone intensity histogram (soft tissue → bone → air),
     as opposed to the flat, spiky histograms of UI screenshots or slides

This is a cheap heuristic, NOT a trained OOD/anomaly detector. It is
deliberately conservative: false rejections of real X-rays are worse than
letting a handful of odd-but-real ones through (the LLM does a second,
independent visual check downstream — see report_generator.py).

Install: none — numpy + Pillow only, no extra dependencies.
"""

import logging
from io import BytesIO
from typing import Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def validate_chest_xray(raw_bytes: bytes) -> Dict:
    """
    Heuristic check for whether an image plausibly looks like a chest X-ray.

    Returns:
        {
          "is_valid": bool,
          "confidence": float,       # fraction of checks passed
          "reasons": List[str],      # human-readable reasons for any failed checks
          "checks": Dict[str, Dict], # raw values per check, for the agent trace / debugging
        }
    """
    reasons = []
    checks = {}

    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "reasons": [f"Could not decode image: {e}"],
            "checks": {},
        }

    width, height = img.size
    arr = np.array(img, dtype=np.float32)

    # ── Check 1: grayscale-ness ──────────────────────────────────────────
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_diff = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))
    is_grayscale_like = channel_diff < 12.0
    checks["grayscale_like"] = {"value": round(channel_diff, 2), "pass": is_grayscale_like}
    if not is_grayscale_like:
        reasons.append(
            f"Significant color separation between channels (diff {channel_diff:.1f}) "
            f"— chest X-rays are near-grayscale even when saved as RGB"
        )

    # ── Check 2: aspect ratio ────────────────────────────────────────────
    aspect = width / height
    is_reasonable_aspect = 0.65 <= aspect <= 1.5
    checks["aspect_ratio"] = {"value": round(aspect, 2), "pass": is_reasonable_aspect}
    if not is_reasonable_aspect:
        reasons.append(
            f"Unusual aspect ratio ({width}x{height} = {aspect:.2f}) "
            f"— not typical of a chest radiograph export"
        )

    # ── Check 3: intensity distribution ─────────────────────────────────
    gray = arr.mean(axis=-1)
    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    hist_norm = hist / hist.sum()
    top_bin_fraction = float(hist_norm.max())
    is_continuous_tone = top_bin_fraction < 0.55
    checks["intensity_spread"] = {"value": round(top_bin_fraction, 2), "pass": is_continuous_tone}
    if not is_continuous_tone:
        reasons.append(
            f"Intensity histogram dominated by a single band ({top_bin_fraction * 100:.0f}% of pixels) "
            f"— typical of UI captures/slides/screenshots, not radiographic film"
        )

    passed = sum(1 for c in checks.values() if c["pass"])
    total = len(checks)
    confidence = round(passed / total, 2) if total else 0.0

    # A flat, single-band-dominated intensity histogram is the single strongest
    # giveaway of a non-radiograph (UI screens, slides, and text-on-dark-background
    # screenshots are almost never continuous-tone). A naive "2 of 3 pass" majority
    # vote lets exactly this kind of image through when it's coincidentally
    # near-square and already grayscale (e.g. a black/white presentation slide or
    # video screenshot) — so the intensity check is a mandatory gate, not just
    # one equal vote among three.
    is_valid = checks["intensity_spread"]["pass"] and passed >= 2

    return {
        "is_valid": is_valid,
        "confidence": confidence,
        "reasons": reasons,
        "checks": checks,
    }