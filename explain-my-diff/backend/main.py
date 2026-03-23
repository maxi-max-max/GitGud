import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from dotenv import load_dotenv

# Load local secrets from the repo root `.env`.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview").strip()
MAX_DIFF_CHARS = int(os.getenv("MAX_DIFF_CHARS", "25000"))

if not GEMINI_API_KEY:
    # Fail fast with a helpful error when the endpoint is hit.
    genai = None  # type: ignore[assignment]
else:
    genai.configure(api_key=GEMINI_API_KEY)

print(f"[startup] Using GEMINI_MODEL={GEMINI_MODEL}")

app = FastAPI(title="Explain My Diff")

# Allow local React dev servers to call this API.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "service": "explain-my-diff"}


class AnalyzeRequest(BaseModel):
    diff: str


def _try_extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort extraction of a single top-level JSON object from model output.
    Gemini sometimes wraps JSON in code fences; this removes them and parses.
    """

    if not text:
        return None

    cleaned = text.strip()

    # Remove common Markdown code fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # If the model includes extra text, keep the first `{...}` block.
    match = re.search(r"\{[\s\S]*?\}", cleaned)
    if match:
        cleaned = match.group(0)

    # Remove trailing commas before `}` or `]` (common JSON-ish mistake).
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _normalize_risks(risks_value) -> list[str]:
    if isinstance(risks_value, list):
        return [str(x).strip() for x in risks_value if str(x).strip()]
    if isinstance(risks_value, str):
        # Handle either bullet lines or a single paragraph.
        lines = [ln.strip() for ln in risks_value.splitlines() if ln.strip()]
        normalized: list[str] = []
        for ln in lines:
            ln = re.sub(r"^[-*]\s*", "", ln)
            if ln:
                normalized.append(ln)
        return normalized
    return [str(risks_value).strip()] if str(risks_value).strip() else []


def _get_gemini_response_text(resp) -> str:
    # The google-generativeai SDK typically exposes `resp.text`, but we keep
    # fallbacks to be tolerant to minor SDK response shape differences.
    text = getattr(resp, "text", None)
    if text:
        return str(text)
    try:
        return str(resp.candidates[0].content.parts[0].text)
    except Exception:
        return str(resp)


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if genai is None:
        raise HTTPException(
            status_code=500,
            detail="Missing GEMINI_API_KEY. Set it in the repo root .env file.",
        )

    diff = (req.diff or "").strip()
    if not diff:
        raise HTTPException(status_code=400, detail="diff must not be empty")

    truncated = False
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS]
        truncated = True

    base_instructions = (
        "You are an expert software engineer and code reviewer. "
        "Analyze the following unified git diff and respond with ONLY valid JSON."
    )

    output_contract = (
        "Return a single JSON object with exactly these keys:\n"
        "- summary: string (1-2 sentences)\n"
        "- explanation: string (what changed and why)\n"
        "- risks: array of strings (each item a short risk statement)\n"
        "- commit_message: string (Conventional Commits style)"
    )

    diff_note = "The diff may be truncated; analyze best effort." if truncated else ""

    prompt = (
        f"{base_instructions}\n"
        f"{output_contract}\n"
        f"{diff_note}\n"
        "Diff:\n"
        f"{diff}\n"
    )

    strict_prompt = (
        f"{base_instructions}\n"
        "Rules:\n"
        "- Output JSON only (no Markdown, no code fences, no surrounding text).\n"
        "- If unsure, still produce the JSON with best-effort content.\n"
        f"{output_contract}\n"
        f"{diff_note}\n"
        "Diff:\n"
        f"{diff}\n"
    )

    model = genai.GenerativeModel(GEMINI_MODEL)

    # First attempt (normal JSON-only instruction).
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 700,
            },
        )
        text = _get_gemini_response_text(resp)
        payload = _try_extract_json_object(text)
        if payload is None:
            raise ValueError("Failed to parse JSON from Gemini output")
    except Exception:
        # Second attempt (stricter instruction).
        try:
            resp = model.generate_content(
                strict_prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 700,
                },
            )
            text = _get_gemini_response_text(resp)
            payload = _try_extract_json_object(text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini call failed (model={GEMINI_MODEL}): {e}",
            )

    if payload is None:
        raise HTTPException(status_code=500, detail="Gemini returned non-JSON output")

    return {
        "summary": str(payload.get("summary", "") or ""),
        "explanation": str(payload.get("explanation", "") or ""),
        "risks": _normalize_risks(payload.get("risks", [])),
        "commit_message": str(payload.get("commit_message", "") or ""),
    }

