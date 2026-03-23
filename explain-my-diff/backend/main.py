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


def _normalize_risk_level(value) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "high":
        return "High"
    if normalized == "medium":
        return "Medium"
    return "Low"


def _normalize_commit_message(value) -> str:
    message = str(value or "").strip()
    # Remove common Conventional Commit prefixes for cleaner plain messages.
    message = re.sub(
        r"^(feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert)(\([^)]+\))?!?:\s*",
        "",
        message,
        flags=re.IGNORECASE,
    )
    return message.strip()


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


def _generate_json_with_retry(prompt: str, strict_prompt: str, max_output_tokens: int) -> dict:
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": max_output_tokens,
            },
        )
        text = _get_gemini_response_text(resp)
        payload = _try_extract_json_object(text)
        if payload is None:
            raise ValueError("Failed to parse JSON from Gemini output")
        return payload
    except Exception:
        try:
            resp = model.generate_content(
                strict_prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": max_output_tokens,
                },
            )
            text = _get_gemini_response_text(resp)
            payload = _try_extract_json_object(text)
            if payload is None:
                raise ValueError("Failed to parse JSON from Gemini output")
            return payload
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini call failed (model={GEMINI_MODEL}): {e}",
            )


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
        "- explanation: string (clear markdown-friendly bullets: start each line with '- ')\n"
        "- risks: array of strings (each item a short risk statement)\n"
        "- risk_level: string ('Low' | 'Medium' | 'High') based on impact + confidence\n"
        "- commit_message: string (plain imperative sentence, no prefix like 'chore:' or 'feat:')"
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

    payload = _generate_json_with_retry(prompt, strict_prompt, max_output_tokens=700)

    return {
        "summary": str(payload.get("summary", "") or ""),
        "explanation": str(payload.get("explanation", "") or ""),
        "risks": _normalize_risks(payload.get("risks", [])),
        "risk_level": _normalize_risk_level(payload.get("risk_level", "Low")),
        "commit_message": _normalize_commit_message(payload.get("commit_message", "")),
    }


@app.post("/pr-description")
def pr_description(req: AnalyzeRequest):
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
        "You are an expert software engineer writing concise, high-quality pull request text. "
        "Analyze the unified git diff and respond with ONLY valid JSON."
    )
    output_contract = (
        "Return a single JSON object with exactly one key:\n"
        "- pr_description: string (Markdown with these sections and no extra sections)\n"
        "  ## Summary\n"
        "  - 2 to 4 bullet points\n\n"
        "  ## Test Plan\n"
        "  - 2 to 5 checklist items using '- [ ] '"
    )
    diff_note = "The diff may be truncated; write best effort." if truncated else ""
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
        "- Output JSON only (no Markdown code fences, no surrounding text).\n"
        "- Include exactly one field: pr_description.\n"
        f"{output_contract}\n"
        f"{diff_note}\n"
        "Diff:\n"
        f"{diff}\n"
    )

    payload = _generate_json_with_retry(prompt, strict_prompt, max_output_tokens=900)
    pr_text = str(payload.get("pr_description", "") or "").strip()
    if not pr_text:
        raise HTTPException(status_code=500, detail="Gemini returned empty pr_description")

    return {"pr_description": pr_text}

