from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # Stub implementation: returns mocked data for now.
    return {
        "summary": "Mock summary of the changes.",
        "explanation": "Mock explanation of what changed (frontend + backend stub).",
        "risks": [
            "May not reflect the real diff yet (stubbed response).",
            "Potential mismatch between expected and actual response schema.",
        ],
        "commit_message": "chore: explain diff changes (mock)",
    }

