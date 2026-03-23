import React, { useState } from "react";

const API_BASE_URL = "http://localhost:8000";

function Section({ title, children }) {
  return (
    <section className="section">
      <h2 className="sectionTitle">{title}</h2>
      <div className="sectionBody">{children}</div>
    </section>
  );
}

export default function App() {
  const [diffText, setDiffText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [summary, setSummary] = useState("");
  const [explanation, setExplanation] = useState("");
  const [risks, setRisks] = useState("");
  const [commitMessage, setCommitMessage] = useState("");

  async function analyzeDiff() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ diff: diffText }),
      });

      if (!res.ok) {
        throw new Error(`Backend error: ${res.status}`);
      }

      const data = await res.json();
      setSummary(data.summary ?? "");
      setExplanation(data.explanation ?? "");
      setRisks(Array.isArray(data.risks) ? data.risks.join("\n") : data.risks ?? "");
      setCommitMessage(data.commit_message ?? "");
    } catch (e) {
      setError(e?.message ?? "Failed to analyze diff");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1 className="title">Explain My Diff</h1>
        <p className="subtitle">
          Paste a git diff and get a short summary, explanation, risks, and a commit message suggestion.
        </p>
      </header>

      <main className="main">
        <div className="inputBlock">
          <label className="label" htmlFor="diff">
            Git diff
          </label>
          <textarea
            id="diff"
            className="textarea"
            value={diffText}
            onChange={(e) => setDiffText(e.target.value)}
            placeholder={"--- a/file.txt\n++++ b/file.txt\n@@ ..."}
          />
          <button className="button" onClick={analyzeDiff} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Diff"}
          </button>
          {error ? <div className="error">{error}</div> : null}
        </div>

        <div className="resultsGrid">
          <Section title="Summary">{summary ? summary : ""}</Section>
          <Section title="Explanation">{explanation ? explanation : ""}</Section>
          <Section title="Risks">{risks ? risks : ""}</Section>
          <Section title="Commit Message">{commitMessage ? commitMessage : ""}</Section>
        </div>
      </main>
    </div>
  );
}

