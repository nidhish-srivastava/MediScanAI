import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = import.meta.env.VITE_API_URL;

function formatTime(seconds) {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function getConfidenceSummary(findings) {
  if (!findings?.length) return null;
  const high   = findings.filter(f => f.confidence >= 0.70).map(f => f.label);
  const medium = findings.filter(f => f.confidence >= 0.50 && f.confidence < 0.70).map(f => f.label);
  return { high, medium };
}

// ── CSS keyframes injected once ────────────────────────────────────────────
const STYLES = `
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
    50%       { box-shadow: 0 0 0 6px rgba(59,130,246,0.18); }
  }
  @keyframes progress-bar {
    0%   { width: 0%; }
    100% { width: 95%; }
  }
  @keyframes fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes slide-in {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  .agent-spinner {
    width: 18px; height: 18px; border-radius: 50%;
    border: 2px solid rgba(59,130,246,0.3);
    border-top-color: #3b82f6;
    animation: spin 0.8s linear infinite;
  }
  .agent-active-card {
    animation: pulse-glow 2s ease-in-out infinite;
  }
  .agent-progress {
    height: 2px; background: #1e293b; border-radius: 1px; overflow: hidden; margin-top: 8px;
  }
  .agent-progress-fill {
    height: 100%; background: linear-gradient(90deg, #3b82f6, #818cf8);
    border-radius: 1px;
    animation: progress-bar 12s ease-out forwards;
  }
  .fade-in { animation: fade-in 0.4s ease forwards; }
  .slide-in { animation: slide-in 0.3s ease forwards; }
  .connector-line {
    width: 2px; height: 10px; margin: 0 auto;
    background: linear-gradient(to bottom, #1e293b, #334155);
  }
  .connector-line.done {
    background: linear-gradient(to bottom, #166534, #15803d);
  }
`;

const AGENT_DEFS = [
  {
    key: "vision",
    name: "Vision Screening Agent",
    icon: "🔬",
    model: "TorchXRayVision DenseNet-121",
    description: "Scanning image for 18 pathology classes",
    activeStatus: "cv",
  },
  {
    key: "radiologist",
    name: "Radiologist Agent",
    icon: "🩺",
    model: "Groq · Llama-4-Scout",
    description: "Generating structured clinical report",
    activeStatus: "llm",
  },
  {
    key: "safety",
    name: "Safety Validation Agent",
    icon: "🛡️",
    model: "Rule-based validator",
    description: "Suppressing unconfirmed urgent flags",
    activeStatus: "safety",
  },
  {
    key: "formatting",
    name: "Report Formatting Agent",
    icon: "📄",
    model: "Output assembler",
    description: "Packaging report with patient context",
    activeStatus: "formatting",
  },
];

function AgentPipelinePanel({ status, agentTrace, totalElapsed }) {
  if (status === "idle") return null;

  const agentStatusOrder = ["cv", "llm", "safety", "formatting"];
  const isDone = status === "done";

  const getAgentState = (agentDef, traceEntry) => {
    if (agentTrace) {
      return { state: "done", elapsed: traceEntry?.elapsed_seconds, summary: traceEntry?.summary };
    }
    const agentIdx   = agentStatusOrder.indexOf(agentDef.activeStatus);
    const currentIdx = agentStatusOrder.indexOf(status);
    if (agentIdx < currentIdx)  return { state: "done" };
    if (agentIdx === currentIdx) return { state: "active" };
    return { state: "waiting" };
  };

  return (
    <div className="fade-in" style={{
      background: "#0a0f1e",
      borderRadius: 14,
      padding: "20px 20px 16px",
      marginBottom: 16,
      border: "1px solid #1e293b",
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: isDone ? "#22c55e" : "#3b82f6", boxShadow: isDone ? "0 0 6px #22c55e" : "0 0 6px #3b82f6" }} />
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em", color: "#64748b" }}>
            Multi-Agent Pipeline
          </span>
        </div>
        {isDone && totalElapsed !== null && (
          <span style={{ fontSize: 11, fontWeight: 600, color: "#22c55e" }}>
            ✓ {formatTime(totalElapsed)}
          </span>
        )}
        {!isDone && (
          <span style={{ fontSize: 11, color: "#475569" }}>
            {totalElapsed !== null ? `${totalElapsed}s elapsed` : "starting…"}
          </span>
        )}
      </div>

      {/* Agent cards with connectors */}
      {AGENT_DEFS.map((agentDef, i) => {
        const traceEntry = agentTrace?.[i];
        const { state, elapsed, summary } = getAgentState(agentDef, traceEntry);
        const isActive  = state === "active";
        const isDoneCard = state === "done";
        const isWaiting = state === "waiting";

        return (
          <div key={agentDef.key}>
            {/* Connector line between cards */}
            {i > 0 && (
              <div className={`connector-line${isDoneCard ? " done" : ""}`} />
            )}

            <div
              className={isActive ? "agent-active-card" : ""}
              style={{
                display: "flex", alignItems: "flex-start", gap: 12,
                padding: "12px 14px", borderRadius: 10,
                background: isDoneCard ? "rgba(5,46,22,0.6)" : isActive ? "rgba(12,26,58,0.9)" : "rgba(15,23,42,0.4)",
                border: `1px solid ${isDoneCard ? "#166534" : isActive ? "#1e40af" : "#1e293b"}`,
                transition: "all 0.4s ease",
                position: "relative", overflow: "hidden",
              }}
            >
              {/* Left glow bar for active */}
              {isActive && (
                <div style={{
                  position: "absolute", left: 0, top: 0, bottom: 0, width: 3,
                  background: "linear-gradient(to bottom, #3b82f6, #818cf8)",
                  borderRadius: "10px 0 0 10px",
                }} />
              )}
              {isDoneCard && (
                <div style={{
                  position: "absolute", left: 0, top: 0, bottom: 0, width: 3,
                  background: "#22c55e", borderRadius: "10px 0 0 10px",
                }} />
              )}

              {/* Status indicator */}
              <div style={{ flexShrink: 0, marginTop: 1 }}>
                {isDoneCard ? (
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%",
                    background: "#16a34a",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    boxShadow: "0 0 8px rgba(34,197,94,0.4)",
                  }}>
                    <span style={{ fontSize: 11, color: "#fff", fontWeight: 800 }}>✓</span>
                  </div>
                ) : isActive ? (
                  <div className="agent-spinner" />
                ) : (
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%",
                    border: "1px solid #334155", background: "transparent",
                  }} />
                )}
              </div>

              {/* Content */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: isDoneCard ? "#f0fdf4" : isActive ? "#eff6ff" : "#475569", lineHeight: 1.3 }}>
                      {agentDef.icon} {agentDef.name}
                    </div>
                    <div style={{ fontSize: 10, color: isDoneCard ? "#4ade80" : isActive ? "#60a5fa" : "#334155", marginTop: 1, fontFamily: "monospace" }}>
                      {agentDef.model}
                    </div>
                  </div>
                  {isDoneCard && elapsed !== undefined && (
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: "#4ade80", background: "rgba(34,197,94,0.1)",
                      padding: "2px 8px", borderRadius: 20, border: "1px solid rgba(34,197,94,0.2)",
                      flexShrink: 0, marginLeft: 8,
                    }}>
                      {elapsed}s
                    </span>
                  )}
                </div>

                {/* Description / summary */}
                <div style={{
                  fontSize: 11, marginTop: 4,
                  color: isDoneCard ? "#86efac" : isActive ? "#93c5fd" : "#334155",
                  lineHeight: 1.5,
                }}>
                  {isDoneCard && summary ? (
                    <span className="slide-in">{summary}</span>
                  ) : (
                    agentDef.description
                  )}
                </div>

                {/* Progress bar for active agent */}
                {isActive && (
                  <div className="agent-progress">
                    <div className="agent-progress-fill" />
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}

      {/* Footer */}
      {isDone ? (
        <div className="fade-in" style={{
          marginTop: 14, paddingTop: 12, borderTop: "1px solid #1e293b",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 6px #22c55e" }} />
          <span style={{ fontSize: 11, color: "#22c55e", fontWeight: 600, letterSpacing: "0.04em" }}>
            All agents completed successfully
          </span>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 6px #22c55e" }} />
        </div>
      ) : (
        <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid #1e293b" }}>
          <div style={{ fontSize: 10, color: "#334155", textAlign: "center", letterSpacing: "0.04em" }}>
            AGENTS COMMUNICATING IN SEQUENCE · REAL-TIME PROCESSING
          </div>
        </div>
      )}
    </div>
  );
}

// ── Safety badge ────────────────────────────────────────────────────────────
function SafetyBadge({ safety }) {
  if (!safety?.overrides?.length) return null;
  return (
    <div style={{ background: "#fefce8", border: "1px solid #fde047", borderRadius: 8, padding: "10px 14px", marginBottom: 12, fontSize: 12, color: "#713f12", display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: 16 }}>🛡️</span>
      <div>
        <span style={{ fontWeight: 600 }}>Safety Agent suppressed unconfirmed flag(s): </span>
        {safety.overrides.join(", ")} — not visually confirmed by Radiologist Agent
      </div>
    </div>
  );
}

// ── Confidence summary ──────────────────────────────────────────────────────
function ConfidenceSummary({ findings }) {
  const summary = getConfidenceSummary(findings);
  if (!summary) return null;
  return (
    <div style={{ background: "#f0f9ff", border: "1px solid #bae6fd", borderRadius: 8, padding: "10px 14px", marginBottom: 12, fontSize: 13, lineHeight: 1.6 }}>
      {summary.high.length > 0 && (
        <div><span style={{ fontWeight: 600, color: "#0369a1" }}>High confidence: </span><span style={{ color: "#0c4a6e" }}>{summary.high.join(", ")}</span></div>
      )}
      {summary.medium.length > 0 && (
        <div style={{ marginTop: summary.high.length ? 3 : 0 }}>
          <span style={{ fontWeight: 600, color: "#92400e" }}>Needs confirmation: </span>
          <span style={{ color: "#78350f" }}>{summary.medium.join(", ")}</span>
        </div>
      )}
      {summary.high.length === 0 && summary.medium.length === 0 && (
        <span style={{ color: "#374151" }}>No high-confidence findings detected.</span>
      )}
    </div>
  );
}

// ── Confidence bar ──────────────────────────────────────────────────────────
function ConfidenceBar({ confidence }) {
  const pct   = Math.round(confidence * 100);
  const color = pct >= 70 ? "#dc2626" : pct >= 50 ? "#d97706" : "#16a34a";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 90, height: 5, background: "#f3f4f6", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 700, color, minWidth: 34 }}>{pct}%</span>
    </div>
  );
}

// ── Findings panel ──────────────────────────────────────────────────────────
function FindingsPanel({ findings, reportIsUrgent, urgentFinding }) {
  if (!findings?.length) return null;
  const isLLMUrgent = (label) => {
    if (!reportIsUrgent || !urgentFinding) return false;
    return urgentFinding.toLowerCase().includes(label.toLowerCase());
  };
  return (
    <div className="fade-in" style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: "16px 20px", marginBottom: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12 }}>
        <span style={{ fontSize: 14 }}>🔬</span>
        <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em", color: "#6b7280" }}>
          Vision Agent — Detected Findings
        </span>
      </div>
      <ConfidenceSummary findings={findings} />
      {findings.map((f, i) => (
        <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: i < findings.length - 1 ? "1px solid #f9fafb" : "none" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 13, color: "#111827", fontWeight: 500 }}>{f.label}</span>
            {isLLMUrgent(f.label) && (
              <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 4, background: "#fef2f2", color: "#dc2626", border: "1px solid #fecaca", letterSpacing: "0.05em" }}>URGENT</span>
            )}
            {f.note && <span style={{ fontSize: 10, color: "#9ca3af", fontStyle: "italic" }}>fallback</span>}
          </div>
          <ConfidenceBar confidence={f.confidence} />
        </div>
      ))}
    </div>
  );
}

// ── Report panel ────────────────────────────────────────────────────────────
function ReportPanel({ report, elapsed }) {
  if (!report) return null;
  return (
    <div className="fade-in" style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: "20px 24px", boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 14 }}>🩺</span>
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em", color: "#6b7280" }}>
            Radiologist Agent — Clinical Report
          </span>
        </div>
        {elapsed !== null && (
          <span style={{ fontSize: 11, fontWeight: 600, color: "#16a34a", background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 20, padding: "3px 10px" }}>
            ✓ {formatTime(elapsed)}
          </span>
        )}
      </div>

      {report.is_urgent && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 10, padding: "12px 16px", marginBottom: 16 }}>
          <span style={{ fontSize: 20 }}>⚠️</span>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#991b1b", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Urgent Finding</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#dc2626" }}>{report.urgent_finding}</div>
          </div>
        </div>
      )}

      <div style={{ fontSize: 13, lineHeight: 1.8, color: "#374151" }}>
        <ReactMarkdown>{report.text}</ReactMarkdown>
      </div>

      <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid #f3f4f6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#9ca3af" }}>Model: {report.model}</span>
        <span style={{ fontSize: 11, color: "#9ca3af", background: "#f9fafb", padding: "2px 8px", borderRadius: 6 }}>{report.tokens_used} tokens</span>
      </div>
    </div>
  );
}

// ── PDF generator ───────────────────────────────────────────────────────────
function generatePDF({ report, patient, labName, findings, elapsed, safety }) {
  const date    = new Date().toLocaleDateString("en-IN", { year: "numeric", month: "long", day: "numeric" });
  const timeStr = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
  const lab     = labName || "MediScan AI";
  const summary = getConfidenceSummary(findings);

  const mdToHtml = (text) =>
    text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/^#{1,3} (.+)$/gm, "<h3>$1</h3>")
      .replace(/^\* (.+)$/gm, "<li>$1</li>")
      .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
      .replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br/>");

  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:Inter,sans-serif;font-size:13px;color:#1a1a1a;background:#fff;padding:40px 48px}
  .header{display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:20px;border-bottom:2px solid #111;margin-bottom:24px}
  .lab-name{font-size:20px;font-weight:700}.lab-sub{font-size:11px;color:#6b7280;margin-top:2px}
  .report-meta{text-align:right;font-size:11px;color:#6b7280;line-height:1.7}
  .patient-block{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px 18px;margin-bottom:20px;display:grid;grid-template-columns:1fr 1fr;gap:6px 24px}
  .patient-field{font-size:12px}.patient-field span{color:#6b7280}
  .urgent-box{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px 16px;margin-bottom:20px;font-weight:600;color:#dc2626}
  .safety-box{background:#fefce8;border:1px solid #fde047;border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:12px;color:#713f12}
  .summary-box{background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:12px 16px;margin-bottom:20px;font-size:12px;line-height:1.7}
  .summary-box .label{font-weight:600;color:#0369a1}
  .agent-tag{display:inline-flex;align-items:center;gap:6px;background:#0f172a;border-radius:20px;padding:4px 12px;font-size:10px;font-weight:700;color:#64748b;margin-bottom:16px;letter-spacing:0.06em}
  .agent-tag span{color:#22c55e}
  .report-body{line-height:1.85}.report-body h3{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#6b7280;margin:20px 0 6px;border-bottom:1px solid #f3f4f6;padding-bottom:4px}
  .report-body ul{padding-left:20px;margin:4px 0}.report-body li{margin-bottom:4px}.report-body p{margin-bottom:8px}
  .footer{margin-top:36px;padding-top:14px;border-top:1px solid #e5e7eb;display:flex;justify-content:space-between;font-size:10px;color:#9ca3af}
</style>
</head><body>
  <div class="header">
    <div><div class="lab-name">${lab}</div><div class="lab-sub">Multi-Agent AI Radiology Report</div></div>
    <div class="report-meta">
      <div><strong>Date:</strong> ${date}</div>
      <div><strong>Time:</strong> ${timeStr}</div>
      <div><strong>Report ID:</strong> CXR-${Date.now().toString().slice(-6)}</div>
      ${elapsed ? `<div><strong>Processing:</strong> ${formatTime(elapsed)}</div>` : ""}
    </div>
  </div>
  ${patient.name || patient.age || patient.sex || patient.ref_doc ? `<div class="patient-block">
    ${patient.name    ? `<div class="patient-field"><span>Patient: </span><strong>${patient.name}</strong></div>` : ""}
    ${patient.age     ? `<div class="patient-field"><span>Age: </span><strong>${patient.age}</strong></div>` : ""}
    ${patient.sex     ? `<div class="patient-field"><span>Sex: </span><strong>${patient.sex}</strong></div>` : ""}
    ${patient.ref_doc ? `<div class="patient-field"><span>Referring Doctor: </span><strong>Dr. ${patient.ref_doc}</strong></div>` : ""}
    ${patient.history ? `<div class="patient-field" style="grid-column:1/-1"><span>Clinical History: </span><strong>${patient.history}</strong></div>` : ""}
  </div>` : ""}
  <div class="agent-tag">🤖 GENERATED BY 4-AGENT AI PIPELINE <span>· ${elapsed ? formatTime(elapsed) : ""}  ✓</span></div>
  ${report.is_urgent ? `<div class="urgent-box">⚠️ URGENT: ${report.urgent_finding}</div>` : ""}
  ${safety?.overrides?.length ? `<div class="safety-box">🛡️ Safety Agent suppressed unconfirmed flag(s): ${safety.overrides.join(", ")}</div>` : ""}
  ${summary && (summary.high.length || summary.medium.length) ? `<div class="summary-box">
    ${summary.high.length   ? `<div><span class="label">High confidence findings: </span>${summary.high.join(", ")}</div>` : ""}
    ${summary.medium.length ? `<div><span class="label">Needs confirmation: </span>${summary.medium.join(", ")}</div>` : ""}
  </div>` : ""}
  <div class="report-body"><p>${mdToHtml(report.text)}</p></div>
  <div class="footer">
    <div style="font-style:italic;max-width:60%">For clinical decision support only. Not a substitute for qualified radiologist review. All findings require clinical correlation.</div>
    <div style="font-weight:600">Powered by ${lab} · Multi-Agent AI</div>
  </div>
</body></html>`;

  const blob = new Blob([html], { type: "text/html" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = `CXR_${patient.name?.replace(/\s+/g, "_") || "Report"}_${new Date().toISOString().slice(0,10)}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Patient form ────────────────────────────────────────────────────────────
function PatientForm({ patient, onChange, labName, onLabChange, collapsed, onToggle }) {
  const field = (label, key, placeholder) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</label>
      <input type="text" placeholder={placeholder} value={patient[key] || ""}
        onChange={e => onChange({ ...patient, [key]: e.target.value })}
        style={{ padding: "8px 10px", borderRadius: 7, border: "1px solid #e5e7eb", fontSize: 13, color: "#111827", outline: "none", background: "#fff" }} />
    </div>
  );

  const hasData = labName || Object.values(patient).some(Boolean);

  return (
    <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, marginBottom: 14, overflow: "hidden", boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
      {/* Collapsible header */}
      <div onClick={onToggle} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "14px 20px", cursor: "pointer", userSelect: "none" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13 }}>🏥</span>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>Patient & Institution</span>
          {hasData && collapsed && (
            <span style={{ fontSize: 10, background: "#dcfce7", color: "#16a34a", padding: "1px 7px", borderRadius: 20, fontWeight: 600 }}>Filled</span>
          )}
        </div>
        <span style={{ fontSize: 12, color: "#9ca3af", transform: collapsed ? "rotate(0deg)" : "rotate(180deg)", transition: "transform 0.2s", display: "inline-block" }}>▼</span>
      </div>

      {!collapsed && (
        <div style={{ padding: "0 20px 16px", borderTop: "1px solid #f3f4f6" }}>
          <div style={{ paddingTop: 14, marginBottom: 14 }}>
            <label style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>Lab / Institution Name</label>
            <input type="text" placeholder="e.g. Apollo Diagnostics, City Radiology..." value={labName}
              onChange={e => onLabChange(e.target.value)}
              style={{ width: "100%", padding: "8px 10px", borderRadius: 7, border: "1px solid #c7d2fe", fontSize: 13, color: "#111827", outline: "none", background: "#eef2ff" }} />
            <div style={{ fontSize: 11, color: "#818cf8", marginTop: 3 }}>Appears on PDF report header</div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {field("Patient Name", "name", "Full name")}
            {field("Age", "age", "e.g. 45 years")}
            {field("Sex", "sex", "M / F / Other")}
            {field("Referring Doctor", "ref_doc", "Dr. Name")}
          </div>
          <div style={{ marginTop: 10 }}>
            <label style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>Clinical History</label>
            <input type="text" placeholder="e.g. Fever 3 days, cough, decreased breath sounds"
              value={patient.history || ""} onChange={e => onChange({ ...patient, history: e.target.value })}
              style={{ width: "100%", padding: "8px 10px", borderRadius: 7, border: "1px solid #e5e7eb", fontSize: 13, color: "#111827", outline: "none" }} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [image,         setImage]         = useState(null);
  const [status,        setStatus]        = useState("idle");
  const [result,        setResult]        = useState(null);
  const [error,         setError]         = useState("");
  const [dragging,      setDragging]      = useState(false);
  const [patient,       setPatient]       = useState({});
  const [labName,       setLabName]       = useState("");
  const [elapsed,       setElapsed]       = useState(null);
  const [formCollapsed, setFormCollapsed] = useState(false);

  const timerRef  = useRef(null);
  const startRef  = useRef(null);
  const fileRef   = useRef();
  const resultRef = useRef();

  const startTimer = () => {
    startRef.current = Date.now();
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
  };

  const stopTimer = () => {
    clearInterval(timerRef.current);
    setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
  };

  useEffect(() => () => clearInterval(timerRef.current), []);

  // Auto-scroll to results when done
  useEffect(() => {
    if (status === "done" && resultRef.current) {
      setTimeout(() => resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
    }
  }, [status]);

  const loadFile = (file) => {
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Only JPEG, PNG, or WebP images supported."); return;
    }
    setImage({ file, url: URL.createObjectURL(file) });
    setResult(null); setError(""); setStatus("idle"); setElapsed(null);
    setFormCollapsed(true); // collapse form after image load
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false); loadFile(e.dataTransfer.files[0]);
  }, []);

  const analyze = async () => {
    if (!image) return;
    setError(""); setResult(null);
    startTimer();
    setStatus("uploading");

    const form = new FormData();
    form.append("file", image.file);
    if (patient.name)    form.append("patient_name",    patient.name);
    if (patient.age)     form.append("patient_age",     patient.age);
    if (patient.sex)     form.append("patient_sex",     patient.sex);
    if (patient.ref_doc) form.append("patient_ref_doc", patient.ref_doc);
    if (patient.history) form.append("patient_history", patient.history);

    // Simulate agent transitions (calibrated for ~20s total response)
    setTimeout(() => setStatus("cv"),         300);
    setTimeout(() => setStatus("llm"),        3500);
    setTimeout(() => setStatus("safety"),     5500);
    setTimeout(() => setStatus("formatting"), 6500);

    try {
      const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      stopTimer();
      setResult(data);
      setStatus("done");
    } catch (e) {
      stopTimer(); setError(e.message); setStatus("error");
    }
  };

  const reset = () => {
    setImage(null); setResult(null); setError("");
    setStatus("idle"); setElapsed(null); setFormCollapsed(false);
  };

  const isRunning = ["uploading", "cv", "llm", "safety", "formatting"].includes(status);

  return (
    <>
      <style>{STYLES}</style>
      <div style={{ minHeight: "100vh", background: "#f3f4f6", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
        <div style={{ maxWidth: 760, margin: "0 auto", padding: "28px 16px 48px" }}>

          {/* Header */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <h1 style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", margin: 0, letterSpacing: "-0.5px" }}>MediScan AI</h1>
              <span style={{ fontSize: 10, fontWeight: 700, padding: "3px 9px", borderRadius: 20, background: "#ede9fe", color: "#7c3aed", border: "1px solid #ddd6fe", letterSpacing: "0.06em" }}>
                4-AGENT PIPELINE
              </span>
            </div>
            <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>
              Vision Agent → Radiologist Agent → Safety Agent → Formatting Agent
            </p>
          </div>

          {/* Patient form — collapsible */}
          <PatientForm
            patient={patient} onChange={setPatient}
            labName={labName} onLabChange={setLabName}
            collapsed={formCollapsed} onToggle={() => setFormCollapsed(v => !v)}
          />

          {/* Upload zone */}
          {!image ? (
            <div onClick={() => fileRef.current.click()} onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              style={{
                border: `2px dashed ${dragging ? "#7c3aed" : "#d1d5db"}`,
                borderRadius: 14, padding: "52px 24px", textAlign: "center", cursor: "pointer",
                background: dragging ? "#f5f3ff" : "#fff",
                transition: "all 0.15s", marginBottom: 14,
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
              }}>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => loadFile(e.target.files[0])} />
              <div style={{ fontSize: 44, marginBottom: 12 }}>🫁</div>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#111827", marginBottom: 4 }}>Drop chest X-ray here</div>
              <div style={{ fontSize: 13, color: "#9ca3af" }}>or click to browse · JPEG, PNG, WebP · max 15MB</div>
            </div>
          ) : (
            <div style={{ position: "relative", marginBottom: 14, borderRadius: 14, overflow: "hidden", border: "1px solid #e5e7eb", background: "#000", boxShadow: "0 2px 8px rgba(0,0,0,0.12)" }}>
              <img src={image.url} alt="X-ray" style={{ display: "block", width: "100%", maxHeight: 360, objectFit: "contain" }} />
              {/* Image meta overlay */}
              <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "linear-gradient(transparent, rgba(0,0,0,0.7))", padding: "16px 14px 10px", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.7)", fontFamily: "monospace" }}>{image.file.name}</span>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)" }}>{(image.file.size / 1024).toFixed(0)} KB</span>
              </div>
              {!isRunning && (
                <button onClick={reset} style={{ position: "absolute", top: 10, right: 10, background: "rgba(0,0,0,0.6)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: "50%", width: 30, height: 30, color: "#fff", cursor: "pointer", fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center" }}>✕</button>
              )}
            </div>
          )}

          {error && (
            <div style={{ background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#dc2626", marginBottom: 14 }}>
              {error}
            </div>
          )}

          {/* Analyze button */}
          {image && status !== "done" && (
            <button onClick={analyze} disabled={isRunning} style={{
              width: "100%", padding: "14px", borderRadius: 12, border: "none",
              background: isRunning ? "#94a3b8" : "linear-gradient(135deg, #0f172a, #1e293b)",
              color: "#fff", fontSize: 14, fontWeight: 700, cursor: isRunning ? "not-allowed" : "pointer",
              marginBottom: 14, letterSpacing: "0.02em", transition: "opacity 0.15s",
            }}>
              {isRunning ? "⟳  Agents running…" : "▶  Run Agent Pipeline"}
            </button>
          )}

          {/* Agent pipeline panel */}
          <AgentPipelinePanel
            status={status}
            agentTrace={result?.agent_trace || null}
            totalElapsed={elapsed}
          />

          {/* Results */}
          {result && (
            <div ref={resultRef}>
              <SafetyBadge safety={result.safety} />
              <FindingsPanel
                findings={result.cv_findings}
                reportIsUrgent={result.report?.is_urgent}
                urgentFinding={result.report?.urgent_finding}
              />
              <ReportPanel report={result.report} elapsed={elapsed} />

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 12 }}>
                <button
                  onClick={() => generatePDF({ report: result.report, patient, labName, findings: result.cv_findings, elapsed, safety: result.safety })}
                  style={{ padding: "13px", borderRadius: 10, border: "none", background: "#1d4ed8", color: "#fff", fontSize: 13, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                  ⬇ Download PDF
                </button>
                <button onClick={reset}
                  style={{ padding: "13px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", fontSize: 13, color: "#374151", cursor: "pointer", fontWeight: 500 }}>
                  ↩ New Analysis
                </button>
              </div>
            </div>
          )}

          <p style={{ fontSize: 11, color: "#9ca3af", textAlign: "center", marginTop: 28, lineHeight: 1.6 }}>
            For clinical decision support only · Not a substitute for qualified radiologist review
          </p>
        </div>
      </div>
    </>
  );
}