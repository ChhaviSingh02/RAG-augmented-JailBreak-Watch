import { useState, useEffect, useRef } from "react";

const API_URL = "http://localhost:8000";

function cosimScore(score) {
  const w = Math.round(score * 100);
  const color = score > 0.7 ? "#ff3b3b" : score > 0.4 ? "#f5a623" : "#00e676";
  return { w, color };
}

function timeAgo(ts) {
  const date = new Date(ts);
  const s = Math.floor((Date.now() - date.getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

const ACTION_COLORS = { Block: "#ff3b3b", Warn: "#f5a623", Log: "#4fc3f7" };

export default function App() {
  const [prompts, setPrompts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [search, setSearch] = useState("");
  const [filterMode, setFilterMode] = useState("all");
  const [audit, setAudit] = useState([]);
  const [stats, setStats] = useState({ total: 0, jailbreaks: 0, blocked: 0, pending: 0 });
  const [feedbackNote, setFeedbackNote] = useState("");
  const [severityRating, setSeverityRating] = useState(3);
  const [exportMsg, setExportMsg] = useState("");
  const wsRef = useRef(null);

  // WebSocket connection for live feed
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/feed");
    
    ws.onopen = () => {
      console.log("WebSocket connected");
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === "new_detection") {
        const data = message.data;
        const newPrompt = {
          id: data.prompt_id,
          text: data.text,
          safe: data.is_safe,
          score: data.similarity_score,
          ts: data.timestamp,
          status: "pending",
          classifier_label: data.classifier_label,
          classifier_confidence: data.classifier_confidence,
          agent_action: data.agent_action,
          agent_confidence: data.agent_confidence,
          attack_vector: data.attack_vector,
          escalation: data.escalation,
          reasoning: data.reasoning,
          top_matches: data.top_matches
        };
        
        setPrompts(prev => [newPrompt, ...prev].slice(0, 100));
        
        // Auto-select first prompt if none selected
        if (!selected) {
          setSelected(newPrompt);
        }
      }
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
    
    wsRef.current = ws;
    
    return () => ws.close();
  }, []);

  // Load statistics periodically
  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetch(`${API_URL}/statistics`);
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error("Failed to load stats:", error);
      }
    };
    
    loadStats();
    const interval = setInterval(loadStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load audit trail
  useEffect(() => {
    const loadAudit = async () => {
      try {
        const response = await fetch(`${API_URL}/audit-trail`);
        const data = await response.json();
        setAudit(data.entries || []);
      } catch (error) {
        console.error("Failed to load audit:", error);
      }
    };
    
    loadAudit();
  }, []);

  const filtered = prompts.filter(p => {
    const matchSearch = p.text.toLowerCase().includes(search.toLowerCase());
    const matchFilter =
      filterMode === "all" ||
      (filterMode === "safe" && p.safe) ||
      (filterMode === "jailbreak" && !p.safe);
    return matchSearch && matchFilter;
  });

  const handleAction = async (action) => {
    if (!selected) return;
    
    try {
      await fetch(`${API_URL}/human-decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_id: selected.id,
          action: action,
          note: feedbackNote,
          severity: severityRating
        })
      });
      
      setPrompts(prev => prev.map(p => p.id === selected.id ? { ...p, status: action } : p));
      setSelected(prev => ({ ...prev, status: action }));
      setFeedbackNote("");
      
      // Reload audit trail
      const response = await fetch(`${API_URL}/audit-trail`);
      const data = await response.json();
      setAudit(data.entries || []);
      
    } catch (error) {
      console.error("Failed to submit decision:", error);
    }
  };

  const exportLogs = (format) => {
    const data = audit.map(a => ({
      id: a.id,
      prompt: a.prompt_text,
      action: a.action,
      score: a.score ? a.score.toFixed(2) : "N/A",
      note: a.note || "",
      severity: a.severity || 0,
      timestamp: a.ts
    }));
    
    let content, mime, ext;
    if (format === "csv") {
      const headers = Object.keys(data[0] || {}).join(",");
      const rows = data.map(r => Object.values(r).map(v => `"${v}"`).join(",")).join("\n");
      content = headers + "\n" + rows;
      mime = "text/csv"; ext = "csv";
    } else {
      content = JSON.stringify(data, null, 2);
      mime = "application/json"; ext = "json";
    }
    
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `audit_log.${ext}`; a.click();
    setExportMsg(`Exported ${data.length} entries as ${ext.toUpperCase()}`);
    setTimeout(() => setExportMsg(""), 3000);
  };

  const topPatterns = selected?.top_matches || [];
  const confidence = selected?.agent_confidence || 0;
  const proposedAction = selected?.agent_action || "Log";

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0c10",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
      display: "flex",
      flexDirection: "column",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0c10; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }
        .pulse-red { animation: pulseRed 1.5s ease-in-out infinite; }
        @keyframes pulseRed { 0%,100% { box-shadow: 0 0 0 0 rgba(255,59,59,0.4); } 50% { box-shadow: 0 0 0 6px rgba(255,59,59,0); } }
        .pulse-green { animation: pulseGreen 2s ease-in-out infinite; }
        @keyframes pulseGreen { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .slide-in { animation: slideIn 0.3s ease-out; }
        @keyframes slideIn { from { transform: translateX(-12px); opacity: 0; } to { transform: none; opacity: 1; } }
        .btn { cursor: pointer; border: none; border-radius: 4px; font-family: inherit; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; transition: all 0.15s; }
        .btn:hover { filter: brightness(1.2); transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        .tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 9px; font-weight: 700; letter-spacing: 0.12em; }
        .section-title { font-family: 'Syne', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: #4a6fa5; margin-bottom: 10px; border-bottom: 1px solid #1a2744; padding-bottom: 6px; }
        .prompt-row { padding: 10px 12px; border-bottom: 1px solid #111620; cursor: pointer; transition: background 0.12s; display: flex; gap: 10px; align-items: flex-start; }
        .prompt-row:hover { background: #111620; }
        .prompt-row.selected { background: #131d30; border-left: 2px solid #4fc3f7; }
        .score-bar-bg { height: 4px; background: #1a2030; border-radius: 2px; margin-top: 6px; }
        .panel { background: #0d1117; border: 1px solid #1a2030; border-radius: 6px; padding: 14px; }
      `}</style>

      {/* HEADER */}
      <div style={{ background: "#0d1117", borderBottom: "1px solid #1a2030", padding: "12px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em", color: "#fff" }}>
            <span style={{ color: "#ff3b3b" }}>⬡</span> JAILBREAK<span style={{ color: "#4fc3f7" }}>WATCH</span>
          </div>
          <div style={{ display: "flex", gap: 16, fontSize: 11 }}>
            {[["TOTAL", stats.total, "#4fc3f7"], ["THREATS", stats.jailbreaks, "#ff3b3b"], ["BLOCKED", stats.blocked, "#f5a623"], ["PENDING", stats.pending, "#a0aec0"]].map(([l, v, c]) => (
              <div key={l} style={{ display: "flex", gap: 5, alignItems: "center" }}>
                <span style={{ color: "#4a6fa5" }}>{l}</span>
                <span style={{ color: c, fontWeight: 700 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div className="pulse-green" style={{ width: 8, height: 8, borderRadius: "50%", background: "#00e676" }} />
          <span style={{ fontSize: 10, color: "#00e676" }}>LIVE</span>
        </div>
      </div>

      {/* MAIN GRID */}
      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gridTemplateRows: "1fr auto", flex: 1, overflow: "hidden", gap: 0 }}>

        {/* LEFT — PROMPT MONITOR */}
        <div style={{ borderRight: "1px solid #1a2030", display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ padding: "12px", borderBottom: "1px solid #1a2030" }}>
            <div className="section-title">⬡ Prompt Monitor</div>
            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search prompts..."
                style={{ flex: 1, background: "#111620", border: "1px solid #1a2030", borderRadius: 4, padding: "6px 10px", color: "#e2e8f0", fontSize: 11, fontFamily: "inherit", outline: "none" }}
              />
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {["all", "safe", "jailbreak"].map(m => (
                <button key={m} className="btn" onClick={() => setFilterMode(m)}
                  style={{ flex: 1, padding: "5px 0", background: filterMode === m ? "#162040" : "transparent", color: filterMode === m ? "#4fc3f7" : "#4a6fa5", border: `1px solid ${filterMode === m ? "#1e3a5f" : "#1a2030"}` }}>
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div style={{ flex: 1, overflowY: "auto" }}>
            {filtered.length === 0 && (
              <div style={{ padding: 20, textAlign: "center", color: "#4a6fa5", fontSize: 11 }}>
                Waiting for prompts...
              </div>
            )}
            {filtered.map(p => {
              const { w, color } = cosimScore(p.score);
              return (
                <div key={p.id} className={`prompt-row slide-in${selected?.id === p.id ? " selected" : ""}`} onClick={() => setSelected(p)}>
                  <div style={{ flexShrink: 0, marginTop: 2 }}>
                    {p.status === "Block" ? <span style={{ color: "#ff3b3b", fontSize: 13 }}>⊘</span>
                      : p.status === "Warn" ? <span style={{ color: "#f5a623", fontSize: 13 }}>⚠</span>
                      : p.status === "Log" ? <span style={{ color: "#4fc3f7", fontSize: 13 }}>◎</span>
                      : p.safe ? <span style={{ color: "#00e676", fontSize: 13 }}>●</span>
                      : <span className="pulse-red" style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#ff3b3b" }} />}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 11, lineHeight: 1.4, color: p.safe ? "#c8d6e8" : "#ffb3b3", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.text}</div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                      <span className="tag" style={{ background: p.safe ? "#0a2010" : "#200a0a", color: p.safe ? "#00e676" : "#ff3b3b" }}>
                        {p.safe ? "SAFE" : "THREAT"}
                      </span>
                      <span style={{ fontSize: 9, color: "#4a6fa5" }}>{timeAgo(p.ts)}</span>
                    </div>
                    <div className="score-bar-bg">
                      <div style={{ width: `${w}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.5s" }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* RIGHT — DETECTION + AGENT + OVERSIGHT */}
        <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>

            {/* SELECTED PROMPT BANNER */}
            {selected && (
              <div style={{ background: "#0d1117", border: `1px solid ${selected.safe ? "#0a3020" : "#3a1010"}`, borderRadius: 6, padding: "10px 14px" }}>
                <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 4 }}>ANALYZING PROMPT #{selected.id}</div>
                <div style={{ fontSize: 12, color: selected.safe ? "#c8d6e8" : "#ffb3b3", lineHeight: 1.5, fontStyle: "italic" }}>"{selected.text}"</div>
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {/* DETECTION RESULTS */}
              <div className="panel">
                <div className="section-title">◈ Detection Results</div>
                {selected ? (
                  <>
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 4 }}>SIMILARITY SCORE</div>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ flex: 1, height: 8, background: "#1a2030", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ width: `${selected.score * 100}%`, height: "100%", background: cosimScore(selected.score).color, borderRadius: 4, transition: "width 0.6s ease" }} />
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 700, color: cosimScore(selected.score).color, minWidth: 40 }}>
                          {(selected.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 6 }}>CLASSIFIER OUTPUT</div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <div style={{ flex: 1, padding: "8px", background: selected.classifier_label === 0 ? "#0a2010" : "#0d0d0d", border: `1px solid ${selected.classifier_label === 0 ? "#0d3020" : "#1a1a1a"}`, borderRadius: 4, textAlign: "center" }}>
                          <div style={{ fontSize: 16, marginBottom: 2 }}>✓</div>
                          <div style={{ fontSize: 9, color: selected.classifier_label === 0 ? "#00e676" : "#2a4a2a" }}>SAFE</div>
                          <div style={{ fontSize: 12, fontWeight: 700, color: selected.classifier_label === 0 ? "#00e676" : "#2a4a2a" }}>{selected.classifier_label === 0 ? `${(selected.classifier_confidence * 100).toFixed(0)}%` : `${((1 - selected.classifier_confidence) * 100).toFixed(0)}%`}</div>
                        </div>
                        <div style={{ flex: 1, padding: "8px", background: selected.classifier_label === 1 ? "#200a0a" : "#0d0d0d", border: `1px solid ${selected.classifier_label === 1 ? "#3a1010" : "#1a1a1a"}`, borderRadius: 4, textAlign: "center" }}>
                          <div style={{ fontSize: 16, marginBottom: 2 }}>⚡</div>
                          <div style={{ fontSize: 9, color: selected.classifier_label === 1 ? "#ff3b3b" : "#4a2a2a" }}>JAILBREAK</div>
                          <div style={{ fontSize: 12, fontWeight: 700, color: selected.classifier_label === 1 ? "#ff3b3b" : "#4a2a2a" }}>{selected.classifier_label === 1 ? `${(selected.classifier_confidence * 100).toFixed(0)}%` : `${((1 - selected.classifier_confidence) * 100).toFixed(0)}%`}</div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 6 }}>RAG — TOP 5 RELATED PATTERNS</div>
                      {topPatterns.slice(0, 5).map((p, i) => (
                        <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 5, padding: "5px 8px", background: "#0a0c10", borderRadius: 3, border: "1px solid #111620" }}>
                          <span style={{ fontSize: 9, color: "#4a6fa5", minWidth: 14 }}>#{i + 1}</span>
                          <span style={{ fontSize: 9, color: "#8a9ab0", flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.text}</span>
                          <span style={{ fontSize: 9, fontWeight: 700, color: cosimScore(p.score).color, minWidth: 30, textAlign: "right" }}>{(p.score * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : <div style={{ fontSize: 11, color: "#4a6fa5" }}>Select a prompt to analyze</div>}
              </div>

              {/* AGENT ACTIONS */}
              <div className="panel">
                <div className="section-title">⬡ Agent Actions</div>
                {selected ? (
                  <>
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 6 }}>PROPOSED COUNTERMEASURE</div>
                      <div style={{ display: "flex", gap: 6 }}>
                        {["Block", "Warn", "Log"].map(a => (
                          <div key={a} style={{ flex: 1, padding: "10px 6px", background: proposedAction === a ? `${ACTION_COLORS[a]}22` : "#0a0c10", border: `1px solid ${proposedAction === a ? ACTION_COLORS[a] : "#1a2030"}`, borderRadius: 4, textAlign: "center" }}>
                            <div style={{ fontSize: 14, marginBottom: 2 }}>{a === "Block" ? "⊘" : a === "Warn" ? "⚠" : "◎"}</div>
                            <div style={{ fontSize: 9, color: proposedAction === a ? ACTION_COLORS[a] : "#4a6fa5", fontWeight: 700 }}>{a.toUpperCase()}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 4 }}>CONFIDENCE LEVEL</div>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ flex: 1, height: 6, background: "#1a2030", borderRadius: 3 }}>
                          <div style={{ width: `${confidence}%`, height: "100%", background: "#4fc3f7", borderRadius: 3, transition: "width 0.6s" }} />
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 700, color: "#4fc3f7", minWidth: 40 }}>{confidence}%</span>
                      </div>
                    </div>

                    <div style={{ padding: "10px", background: "#0a0c10", border: "1px solid #1a2030", borderRadius: 4, marginBottom: 10 }}>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 4 }}>REASONING</div>
                      <div style={{ fontSize: 10, color: "#c8d6e8", lineHeight: 1.5 }}>
                        {selected.reasoning}
                      </div>
                    </div>

                    <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 4 }}>ESCALATION: <span style={{ color: selected.escalation === "Urgent" ? "#ff3b3b" : selected.escalation === "Human Review" ? "#f5a623" : "#00e676", fontWeight: 700 }}>{selected.escalation}</span></div>
                  </>
                ) : <div style={{ fontSize: 11, color: "#4a6fa5" }}>Select a prompt to analyze</div>}
              </div>
            </div>

            {/* HUMAN OVERSIGHT */}
            <div className="panel">
              <div className="section-title">◉ Human Oversight</div>
              {selected ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <div>
                    <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 6 }}>APPROVE / REJECT / OVERRIDE</div>
                    <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                      {["Block", "Warn", "Log"].map(a => (
                        <button key={a} className="btn" onClick={() => handleAction(a)}
                          style={{ flex: 1, padding: "8px 4px", background: selected.status === a ? `${ACTION_COLORS[a]}33` : "#111620", color: ACTION_COLORS[a], border: `1px solid ${ACTION_COLORS[a]}66` }}>
                          {a === "Block" ? "⊘ BLOCK" : a === "Warn" ? "⚠ WARN" : "◎ LOG"}
                        </button>
                      ))}
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button className="btn" onClick={() => handleAction("ForceAllow")} style={{ flex: 1, padding: "7px", background: "#0a1a0a", color: "#00e676", border: "1px solid #0a3020" }}>
                        ✓ FORCE ALLOW
                      </button>
                      <button className="btn" onClick={() => handleAction("ForceBlock")} style={{ flex: 1, padding: "7px", background: "#1a0a0a", color: "#ff3b3b", border: "1px solid #3a1010" }}>
                        ⊘ FORCE BLOCK
                      </button>
                    </div>
                    {selected.status && selected.status !== "pending" && (
                      <div style={{ marginTop: 8, padding: "6px 10px", background: "#0a0c10", borderRadius: 4, border: "1px solid #1a2030", fontSize: 10 }}>
                        Status: <span style={{ color: ACTION_COLORS[selected.status] || "#00e676", fontWeight: 700 }}>{selected.status.toUpperCase()}</span>
                      </div>
                    )}
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 6 }}>FEEDBACK & SEVERITY</div>
                    <textarea
                      value={feedbackNote}
                      onChange={e => setFeedbackNote(e.target.value)}
                      placeholder="Add notes or context..."
                      style={{ width: "100%", height: 64, background: "#0a0c10", border: "1px solid #1a2030", borderRadius: 4, padding: "8px", color: "#e2e8f0", fontSize: 11, fontFamily: "inherit", resize: "none", outline: "none" }}
                    />
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 9, color: "#4a6fa5", marginBottom: 4 }}>SEVERITY: {severityRating}/5</div>
                      <div style={{ display: "flex", gap: 4 }}>
                        {[1, 2, 3, 4, 5].map(n => (
                          <button key={n} className="btn" onClick={() => setSeverityRating(n)}
                            style={{ flex: 1, padding: "5px 0", background: n <= severityRating ? "#1a0a0a" : "#0a0c10", color: n <= severityRating ? "#ff3b3b" : "#4a6fa5", border: `1px solid ${n <= severityRating ? "#3a1010" : "#1a2030"}`, fontSize: 14 }}>
                            {n <= severityRating ? "★" : "☆"}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : <div style={{ fontSize: 11, color: "#4a6fa5" }}>Select a prompt to review</div>}
            </div>
          </div>
        </div>
      </div>

      {/* AUDIT TRAIL */}
      <div style={{ background: "#0d1117", borderTop: "1px solid #1a2030", padding: "12px 16px", maxHeight: 200, overflowY: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div className="section-title" style={{ marginBottom: 0 }}>◈ Audit Trail — {audit.length} entries</div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {exportMsg && <span style={{ fontSize: 9, color: "#00e676" }}>{exportMsg}</span>}
            <button className="btn" onClick={() => exportLogs("csv")} style={{ padding: "4px 10px", background: "#111620", color: "#4fc3f7", border: "1px solid #1e3a5f" }}>↓ CSV</button>
            <button className="btn" onClick={() => exportLogs("json")} style={{ padding: "4px 10px", background: "#111620", color: "#4fc3f7", border: "1px solid #1e3a5f" }}>↓ JSON</button>
          </div>
        </div>
        {audit.length === 0 ? (
          <div style={{ fontSize: 10, color: "#4a6fa5" }}>No actions taken yet. Review prompts and take action above.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {audit.map(a => (
              <div key={a.id} className="slide-in" style={{ display: "grid", gridTemplateColumns: "120px 1fr 80px 80px auto", gap: 10, alignItems: "center", padding: "5px 10px", background: "#0a0c10", borderRadius: 3, border: "1px solid #111620", fontSize: 10 }}>
                <span style={{ color: "#4a6fa5" }}>{timeAgo(a.ts)}</span>
                <span style={{ color: "#8a9ab0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>"{a.prompt_text}"</span>
                <span className="tag" style={{ background: `${ACTION_COLORS[a.action] || "#00e676"}22`, color: ACTION_COLORS[a.action] || "#00e676" }}>{a.action}</span>
                <span style={{ color: "#f5a623" }}>{"★".repeat(a.severity || 0)}{"☆".repeat(5 - (a.severity || 0))}</span>
                <span style={{ color: "#4a6fa5", maxWidth: 200, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{a.note || "—"}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}