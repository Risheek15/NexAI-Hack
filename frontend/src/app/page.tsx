"use client";

import React, { useState, useEffect } from "react";
import {
  Wrench,
  Plane,
  Search,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Database,
  BookOpen,
  Copy,
  Check,
  RefreshCw,
  Terminal
} from "lucide-react";

interface CitationDetail {
  record_id: string;
  problem_raw: string;
  action_raw: string;
  rank?: number;
}

interface ToolTraceItem {
  tool_name: string;
  arguments: Record<string, unknown>;
  output_summary: string;
  execution_time_ms: number;
  is_error: boolean;
}

interface GlossaryItem {
  expansion: string;
  status: string;
  frequency: number;
  sample?: string;
}

interface PolicyAudit {
  is_compliant: boolean;
  has_safety_disclaimer: boolean;
  conflicting_evidence_notes: string[];
  potential_causation_fallacies: string[];
  policy_summary: string;
}

interface ChatResponse {
  query: string;
  answer: string;
  evidence_status: "sufficient" | "insufficient" | "needs_clarification";
  evidence_reason: string;
  cited_record_ids: string[];
  citations: CitationDetail[];
  tool_trace: ToolTraceItem[];
  total_execution_time_ms: number;
  policy_audit: PolicyAudit;
  is_fallback: boolean;
  safety_disclaimer: string;
}

interface HealthData {
  status: string;
  total_records: number;
  total_glossary_terms: number;
  model_name: string;
  api_key_configured: boolean;
  environment: string;
}

const SAMPLE_QUERIES = [
  { label: "Cyl #2 Plug Fouled", query: "What actions were taken when cylinder 2 lower spark plug was fouled?", category: "Incident" },
  { label: "Engine Choke on Departure", query: "Had engine choke & briefly lose power on departure", category: "Discrepancy" },
  { label: "Baffle Crack Repair", query: "Stop drilled crack on L/H rear engine baffle", category: "Hardware" },
  { label: "Fuel Servo Replacement", query: "Removed and replaced fuel servo", category: "Component" },
  { label: "Ambiguous Squawk", query: "leak", category: "Clarification" },
  { label: "Out of Domain Query", query: "Boeing 777 autopilot computer failure and yaw damper disconnect", category: "Abstention" },
];

export default function MaintenanceCopilot() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<CitationDetail | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [glossary, setGlossary] = useState<Record<string, GlossaryItem>>({});
  const [showGlossary, setShowGlossary] = useState(false);

  // Poll backend health check
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/health");
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
      } catch (err) {
        console.warn("Backend currently unreachable:", err);
      }
    }
    checkHealth();
  }, []);

  // Fetch glossary on mount
  useEffect(() => {
    async function fetchGlossary() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/glossary");
        if (res.ok) {
          const data = await res.json();
          setGlossary(data);
        }
      } catch (err) {
        console.warn("Glossary fetch failed:", err);
      }
    }
    fetchGlossary();
  }, []);

  const handleSearch = async (targetQuery?: string) => {
    const q = targetQuery || query;
    if (!q.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setSelectedRecord(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5 }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned status ${res.status}`);
      }

      const data: ChatResponse = await res.json();
      setResponse(data);
      if (data.citations && data.citations.length > 0) {
        setSelectedRecord(data.citations[0]);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to communicate with Maintenance Copilot backend.");
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(text);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
              <Plane className="h-5 w-5 rotate-[-45deg]" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-white">Maintenance Copilot</span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  NEXAI 2026
                </span>
              </div>
              <p className="text-xs text-slate-400">Agentic RAG over 6,169 Aviation Maintenance Logbooks</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {health ? (
              <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-xs text-slate-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>{health.total_records.toLocaleString()} Records</span>
                <span className="text-slate-500">•</span>
                <span>{health.model_name}</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-xs text-amber-400">
                <RefreshCw className="h-3 w-3 animate-spin" />
                <span>Connecting to API...</span>
              </div>
            )}

            <button
              onClick={() => setShowGlossary(!showGlossary)}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              title="Aviation Acronym Glossary"
            >
              <BookOpen className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col space-y-8">
        {/* Search & Query Input Hero Section */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <label className="block text-sm font-semibold text-slate-300 mb-2 flex items-center justify-between">
            <span>Historical Logbook Query & Symptom Search</span>
            <span className="text-xs font-normal text-slate-400">Hybrid BM25 + Dense Semantic + RRF</span>
          </label>

          <div className="relative flex items-center">
            <Search className="absolute left-4 h-5 w-5 text-slate-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Describe maintenance squawk (e.g. 'cylinder 2 spark plug fouled', 'fuel servo leak', 'engine choke on departure')..."
              className="w-full pl-12 pr-32 py-3.5 bg-slate-950 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            />
            <button
              onClick={() => handleSearch()}
              disabled={loading || !query.trim()}
              className="absolute right-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-md text-sm font-medium transition flex items-center space-x-1.5 shadow"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  <span>Scanning...</span>
                </>
              ) : (
                <>
                  <Wrench className="h-4 w-4" />
                  <span>Investigate</span>
                </>
              )}
            </button>
          </div>

          {/* Quick Demo Sample Queries */}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-medium mr-1">Sample Queries:</span>
            {SAMPLE_QUERIES.map((sq, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setQuery(sq.query);
                  handleSearch(sq.query);
                }}
                className={`text-xs px-2.5 py-1 rounded-md border transition flex items-center space-x-1 ${
                  sq.category === "Abstention"
                    ? "bg-rose-950/40 border-rose-800/60 text-rose-300 hover:bg-rose-900/50"
                    : sq.category === "Clarification"
                    ? "bg-amber-950/40 border-amber-800/60 text-amber-300 hover:bg-amber-900/50"
                    : "bg-slate-800/80 border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white"
                }`}
              >
                <span>{sq.label}</span>
                <span className="text-[10px] opacity-60">({sq.category})</span>
              </button>
            ))}
          </div>
        </div>

        {/* Loading Skeleton */}
        {loading && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center space-y-4 animate-pulse">
            <div className="h-10 w-10 rounded-full bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
              <RefreshCw className="h-5 w-5 animate-spin" />
            </div>
            <div className="text-center">
              <p className="text-sm font-semibold text-slate-200">Executing Bounded Gemini Agent Loop...</p>
              <p className="text-xs text-slate-400 mt-1">
                Scanning 6,169 records via BM25 & Dense Vectors • Enforcing Evidence Gate • Validating Citations
              </p>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="bg-rose-950/30 border border-rose-800 rounded-xl p-4 flex items-start space-x-3 text-rose-300">
            <ShieldAlert className="h-5 w-5 flex-shrink-0 mt-0.5 text-rose-400" />
            <div>
              <p className="text-sm font-semibold">Investigation Error</p>
              <p className="text-xs mt-1 text-rose-200">{error}</p>
            </div>
          </div>
        )}

        {/* Results Presentation Area */}
        {response && !loading && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Center Area: Synthesis, Evidence Status, and Trace (2 cols) */}
            <div className="lg:col-span-2 space-y-6">
              {/* Evidence Status Card */}
              <div
                className={`border rounded-xl p-5 ${
                  response.evidence_status === "sufficient"
                    ? "bg-emerald-950/20 border-emerald-800/60"
                    : response.evidence_status === "needs_clarification"
                    ? "bg-sky-950/20 border-sky-800/60"
                    : "bg-amber-950/20 border-amber-800/60"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {response.evidence_status === "sufficient" ? (
                      <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                    ) : response.evidence_status === "needs_clarification" ? (
                      <HelpCircle className="h-5 w-5 text-sky-400" />
                    ) : (
                      <AlertTriangle className="h-5 w-5 text-amber-400" />
                    )}
                    <span
                      className={`text-sm font-bold uppercase tracking-wider ${
                        response.evidence_status === "sufficient"
                          ? "text-emerald-400"
                          : response.evidence_status === "needs_clarification"
                          ? "text-sky-400"
                          : "text-amber-400"
                      }`}
                    >
                      Evidence Assessment: {response.evidence_status.replace("_", " ")}
                    </span>
                  </div>

                  <span className="text-xs text-slate-400 font-mono">
                    Latency: {response.total_execution_time_ms.toFixed(1)} ms
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{response.evidence_reason}</p>
              </div>

              {/* Synthesized Answer Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h2 className="text-base font-semibold text-white flex items-center space-x-2">
                    <Wrench className="h-4 w-4 text-blue-400" />
                    <span>Historical Maintenance Analysis</span>
                  </h2>
                  {response.is_fallback && (
                    <span className="px-2 py-0.5 text-xs font-semibold rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      Fallback Mode
                    </span>
                  )}
                </div>

                {/* Formatted Answer Body */}
                <div className="prose prose-invert max-w-none text-sm text-slate-200 leading-relaxed space-y-3 whitespace-pre-wrap">
                  {response.answer}
                </div>

                {/* Mandatory Descriptive Safety Disclaimer Banner */}
                <div className="mt-4 p-4 rounded-lg bg-slate-950 border border-amber-600/30 text-xs text-amber-200/90 flex items-start space-x-3">
                  <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-amber-300">Descriptive Safety Notice: </span>
                    Historical logbook entries describe past actions recorded by technicians; they are NOT manufacturer-approved maintenance instructions or airworthiness directives. Always refer to the FAA-approved Aircraft Maintenance Manual (AMM) and applicable Service Bulletins.
                  </div>
                </div>

                {/* Policy Audit Status Pill */}
                <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center space-x-1.5">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                    <span>Citation Verification: {response.policy_audit.policy_summary}</span>
                  </div>
                  <span className="text-slate-500">{response.cited_record_ids.length} Citing Record(s)</span>
                </div>
              </div>

              {/* Agent Execution Trace Accordion */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow">
                <button
                  onClick={() => setTraceOpen(!traceOpen)}
                  className="w-full px-6 py-4 flex items-center justify-between bg-slate-900 hover:bg-slate-800/60 transition text-left text-sm font-semibold text-slate-300"
                >
                  <div className="flex items-center space-x-2">
                    <Terminal className="h-4 w-4 text-blue-400" />
                    <span>Agent Tool Execution Trace ({response.tool_trace.length} Calls)</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono text-slate-400">{response.total_execution_time_ms.toFixed(1)} ms total</span>
                    {traceOpen ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
                  </div>
                </button>

                {traceOpen && (
                  <div className="p-6 bg-slate-950 border-t border-slate-800 space-y-4">
                    {response.tool_trace.map((trace, idx) => (
                      <div key={idx} className="border border-slate-800 bg-slate-900 rounded-lg p-3.5 text-xs font-mono">
                        <div className="flex items-center justify-between text-slate-300 mb-1.5">
                          <span className="font-bold text-blue-400">Step {idx + 1}: {trace.tool_name}()</span>
                          <span className="text-slate-500">{trace.execution_time_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="text-slate-400 mb-1">
                          <span className="text-slate-500">Arguments: </span>
                          {JSON.stringify(trace.arguments)}
                        </div>
                        <div className="text-slate-300">
                          <span className="text-slate-500">Summary: </span>
                          {trace.output_summary}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Source Logbook Inspector (1 col) */}
            <div className="space-y-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <Database className="h-4 w-4 text-emerald-400" />
                    <span>Source Logbook Inspector</span>
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">
                    {response.citations.length} Verified Sources
                  </span>
                </div>

                {/* Citation Selection Buttons */}
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {response.citations.map((c) => (
                    <button
                      key={c.record_id}
                      onClick={() => setSelectedRecord(c)}
                      className={`px-2.5 py-1 text-xs font-mono rounded border transition ${
                        selectedRecord?.record_id === c.record_id
                          ? "bg-blue-600 border-blue-500 text-white font-bold"
                          : "bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-750"
                      }`}
                    >
                      [ID: {c.record_id}]
                    </button>
                  ))}
                </div>

                {/* Selected Record Detail Viewer */}
                {selectedRecord ? (
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-slate-400">IDENT:</span>
                        <span className="text-sm font-mono font-bold text-blue-400">{selectedRecord.record_id}</span>
                      </div>
                      <button
                        onClick={() => copyToClipboard(selectedRecord.record_id)}
                        className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
                        title="Copy Record ID"
                      >
                        {copiedId === selectedRecord.record_id ? (
                          <Check className="h-3.5 w-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )}
                      </button>
                    </div>

                    <div>
                      <span className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                        Reported Problem (Raw):
                      </span>
                      <p className="text-xs font-mono text-slate-200 bg-slate-900 p-2.5 rounded border border-slate-800 leading-relaxed">
                        {selectedRecord.problem_raw}
                      </p>
                    </div>

                    <div>
                      <span className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                        Action Taken by Technician (Raw):
                      </span>
                      <p className="text-xs font-mono text-emerald-300 bg-slate-900 p-2.5 rounded border border-slate-800 leading-relaxed">
                        {selectedRecord.action_raw}
                      </p>
                    </div>

                    <div className="pt-2 text-[11px] text-slate-500 italic">
                      Immutable source entry preserved exactly as recorded. Verified by post-generation validator.
                    </div>
                  </div>
                ) : (
                  <div className="p-8 text-center text-xs text-slate-500">
                    No citation selected. Click any citation ID above to inspect original squawk details.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Glossary Side Modal / Popover */}
        {showGlossary && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <h3 className="text-base font-semibold text-white flex items-center space-x-2">
                <BookOpen className="h-4 w-4 text-blue-400" />
                <span>Validated Aviation Maintenance Glossary</span>
              </h3>
              <button
                onClick={() => setShowGlossary(false)}
                className="text-xs text-slate-400 hover:text-white"
              >
                Close
              </button>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(glossary).map(([term, info]: [string, GlossaryItem]) => (
                <div key={term} className="bg-slate-950 border border-slate-800 rounded p-2 text-xs">
                  <div className="font-mono font-bold text-blue-400">{term}</div>
                  <div className="text-slate-300 truncate" title={info.expansion}>
                    {info.expansion}
                  </div>
                  <div className="text-[10px] text-slate-500">Freq: {info.frequency}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-600">
        Maintenance Copilot • NEXAI 2026 Hackathon MVP • Agentic RAG with Bounded Tool Execution & Evidence Policies
      </footer>
    </div>
  );
}
