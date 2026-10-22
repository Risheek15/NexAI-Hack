"use client";

import React, { useState, useEffect, useCallback } from "react";
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
  Terminal,
  User,
  ShieldCheck,
  PlusCircle,
  ClipboardList,
  CheckCircle,
  XCircle,
  Send,
  Clock,
  Lock,
  LogOut,
  Users,
  UserPlus,
  BarChart3,
  Sun,
  Moon,
  Trash2,
  RotateCcw,
  Filter,
  Calendar,
  X
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

interface PendingFixItem {
  id: number;
  temp_id: string;
  problem_raw: string;
  action_raw: string;
  problem_normalized: string;
  action_normalized: string;
  submitted_by: string;
  submitted_at: string;
  status: string;
  admin_notes?: string;
  approved_record_id?: string;
}

interface QueryLogItem {
  id: number;
  user_name: string;
  user_role: string;
  query: string;
  evidence_status: string;
  evidence_reason: string;
  cited_record_ids: string[];
  execution_time_ms: number;
  timestamp: string;
}

interface UserAccount {
  id: number;
  username: string;
  full_name: string;
  role: "admin" | "technician";
  created_at: string;
  last_active?: string | null;
}

function formatLastActive(timestamp: string | null | undefined): string {
  if (!timestamp) return "Never";
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    if (diffMs < 0) return "Just now";
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return String(timestamp);
  }
}

interface SystemStats {
  total_records: number;
  pending_fixes: number;
  total_technicians: number;
  total_queries: number;
}

interface ParsedAnswerRecord {
  title?: string;
  ids: string[];
  issue: string;
  fix: string;
}

interface ParsedStructuredAnswer {
  intro?: string;
  records: ParsedAnswerRecord[];
  fallbackBlocks: string[];
}

function parseAnswerText(answer: string): ParsedStructuredAnswer {
  if (!answer) return { records: [], fallbackBlocks: [] };

  // 1. Strip trailing safety disclaimers & notes
  let cleaned = answer
    .replace(/[\*\s]*\(?Note:?\s*Historical logbook entries describe past actions[\s\S]*$/i, "")
    .replace(/>\s*\[!NOTE\][\s\S]*$/i, "")
    .trim();

  // Strip introductory safety disclaimer if echoed
  cleaned = cleaned.replace(/^Historical maintenance records describe past actions.*?\.\s*/i, "").trim();

  // 2. Normalize common delimiters so that records split on double newlines
  const normalized = cleaned
    .replace(/\n(?=[\*\•]\s*(?:Record|\*\*))/gi, "\n\n")
    .replace(/\n(?=\d+\.\s+[\*\•]?\s*Record)/gi, "\n\n");

  const rawBlocks = normalized.split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);

  let intro: string | undefined = undefined;
  const records: ParsedAnswerRecord[] = [];
  const fallbackBlocks: string[] = [];

  for (let i = 0; i < rawBlocks.length; i++) {
    const block = rawBlocks[i];

    // Check for problem/issue and action/fix
    const issueMatch = block.match(
      /(?:^|\n)\s*[\*\-\•]?\s*\**\b(?:Problem Reported|Problem|Issue|Discrepancy)\b\**:?\s*([\s\S]*?)(?=\n\s*[\*\-\•]?\s*\**\b(?:Action Taken|Action|Fix|Corrective Action|Resolution)\b|$)/i
    );
    const fixMatch = block.match(
      /(?:^|\n)\s*[\*\-\•]?\s*\**\b(?:Action Taken|Action|Fix|Corrective Action|Resolution)\b\**:?\s*([\s\S]*)$/i
    );

    if (issueMatch && fixMatch) {
      // Extract genuine 6-digit record IDs
      const idMatches = Array.from(block.matchAll(/\b(\d{6})\b/g)).map((m) => m[1]);
      const uniqueIds = Array.from(new Set(idMatches));

      const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
      let title = "";
      if (lines.length > 0 && !/\b(?:Problem|Issue|Discrepancy|Action|Fix)\b/i.test(lines[0])) {
        title = lines[0].replace(/^[\s\*\-\•#:]+|[\s\*\-:]+$/g, "").trim();
      }

      const cleanIssue = issueMatch[1]
        .replace(/^[\s\*\-\•:]+/, "")
        .replace(/[\s\*\-]+$/, "")
        .trim();
      const cleanFix = fixMatch[1]
        .replace(/^[\s\*\-\•:]+/, "")
        .replace(/[\s\*\-]+$/, "")
        .trim();

      records.push({
        title: title || (uniqueIds.length > 0 ? `Record [ID: ${uniqueIds.join(", ")}]` : undefined),
        ids: uniqueIds,
        issue: cleanIssue,
        fix: cleanFix,
      });
    } else {
      // Intro paragraph detection
      if (
        i === 0 &&
        !block.startsWith("•") &&
        !block.startsWith("*") &&
        !block.includes("Record [ID:") &&
        !block.startsWith("###")
      ) {
        intro = block.replace(/^\**|\**$/g, "").trim();
      } else {
        fallbackBlocks.push(block);
      }
    }
  }

  return { intro, records, fallbackBlocks };
}

function renderTextWithCitations(
  text: string,
  citations: CitationDetail[],
  onSelectRecord?: (record: CitationDetail) => void,
  accentColor: "purple" | "cyan" = "purple"
) {
  if (!text) return null;

  // Split on [ID: xxxxxx] citations or bold markdown **...**
  const parts = text.split(/(\[ID:\s*(?:ID:\s*|\d{6}|[\s,])+\]|\*\*[^*]+\*\*)/gi);

  const badgeBg =
    accentColor === "cyan"
      ? "bg-cyan-500/15 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/30"
      : "bg-purple-500/15 text-purple-700 dark:text-purple-300 border-purple-500/30 hover:bg-purple-500/30";

  return parts.map((part, idx) => {
    if (!part) return null;

    // Check if it's a citation block [ID: ...]
    if (/^\[ID:/i.test(part.trim())) {
      const ids = Array.from(part.matchAll(/\b(\d{6})\b/g)).map((m) => m[1]);
      if (ids.length > 0) {
        return (
          <span key={idx} className="inline-flex items-center gap-1 mx-1 align-baseline flex-wrap">
            {ids.map((id) => (
              <button
                key={id}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  const found = citations.find((c) => c.record_id === id);
                  if (found) {
                    onSelectRecord?.(found);
                  } else {
                    onSelectRecord?.({
                      record_id: id,
                      problem_raw: "Historical Logbook Record",
                      action_raw: "Select to inspect logbook record details.",
                    });
                  }
                }}
                className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-mono font-semibold border transition cursor-pointer ${badgeBg}`}
                title={`Inspect record ${id} in Source Logbook Inspector`}
              >
                [ID: {id}]
              </button>
            ))}
          </span>
        );
      }
    }

    // Bold **text**
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={idx} className="font-semibold text-slate-900 dark:text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }

    // Clean stray * markers
    const cleaned = part.replace(/\*/g, "");
    return <span key={idx}>{cleaned}</span>;
  });
}

function StructuredAnswerRenderer({
  answer,
  citations,
  onSelectRecord,
  accentColor = "purple",
}: {
  answer: string;
  citations: CitationDetail[];
  onSelectRecord?: (record: CitationDetail) => void;
  accentColor?: "purple" | "cyan";
}) {
  const parsed = React.useMemo(() => parseAnswerText(answer), [answer]);

  const numBadgeBg =
    accentColor === "cyan"
      ? "bg-cyan-500/15 text-cyan-400 border-cyan-500/30"
      : "bg-purple-500/15 text-purple-600 dark:text-purple-400 border-purple-500/30";

  const idBadgeBg =
    accentColor === "cyan"
      ? "bg-cyan-500/10 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/25"
      : "bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/30 hover:bg-purple-500/25";

  // If no records parsed (e.g. clarification, refusal, or free text), render formatted fallback
  if (parsed.records.length === 0) {
    return (
      <div className="space-y-3">
        {parsed.intro && (
          <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-medium">
            {parsed.intro}
          </p>
        )}
        {parsed.fallbackBlocks.map((block, idx) => (
          <div key={idx} className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed whitespace-pre-wrap">
            {renderTextWithCitations(block, citations, onSelectRecord, accentColor)}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Intro Summary Sentence */}
      {parsed.intro && (
        <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-medium pb-1">
          {parsed.intro}
        </p>
      )}

      {/* Point-wise Cards */}
      <div className="space-y-3.5">
        {parsed.records.map((record, index) => (
          <div
            key={index}
            className="rounded-2xl border border-white/20 dark:border-white/10 bg-white/70 dark:bg-white/[0.03] backdrop-blur-md p-4 transition hover:border-purple-500/30 dark:hover:border-purple-400/30 space-y-3 shadow-sm"
          >
            {/* Header: Sequence Number + Title + Clickable Citation Badges */}
            <div className="flex items-center justify-between border-b border-black/5 dark:border-white/10 pb-2.5">
              <div className="flex items-center space-x-2">
                <span className={`inline-flex items-center justify-center h-5 w-5 rounded-full text-xs font-bold font-mono border ${numBadgeBg}`}>
                  {index + 1}
                </span>
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-800 dark:text-slate-200">
                  {record.title || "Historical Logbook Record"}
                </span>
              </div>

              {/* Top-right record citation buttons */}
              {record.ids.length > 0 && (
                <div className="flex items-center space-x-1.5">
                  {record.ids.map((recId) => (
                    <button
                      key={recId}
                      type="button"
                      onClick={() => {
                        const found = citations.find((c) => c.record_id === recId);
                        if (found) {
                          onSelectRecord?.(found);
                        } else {
                          onSelectRecord?.({
                            record_id: recId,
                            problem_raw: record.issue,
                            action_raw: record.fix,
                          });
                        }
                      }}
                      className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-lg text-xs font-mono font-semibold border transition cursor-pointer ${idBadgeBg}`}
                      title={`Inspect record ${recId} in Source Logbook Inspector`}
                    >
                      <Search className="h-3 w-3" />
                      <span>[ID: {recId}]</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Issue Section */}
            <div className="flex items-start space-x-3 rounded-xl bg-amber-500/[0.08] dark:bg-amber-500/[0.06] border border-amber-500/20 p-3">
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-md bg-amber-500/20 text-amber-800 dark:text-amber-300 border border-amber-500/30 flex-shrink-0 mt-0.5">
                Issue
              </span>
              <div className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed flex-1">
                {renderTextWithCitations(record.issue, citations, onSelectRecord, accentColor)}
              </div>
            </div>

            {/* Fix Section */}
            <div className="flex items-start space-x-3 rounded-xl bg-emerald-500/[0.08] dark:bg-emerald-500/[0.06] border border-emerald-500/20 p-3">
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-md bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border border-emerald-500/30 flex-shrink-0 mt-0.5">
                Fix
              </span>
              <div className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed flex-1">
                {renderTextWithCitations(record.fix, citations, onSelectRecord, accentColor)}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Any remaining non-card blocks */}
      {parsed.fallbackBlocks.length > 0 && (
        <div className="pt-2 space-y-2">
          {parsed.fallbackBlocks.map((b, idx) => (
            <div key={idx} className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              {renderTextWithCitations(b, citations, onSelectRecord, accentColor)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function MaintenanceCopilot() {
  // Theme State (Dark / Light)
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Authentication State
  const [currentUser, setCurrentUser] = useState<UserAccount | null>(null);
  const [loginUsername, setLoginUsername] = useState<string>("");
  const [loginPassword, setLoginPassword] = useState<string>("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loggingIn, setLoggingIn] = useState<boolean>(false);

  // Admin Vertical Dashboard Selection Tab
  const [activeAdminTab, setActiveAdminTab] = useState<"investigate" | "technicians" | "pending" | "audit">("investigate");

  // System Stats
  const [stats, setStats] = useState<SystemStats | null>(null);

  // Query & Chat State
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<CitationDetail | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [glossary, setGlossary] = useState<Record<string, GlossaryItem>>({});
  const [showGlossary, setShowGlossary] = useState(false);
  const [showDatabase, setShowDatabase] = useState(false);

  // Technician Submission Modal State
  const [showSubmitFixModal, setShowSubmitFixModal] = useState<boolean>(false);
  const [submittingFix, setSubmittingFix] = useState<boolean>(false);
  const [fixProblem, setFixProblem] = useState<string>("");
  const [fixAction, setFixAction] = useState<string>("");
  const [submitSuccess, setSubmitSuccess] = useState<string | null>(null);

  // Admin User Creation & Management State
  const [technicians, setTechnicians] = useState<UserAccount[]>([]);
  const [loadingTechs, setLoadingTechs] = useState<boolean>(false);
  const [newTechUser, setNewTechUser] = useState<string>("");
  const [newTechPass, setNewTechPass] = useState<string>("");
  const [newTechName, setNewTechName] = useState<string>("");
  const [creatingTech, setCreatingTech] = useState<boolean>(false);
  const [techActionMsg, setTechActionMsg] = useState<string | null>(null);

  // Admin Pending Queue & Audit Logs State
  const [pendingFixes, setPendingFixes] = useState<PendingFixItem[]>([]);
  const [loadingPending, setLoadingPending] = useState<boolean>(false);
  const [adminActionStatus, setAdminActionStatus] = useState<string | null>(null);
  const [auditLogs, setAuditLogs] = useState<QueryLogItem[]>([]);
  const [loadingAudit, setLoadingAudit] = useState<boolean>(false);
  const [deletingTechId, setDeletingTechId] = useState<number | null>(null);

  // Admin Audit Logs Filters
  const [auditTechFilter, setAuditTechFilter] = useState<string>("all");
  const [auditDateFilter, setAuditDateFilter] = useState<string>("");
  const [auditQueryFilter, setAuditQueryFilter] = useState<string>("");
  const [auditStatusFilter, setAuditStatusFilter] = useState<string>("all");

  // Technician Personal History State & View
  const [techView, setTechView] = useState<"search" | "history">("search");
  const [myQueryLogs, setMyQueryLogs] = useState<QueryLogItem[]>([]);
  const [loadingMyHistory, setLoadingMyHistory] = useState<boolean>(false);

  // Technician Personal History Filters
  const [myHistoryDateFilter, setMyHistoryDateFilter] = useState<string>("");
  const [myHistoryQueryFilter, setMyHistoryQueryFilter] = useState<string>("");
  const [myHistoryStatusFilter, setMyHistoryStatusFilter] = useState<string>("all");

  // Bottom Glossary Filter
  const [glossaryFilter, setGlossaryFilter] = useState<string>("");

  // Initialize theme from localStorage
  useEffect(() => {
    try {
      const savedTheme = (localStorage.getItem("copilot_theme") as "dark" | "light") || "dark";
      setTheme(savedTheme);
      document.documentElement.classList.toggle("dark", savedTheme === "dark");
    } catch {
      // Ignore
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("copilot_theme", nextTheme);
    document.documentElement.classList.toggle("dark", nextTheme === "dark");
  };

  // Initialize user session from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("copilot_user");
      if (saved) {
        setCurrentUser(JSON.parse(saved));
      }
    } catch {
      // Ignore parse error
    }
  }, []);

  // Fetch live system stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/stats");
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) {
      console.warn("Failed to fetch system stats:", err);
    }
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
    fetchStats();
  }, [fetchStats]);

  // Fetch technicians list (Admin only)
  const fetchTechnicians = useCallback(async () => {
    setLoadingTechs(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/users");
      if (res.ok) {
        const data = await res.json();
        setTechnicians(data);
      }
    } catch (err) {
      console.warn("Failed to fetch technicians:", err);
    } finally {
      setLoadingTechs(false);
    }
  }, []);

  // Fetch pending fixes (Admin only)
  const fetchPendingFixes = useCallback(async () => {
    setLoadingPending(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/pending?status_filter=pending");
      if (res.ok) {
        const data = await res.json();
        setPendingFixes(data);
      }
    } catch (err) {
      console.warn("Failed to fetch pending fixes:", err);
    } finally {
      setLoadingPending(false);
    }
  }, []);

  // Fetch audit logs (Admin only)
  const fetchAuditLogs = useCallback(async () => {
    setLoadingAudit(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/audit-logs?limit=100");
      if (res.ok) {
        const data = await res.json();
        setAuditLogs(data);
      }
    } catch (err) {
      console.warn("Failed to fetch audit logs:", err);
    } finally {
      setLoadingAudit(false);
    }
  }, []);

  // Fetch personal query history (Technician only)
  const fetchMyHistory = useCallback(async () => {
    if (!currentUser) return;
    setLoadingMyHistory(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/user/query-logs?user_name=${encodeURIComponent(currentUser.full_name)}`);
      if (res.ok) {
        const data = await res.json();
        setMyQueryLogs(data);
      }
    } catch (err) {
      console.warn("Failed to fetch personal query history:", err);
    } finally {
      setLoadingMyHistory(false);
    }
  }, [currentUser]);

  // Delete technician account (Admin only)
  const handleDeleteTechnician = async (userId: number, username: string) => {
    if (!confirm(`Are you sure you want to delete technician account '${username}'?`)) return;
    setDeletingTechId(userId);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/users/${userId}`, {
        method: "DELETE"
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "Failed to delete technician");
      }
      fetchTechnicians();
      fetchStats();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Error deleting technician account");
    } finally {
      setDeletingTechId(null);
    }
  };

  // Admin clicks technician to view their audit logs with filter pre-applied
  const handleViewTechnicianLogs = (techUsername: string) => {
    setAuditTechFilter(techUsername);
    setAuditDateFilter("");
    setAuditQueryFilter("");
    setAuditStatusFilter("all");
    setActiveAdminTab("audit");
  };

  useEffect(() => {
    if (currentUser?.role === "admin") {
      fetchStats();
      fetchTechnicians();
      fetchPendingFixes();
      fetchAuditLogs();
    } else if (currentUser?.role === "technician") {
      fetchMyHistory();
    }
  }, [currentUser, fetchStats, fetchTechnicians, fetchPendingFixes, fetchAuditLogs, fetchMyHistory]);

  // Real-time automatic polling for live entries & audit updates (every 3 seconds)
  useEffect(() => {
    if (!currentUser) return;

    const interval = setInterval(() => {
      // 1. Keep stats fresh
      fetchStats();

      // 2. If admin is on audit logs or pending fixes, refresh live
      if (currentUser.role === "admin") {
        if (activeAdminTab === "audit") {
          fetchAuditLogs();
        } else if (activeAdminTab === "pending") {
          fetchPendingFixes();
        }
      }

      // 3. If technician is viewing history, refresh live
      if (currentUser.role === "technician" && techView === "history") {
        fetchMyHistory();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [currentUser, activeAdminTab, techView, fetchStats, fetchAuditLogs, fetchPendingFixes, fetchMyHistory]);

  // Handle Login
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginUsername.trim() || !loginPassword.trim() || loggingIn) return;

    setLoggingIn(true);
    setLoginError(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: loginUsername.trim(),
          password: loginPassword.trim()
        })
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.message || "Invalid username or password.");
      }

      setCurrentUser(data.user);
      localStorage.setItem("copilot_user", JSON.stringify(data.user));
      setLoginUsername("");
      setLoginPassword("");
    } catch (err) {
      setLoginError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setLoggingIn(false);
    }
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem("copilot_user");
    setResponse(null);
    setError(null);
  };

  // Handle Search
  const handleSearch = async () => {
    if (!query.trim() || loading || !currentUser) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setSelectedRecord(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          top_k: 5,
          user_name: currentUser.full_name,
          user_role: currentUser.role
        }),
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

      fetchStats();
      if (currentUser.role === "admin") {
        fetchAuditLogs();
      } else if (currentUser.role === "technician") {
        fetchMyHistory();
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to communicate with Maintenance Copilot backend.");
    } finally {
      setLoading(false);
    }
  };

  // Technician Submits Fix
  const handleFixSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fixProblem.trim() || !fixAction.trim() || submittingFix || !currentUser) return;

    setSubmittingFix(true);
    setSubmitSuccess(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/records/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_raw: fixProblem.trim(),
          action_raw: fixAction.trim(),
          submitted_by: currentUser.full_name
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Submission failed");
      }

      const data = await res.json();
      setSubmitSuccess(`Fix successfully queued for Administrator approval (Tracking ID: ${data.temp_id}).`);
      setFixProblem("");
      setFixAction("");
      fetchStats();
      if (currentUser.role === "admin") {
        fetchPendingFixes();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit maintenance fix.");
    } finally {
      setSubmittingFix(false);
    }
  };

  // Admin Approves Fix
  const handleApprove = async (tempId: string) => {
    if (!currentUser) return;
    setAdminActionStatus(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/records/${tempId}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ admin_name: currentUser.full_name })
      });
      if (!res.ok) {
        throw new Error("Failed to approve record");
      }
      const data = await res.json();
      setAdminActionStatus(`Approved! Fix assigned official ID #${data.record_id} and dynamically indexed into fleet logbook.`);
      fetchPendingFixes();
      fetchStats();
    } catch (err) {
      setAdminActionStatus(`Approval error: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  // Admin Rejects Fix
  const handleReject = async (tempId: string) => {
    if (!currentUser) return;
    setAdminActionStatus(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/records/${tempId}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          admin_name: currentUser.full_name,
          reason: "Insufficient discrepancy details or lacks AMM reference"
        })
      });
      if (!res.ok) {
        throw new Error("Failed to reject record");
      }
      setAdminActionStatus(`Request ${tempId} rejected.`);
      fetchPendingFixes();
      fetchStats();
    } catch (err) {
      setAdminActionStatus(`Rejection error: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  // Admin Creates Technician Account
  const handleCreateTechnician = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTechUser.trim() || !newTechPass.trim() || !newTechName.trim() || creatingTech) return;

    setCreatingTech(true);
    setTechActionMsg(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: newTechUser.trim(),
          password: newTechPass.trim(),
          full_name: newTechName.trim(),
          role: "technician"
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Account creation failed.");
      }

      setTechActionMsg(`Technician account '${newTechUser}' created successfully.`);
      setNewTechUser("");
      setNewTechPass("");
      setNewTechName("");
      fetchTechnicians();
      fetchStats();
    } catch (err) {
      setTechActionMsg(`Error: ${err instanceof Error ? err.message : "Creation failed."}`);
    } finally {
      setCreatingTech(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(text);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // --- VIEW: LOGIN SCREEN (When not signed in) ---
  if (!currentUser) {
    return (
      <div className="relative min-h-screen bg-slate-50 dark:bg-black text-slate-900 dark:text-slate-100 flex flex-col justify-center items-center px-4 py-8 transition-colors select-none">
        {/* Top Right Clean Theme Toggle */}
        <div className="fixed top-6 right-6 z-20">
          <button
            onClick={toggleTheme}
            className="p-2.5 rounded-full border border-slate-200/80 dark:border-white/10 bg-white/80 dark:bg-neutral-900/80 backdrop-blur-md text-slate-700 dark:text-slate-300 hover:scale-105 active:scale-95 transition shadow-sm"
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === "dark" ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-slate-700" />}
          </button>
        </div>

        {/* Revamped Liquid Glass Login Card */}
        <div className="relative z-10 max-w-lg w-full border border-slate-200/80 dark:border-white/[0.12] bg-white dark:bg-neutral-950/80 backdrop-blur-2xl rounded-3xl p-8 sm:p-12 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.08)] dark:shadow-[0_25px_70px_-15px_rgba(0,0,0,0.95)] transition-all">
          {/* Header */}
          <div className="text-center space-y-4 mb-8">
            <div className="h-16 w-16 rounded-2xl border border-slate-200 dark:border-white/15 bg-slate-100 dark:bg-neutral-900 flex items-center justify-center text-slate-800 dark:text-white mx-auto shadow-inner transition">
              <Plane className="h-7 w-7 rotate-[-45deg] stroke-[1.75]" />
            </div>
            <div className="space-y-1">
              <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900 dark:text-white">
                Maintenance Copilot
              </h1>
              <p className="text-xs text-slate-500 dark:text-neutral-400 font-normal tracking-wide">
                Aviation Fleet Knowledge & Incident Intelligence
              </p>
            </div>
          </div>

          {loginError && (
            <div className="mb-6 bg-rose-500/10 border border-rose-500/25 rounded-xl p-3 text-xs text-rose-600 dark:text-rose-400 flex items-center space-x-2">
              <ShieldAlert className="h-4 w-4 text-rose-500 flex-shrink-0" />
              <span>{loginError}</span>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div className="space-y-1.5 text-left">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-neutral-400">
                Username
              </label>
              <input
                type="text"
                required
                value={loginUsername}
                onChange={(e) => setLoginUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full px-4 py-3.5 rounded-xl border border-slate-200 dark:border-neutral-800 bg-slate-50/70 dark:bg-neutral-900/60 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-neutral-500 text-sm focus:outline-none focus:border-slate-400 dark:focus:border-neutral-600 focus:ring-1 focus:ring-slate-400 dark:focus:ring-neutral-600 transition text-left"
              />
            </div>

            <div className="space-y-1.5 text-left">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-neutral-400">
                Password
              </label>
              <input
                type="password"
                required
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                placeholder="Enter password"
                className="w-full px-4 py-3.5 rounded-xl border border-slate-200 dark:border-neutral-800 bg-slate-50/70 dark:bg-neutral-900/60 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-neutral-500 text-sm focus:outline-none focus:border-slate-400 dark:focus:border-neutral-600 focus:ring-1 focus:ring-slate-400 dark:focus:ring-neutral-600 transition text-left"
              />
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={loggingIn || !loginUsername.trim() || !loginPassword.trim()}
                className="w-full py-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-black font-semibold text-sm transition active:scale-[0.99] disabled:opacity-40 disabled:pointer-events-none flex items-center justify-center space-x-2 shadow-sm"
              >
                {loggingIn ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin text-slate-400 dark:text-neutral-600" />
                    <span>Signing In...</span>
                  </>
                ) : (
                  <span>Sign In</span>
                )}
              </button>
            </div>
          </form>

          <div className="mt-8 pt-6 border-t border-slate-100 dark:border-white/[0.06] text-center">
            <p className="text-[11px] text-slate-400 dark:text-neutral-500">
              Authorized aircraft technicians & airport administrators
            </p>
          </div>
        </div>
      </div>
    );
  }

  // --- VIEW: AUTHENTICATED PORTAL ---
  return (
    <div className="relative min-h-screen bg-slate-50 dark:bg-black text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors overflow-x-hidden">

      {/* Floating Liquid Glass Top Header */}
      <header className="sticky top-3 z-40 max-w-7xl w-[calc(100%-1.5rem)] mx-auto px-5 h-16 liquid-glass rounded-2xl flex items-center justify-between shadow-xl">
        <div className="flex items-center space-x-3">
          <div className="h-10 w-10 rounded-xl liquid-pill-blue flex items-center justify-center text-white shadow-md">
            <Plane className="h-5 w-5 rotate-[-45deg]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-base tracking-tight text-slate-900 dark:text-white">Maintenance Copilot</span>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">Aviation Fleet Maintenance Knowledge System</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Active User Liquid Capsule */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full liquid-glass-subtle text-xs">
            {currentUser.role === "admin" ? (
              <ShieldCheck className="h-4 w-4 text-purple-600 dark:text-purple-400" />
            ) : (
              <User className="h-4 w-4 text-cyan-600 dark:text-cyan-400" />
            )}
            <span className="font-semibold text-slate-800 dark:text-slate-200">{currentUser.full_name}</span>
            <span
              className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold tracking-wider ${
                currentUser.role === "admin"
                  ? "bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/30"
                  : "bg-cyan-500/20 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30"
              }`}
            >
              {currentUser.role === "admin" ? "Airport Admin" : "Technician"}
            </span>
          </div>

          {/* Technician: My History & Submit Fix Buttons */}
          {currentUser.role === "technician" && (
            <>
              <button
                onClick={() => {
                  const nextView = techView === "history" ? "search" : "history";
                  setTechView(nextView);
                  if (nextView === "history") {
                    fetchMyHistory();
                  }
                }}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition ${
                  techView === "history"
                    ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                    : "liquid-glass-subtle text-slate-700 dark:text-slate-300 hover:scale-[1.02]"
                }`}
                title="View your query history"
              >
                <Clock className="h-3.5 w-3.5" />
                <span>My History</span>
              </button>

              <button
                onClick={() => {
                  setSubmitSuccess(null);
                  setShowSubmitFixModal(true);
                }}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-lg shadow-emerald-600/30 transition hover:scale-[1.02] active:scale-[0.98]"
              >
                <PlusCircle className="h-3.5 w-3.5" />
                <span>Submit Fix</span>
              </button>
            </>
          )}

          {/* Theme Toggle (Light / Dark) */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl liquid-glass-subtle hover:scale-105 active:scale-95 text-slate-700 dark:text-slate-300 transition"
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === "dark" ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-slate-700 dark:text-slate-300" />}
          </button>

          {/* Log Out */}
          <button
            onClick={handleLogout}
            className="p-2 rounded-xl liquid-glass-subtle hover:scale-105 active:scale-95 text-slate-400 hover:text-rose-500 dark:hover:text-rose-400 transition"
            title="Log Out"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Technician Submit Fix Modal */}
      {showSubmitFixModal && (
        <div className="fixed inset-0 z-50 bg-black/60 dark:bg-black/75 backdrop-blur-md flex items-center justify-center p-4">
          <div className="liquid-glass rounded-3xl max-w-xl w-full p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-3">
              <div className="flex items-center space-x-2">
                <PlusCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                <h3 className="text-base font-bold text-slate-900 dark:text-white">Submit Maintenance Fix Request</h3>
              </div>
              <button
                onClick={() => setShowSubmitFixModal(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-white text-xs"
              >
                ✕
              </button>
            </div>

            {submitSuccess ? (
              <div className="bg-emerald-500/15 border border-emerald-500/30 rounded-2xl p-4 text-emerald-800 dark:text-emerald-300 space-y-3">
                <div className="flex items-center space-x-2 font-bold text-sm">
                  <CheckCircle className="h-4 w-4" />
                  <span>Submission Successful</span>
                </div>
                <p className="text-xs leading-relaxed">{submitSuccess}</p>
                <div className="pt-2">
                  <button
                    onClick={() => setShowSubmitFixModal(false)}
                    className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold transition"
                  >
                    Done
                  </button>
                </div>
              </div>
            ) : (
              <form onSubmit={handleFixSubmit} className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
                    Observed Discrepancy / Squawk (Problem Raw):
                  </label>
                  <textarea
                    rows={3}
                    value={fixProblem}
                    onChange={(e) => setFixProblem(e.target.value)}
                    required
                    placeholder="e.g. CYL #3 EXHAUST VALVE STICKING DURING RUN-UP. HIGH EGT OBSERVED."
                    className="w-full px-3.5 py-2.5 liquid-glass-input rounded-2xl text-xs font-mono text-slate-900 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
                    Corrective Maintenance Action Taken (Action Raw):
                  </label>
                  <textarea
                    rows={3}
                    value={fixAction}
                    onChange={(e) => setFixAction(e.target.value)}
                    required
                    placeholder="e.g. REMOVED CYLINDER #3. REAMED EXHAUST VALVE GUIDE PER AMM 72-30-04. REINSTALLED WITH NEW GASKETS. OPS TEST NORMAL."
                    className="w-full px-3.5 py-2.5 liquid-glass-input rounded-2xl text-xs font-mono text-slate-900 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                  />
                </div>

                <div className="p-3.5 rounded-2xl liquid-glass-subtle text-[11px] text-slate-600 dark:text-slate-400">
                  <span className="font-semibold text-slate-800 dark:text-slate-300">Governance Notice: </span>
                  Submitted fixes enter the pending queue. Once reviewed and approved by an Airport Administrator, this fix will receive an official ID and be indexed into active retrieval.
                </div>

                <div className="flex justify-end space-x-2 pt-2 border-t border-white/20 dark:border-white/10">
                  <button
                    type="button"
                    onClick={() => setShowSubmitFixModal(false)}
                    className="px-4 py-2 liquid-glass-subtle text-slate-700 dark:text-slate-300 rounded-xl text-xs font-semibold transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={submittingFix || !fixProblem.trim() || !fixAction.trim()}
                    className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold transition flex items-center space-x-1.5 shadow-lg shadow-emerald-600/30"
                  >
                    {submittingFix ? (
                      <>
                        <RefreshCw className="h-3 w-3 animate-spin" />
                        <span>Submitting...</span>
                      </>
                    ) : (
                      <>
                        <Send className="h-3 w-3" />
                        <span>Submit for Approval</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}

      {/* Main Container */}
      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col space-y-6">
        {/* --- AIRPORT ADMINISTRATOR VIEW: LEFT VERTICAL SIDEBAR + CONTENT --- */}
        {currentUser.role === "admin" ? (
          <div className="flex flex-col lg:flex-row gap-6 items-start">
            {/* Left Vertical Navigation Rail (Liquid Glass Sidebar) */}
            <aside className="w-full lg:w-64 liquid-glass rounded-3xl p-5 shadow-2xl flex flex-col justify-between space-y-6 flex-shrink-0">
              <div className="space-y-4">
                <div className="px-3 py-2 border-b border-white/20 dark:border-white/10">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 dark:text-slate-500">
                    Administrator Controls
                  </span>
                  <div className="text-xs font-bold text-slate-800 dark:text-slate-200 mt-0.5">
                    Operations Dashboard
                  </div>
                </div>

                <nav className="space-y-1.5">
                  <button
                    onClick={() => setActiveAdminTab("investigate")}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl text-xs font-semibold transition text-left ${
                      activeAdminTab === "investigate"
                        ? "liquid-pill-active text-white"
                        : "text-slate-700 dark:text-slate-300 hover:bg-white/40 dark:hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <Search className="h-4 w-4" />
                      <span>Copilot Search</span>
                    </div>
                  </button>

                  <button
                    onClick={() => {
                      setActiveAdminTab("technicians");
                      fetchTechnicians();
                    }}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl text-xs font-semibold transition text-left ${
                      activeAdminTab === "technicians"
                        ? "liquid-pill-active text-white"
                        : "text-slate-700 dark:text-slate-300 hover:bg-white/40 dark:hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <Users className="h-4 w-4" />
                      <span>Technicians</span>
                    </div>
                    {stats && (
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        activeAdminTab === "technicians"
                          ? "bg-white/30 text-white"
                          : "bg-white/60 dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                      }`}>
                        {stats.total_technicians}
                      </span>
                    )}
                  </button>

                  <button
                    onClick={() => {
                      setActiveAdminTab("pending");
                      fetchPendingFixes();
                    }}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl text-xs font-semibold transition text-left ${
                      activeAdminTab === "pending"
                        ? "liquid-pill-active text-white"
                        : "text-slate-700 dark:text-slate-300 hover:bg-white/40 dark:hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <ClipboardList className="h-4 w-4" />
                      <span>Pending Fixes</span>
                    </div>
                    {pendingFixes.length > 0 && (
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500 text-slate-950 shadow-md">
                        {pendingFixes.length}
                      </span>
                    )}
                  </button>

                  <button
                    onClick={() => {
                      setActiveAdminTab("audit");
                      fetchAuditLogs();
                    }}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl text-xs font-semibold transition text-left ${
                      activeAdminTab === "audit"
                        ? "liquid-pill-active text-white"
                        : "text-slate-700 dark:text-slate-300 hover:bg-white/40 dark:hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <Clock className="h-4 w-4" />
                      <span>Query History</span>
                    </div>
                    {stats && (
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono ${
                        activeAdminTab === "audit"
                          ? "bg-white/30 text-white"
                          : "bg-white/60 dark:bg-slate-800 text-slate-700 dark:text-slate-400"
                      }`}>
                        {stats.total_queries}
                      </span>
                    )}
                  </button>
                </nav>
              </div>
            </aside>

            {/* Right Main Content Area */}
            <div className="flex-1 w-full space-y-6">
              {/* Dynamic KPI Bar in Liquid Glass */}
              {stats && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="liquid-glass rounded-2xl p-4 flex items-center space-x-3.5 hover:scale-[1.01] transition shadow-md">
                    <div className="p-3 rounded-xl bg-blue-500/15 text-blue-600 dark:text-blue-400 border border-blue-500/30">
                      <Database className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-xl font-bold text-slate-900 dark:text-white">{stats.total_records.toLocaleString()}</div>
                      <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold">Fleet Records</div>
                    </div>
                  </div>

                  <div className="liquid-glass rounded-2xl p-4 flex items-center space-x-3.5 hover:scale-[1.01] transition shadow-md">
                    <div className="p-3 rounded-xl bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                      <Users className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-xl font-bold text-slate-900 dark:text-white">{stats.total_technicians}</div>
                      <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold">Technicians</div>
                    </div>
                  </div>

                  <div className="liquid-glass rounded-2xl p-4 flex items-center space-x-3.5 hover:scale-[1.01] transition shadow-md">
                    <div className="p-3 rounded-xl bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                      <ClipboardList className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-xl font-bold text-slate-900 dark:text-white">{stats.pending_fixes}</div>
                      <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold">Pending Approvals</div>
                    </div>
                  </div>

                  <div className="liquid-glass rounded-2xl p-4 flex items-center space-x-3.5 hover:scale-[1.01] transition shadow-md">
                    <div className="p-3 rounded-xl bg-purple-500/15 text-purple-600 dark:text-purple-400 border border-purple-500/30">
                      <BarChart3 className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-xl font-bold text-slate-900 dark:text-white">{stats.total_queries}</div>
                      <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold">Audited Queries</div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 1: COPILOT SEARCH */}
              {activeAdminTab === "investigate" && (
                <>
                  <div className="liquid-glass rounded-3xl p-6 shadow-xl">
                    <label className="block text-sm font-semibold text-slate-800 dark:text-slate-300 mb-2.5 flex items-center justify-between">
                      <span>Historical Logbook Query & Symptom Search</span>
                      <span className="text-xs font-normal text-slate-500 dark:text-slate-400">
                        Hybrid BM25 + Dense Semantic RAG
                      </span>
                    </label>

                    <div className="relative flex items-center">
                      <Search className="absolute left-4 h-5 w-5 text-slate-400" />
                      <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                        placeholder="Describe maintenance squawk or discrepancy (e.g. 'cylinder 2 spark plug fouled', 'fuel servo leak')..."
                        className="w-full pl-12 pr-44 sm:pr-48 py-3.5 liquid-glass-input rounded-2xl text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 text-sm"
                      />
                      <button
                        onClick={handleSearch}
                        disabled={loading || !query.trim()}
                        className="absolute right-2 px-4 py-2 liquid-pill-active hover:brightness-110 active:scale-95 disabled:opacity-50 text-white rounded-xl text-sm font-medium transition flex items-center space-x-1.5 shadow"
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
                  </div>

                  {/* Loading Skeleton */}
                  {loading && (
                    <div className="liquid-glass rounded-3xl p-8 flex flex-col items-center justify-center space-y-4 animate-pulse">
                      <div className="h-12 w-12 rounded-2xl liquid-pill-active flex items-center justify-center text-white">
                        <RefreshCw className="h-6 w-6 animate-spin" />
                      </div>
                      <div className="text-center">
                        <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">Executing Bounded Agentic Search...</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                          Matching across logbook database • Enforcing Evidence Gate • Validating Citations
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Error Alert */}
                  {error && (
                    <div className="bg-rose-500/15 border border-rose-500/30 backdrop-blur-md rounded-2xl p-4 flex items-start space-x-3 text-rose-800 dark:text-rose-300">
                      <ShieldAlert className="h-5 w-5 flex-shrink-0 mt-0.5 text-rose-500" />
                      <div>
                        <p className="text-sm font-semibold">Investigation Error</p>
                        <p className="text-xs mt-1 text-rose-700 dark:text-rose-200">{error}</p>
                      </div>
                    </div>
                  )}

                  {/* Search Results */}
                  {response && !loading && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      <div className="lg:col-span-2 space-y-6">
                        {/* Evidence Status Card */}
                        <div
                          className={`rounded-2xl p-5 border shadow-md backdrop-blur-xl ${
                            response.evidence_status === "sufficient"
                              ? "bg-emerald-500/10 border-emerald-500/30"
                              : response.evidence_status === "needs_clarification"
                              ? "bg-sky-500/10 border-sky-500/30"
                              : "bg-amber-500/10 border-amber-500/30"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              {response.evidence_status === "sufficient" ? (
                                <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                              ) : response.evidence_status === "needs_clarification" ? (
                                <HelpCircle className="h-5 w-5 text-sky-600 dark:text-sky-400" />
                              ) : (
                                <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                              )}
                              <span
                                className={`text-sm font-bold uppercase tracking-wider ${
                                  response.evidence_status === "sufficient"
                                    ? "text-emerald-700 dark:text-emerald-400"
                                    : response.evidence_status === "needs_clarification"
                                    ? "text-sky-700 dark:text-sky-400"
                                    : "text-amber-700 dark:text-amber-400"
                                }`}
                              >
                                Evidence Assessment: {response.evidence_status.replace("_", " ")}
                              </span>
                            </div>

                            <span className="text-xs text-slate-500 font-mono">
                              Latency: {response.total_execution_time_ms.toFixed(1)} ms
                            </span>
                          </div>
                          <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">{response.evidence_reason}</p>
                        </div>

                        {/* Synthesized Answer Card */}
                        <div className="liquid-glass rounded-3xl p-6 shadow-xl space-y-4">
                          <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-3">
                            <h2 className="text-base font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
                              <Wrench className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                              <span>Historical Maintenance Analysis</span>
                            </h2>
                            {response.is_fallback && (
                              <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30">
                                Fallback Mode
                              </span>
                            )}
                          </div>

                          <div className="max-w-none">
                            <StructuredAnswerRenderer
                              answer={response.answer}
                              citations={response.citations}
                              onSelectRecord={setSelectedRecord}
                              accentColor="purple"
                            />
                          </div>

                          <div className="mt-4 p-4 rounded-2xl bg-amber-500/10 border border-amber-500/25 text-xs text-amber-800 dark:text-amber-200/90 flex items-start space-x-3">
                            <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                            <div>
                              <span className="font-semibold text-amber-900 dark:text-amber-300">Descriptive Safety Notice: </span>
                              Historical logbook entries describe past actions recorded by technicians; they are NOT manufacturer-approved maintenance instructions. Always refer to the FAA-approved AMM and applicable Service Bulletins.
                            </div>
                          </div>

                          <div className="pt-3 border-t border-white/20 dark:border-white/10 flex items-center justify-between text-xs text-slate-500">
                            <div className="flex items-center space-x-1.5">
                              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                              <span>Citation Verification: {response.policy_audit.policy_summary}</span>
                            </div>
                            <span>{response.cited_record_ids.length} Citing Record(s)</span>
                          </div>
                        </div>

                        {/* Tool Trace Accordion */}
                        <div className="liquid-glass rounded-2xl overflow-hidden shadow-md">
                          <button
                            onClick={() => setTraceOpen(!traceOpen)}
                            className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/40 dark:hover:bg-white/5 transition text-left text-sm font-semibold text-slate-700 dark:text-slate-300"
                          >
                            <div className="flex items-center space-x-2">
                              <Terminal className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                              <span>Agent Execution Trace ({response.tool_trace.length} Calls)</span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-mono text-slate-400">{response.total_execution_time_ms.toFixed(1)} ms total</span>
                              {traceOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                            </div>
                          </button>

                          {traceOpen && (
                            <div className="p-6 border-t border-white/20 dark:border-white/10 space-y-4">
                              {response.tool_trace.map((trace, idx) => (
                                <div key={idx} className="liquid-glass-subtle rounded-xl p-3.5 text-xs font-mono">
                                  <div className="flex items-center justify-between text-slate-800 dark:text-slate-300 mb-1.5">
                                    <span className="font-bold text-purple-600 dark:text-purple-400">Step {idx + 1}: {trace.tool_name}()</span>
                                    <span className="text-slate-500">{trace.execution_time_ms.toFixed(1)} ms</span>
                                  </div>
                                  <div className="text-slate-600 dark:text-slate-400 mb-1">
                                    <span className="text-slate-400 dark:text-slate-500">Arguments: </span>
                                    {JSON.stringify(trace.arguments)}
                                  </div>
                                  <div className="text-slate-700 dark:text-slate-300">
                                    <span className="text-slate-400 dark:text-slate-500">Summary: </span>
                                    {trace.output_summary}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Right: Source Logbook Inspector */}
                      <div className="space-y-6">
                        <div className="liquid-glass rounded-3xl p-5 shadow-xl">
                          <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-3 mb-4">
                            <h3 className="text-sm font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
                              <Database className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                              <span>Source Logbook Inspector</span>
                            </h3>
                            <span className="text-xs text-slate-500 font-mono">
                              {response.citations.length} Verified Sources
                            </span>
                          </div>

                          <div className="flex flex-wrap gap-1.5 mb-4">
                            {response.citations.map((c) => (
                              <button
                                key={c.record_id}
                                onClick={() => setSelectedRecord(c)}
                                className={`px-2.5 py-1 text-xs font-mono rounded-xl border transition ${
                                  selectedRecord?.record_id === c.record_id
                                    ? "liquid-pill-active text-white font-bold"
                                    : "liquid-glass-subtle text-slate-700 dark:text-slate-300 hover:scale-105"
                                }`}
                              >
                                [ID: {c.record_id}]
                              </button>
                            ))}
                          </div>

                          {selectedRecord ? (
                            <div className="liquid-glass-subtle rounded-2xl p-4 space-y-4">
                              <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-2">
                                <div className="flex items-center space-x-2">
                                  <span className="text-xs text-slate-500">IDENT:</span>
                                  <span className="text-sm font-mono font-bold text-purple-600 dark:text-purple-400">{selectedRecord.record_id}</span>
                                </div>
                                <button
                                  onClick={() => copyToClipboard(selectedRecord.record_id)}
                                  className="p-1 rounded-lg hover:bg-white/40 dark:hover:bg-white/10 text-slate-500 dark:text-slate-400 transition"
                                  title="Copy Record ID"
                                >
                                  {copiedId === selectedRecord.record_id ? (
                                    <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                                  ) : (
                                    <Copy className="h-3.5 w-3.5" />
                                  )}
                                </button>
                              </div>

                              <div>
                                <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                                  Reported Problem (Raw):
                                </span>
                                <p className="text-xs font-mono text-slate-800 dark:text-slate-200 liquid-glass-input p-2.5 rounded-xl leading-relaxed">
                                  {selectedRecord.problem_raw}
                                </p>
                              </div>

                              <div>
                                <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                                  Action Taken by Technician (Raw):
                                </span>
                                <p className="text-xs font-mono text-emerald-700 dark:text-emerald-300 liquid-glass-input p-2.5 rounded-xl leading-relaxed">
                                  {selectedRecord.action_raw}
                                </p>
                              </div>
                            </div>
                          ) : (
                            <div className="p-8 text-center text-xs text-slate-400">
                              No citation selected.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* TAB 2: TECHNICIAN ACCOUNTS */}
              {activeAdminTab === "technicians" && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Create Account Form */}
                  <div className="liquid-glass rounded-3xl p-6 shadow-xl space-y-4">
                    <div className="border-b border-white/20 dark:border-white/10 pb-3">
                      <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                        <UserPlus className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                        <span>Create Technician Account</span>
                      </h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        Register a new technician authorized to query logbooks and submit fix requests.
                      </p>
                    </div>

                    {techActionMsg && (
                      <div
                        className={`p-3 rounded-2xl text-xs font-medium ${
                          techActionMsg.startsWith("Error")
                            ? "bg-rose-500/15 border border-rose-500/30 text-rose-700 dark:text-rose-300"
                            : "bg-emerald-500/15 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300"
                        }`}
                      >
                        {techActionMsg}
                      </div>
                    )}

                    <form onSubmit={handleCreateTechnician} className="space-y-3.5">
                      <div>
                        <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Username</label>
                        <input
                          type="text"
                          required
                          value={newTechUser}
                          onChange={(e) => setNewTechUser(e.target.value)}
                          placeholder="e.g. tech_alex"
                          className="w-full px-3.5 py-2.5 liquid-glass-input rounded-2xl text-xs text-slate-900 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                        />
                      </div>

                      <div>
                        <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Full Name & Specialty</label>
                        <input
                          type="text"
                          required
                          value={newTechName}
                          onChange={(e) => setNewTechName(e.target.value)}
                          placeholder="e.g. Alex Mercer (Turbines & Fuel)"
                          className="w-full px-3.5 py-2.5 liquid-glass-input rounded-2xl text-xs text-slate-900 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                        />
                      </div>

                      <div>
                        <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Temporary Password</label>
                        <input
                          type="password"
                          required
                          value={newTechPass}
                          onChange={(e) => setNewTechPass(e.target.value)}
                          placeholder="At least 4 characters"
                          className="w-full px-3.5 py-2.5 liquid-glass-input rounded-2xl text-xs text-slate-900 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                        />
                      </div>

                      <button
                        type="submit"
                        disabled={creatingTech || !newTechUser.trim() || !newTechPass.trim() || !newTechName.trim()}
                        className="w-full py-2.5 liquid-pill-active hover:brightness-110 active:scale-95 disabled:opacity-50 text-white rounded-2xl text-xs font-semibold transition flex items-center justify-center space-x-1.5 shadow-lg"
                      >
                        {creatingTech ? (
                          <>
                            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                            <span>Creating Account...</span>
                          </>
                        ) : (
                          <>
                            <UserPlus className="h-3.5 w-3.5" />
                            <span>Create Technician Account</span>
                          </>
                        )}
                      </button>
                    </form>
                  </div>

                  {/* Registered Staff Table */}
                  <div className="lg:col-span-2 liquid-glass rounded-3xl p-6 shadow-xl space-y-4">
                    <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-3">
                      <div>
                        <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                          <Users className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                          <span>Registered Technicians & Staff</span>
                        </h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                          Current maintenance identities stored in active database.
                        </p>
                      </div>
                      <button
                        onClick={fetchTechnicians}
                        disabled={loadingTechs}
                        className="px-3 py-1.5 liquid-glass-subtle hover:scale-105 active:scale-95 text-slate-700 dark:text-slate-300 rounded-xl text-xs font-medium flex items-center space-x-1.5 transition"
                      >
                        <RefreshCw className={`h-3.5 w-3.5 ${loadingTechs ? "animate-spin" : ""}`} />
                        <span>Refresh</span>
                      </button>
                    </div>

                    {loadingTechs ? (
                      <div className="py-12 text-center text-xs text-slate-500">Loading technician roster...</div>
                    ) : technicians.filter((u) => u.role === "technician" && u.username !== "admin").length === 0 ? (
                      <div className="py-12 text-center text-xs text-slate-500">No technicians registered.</div>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead className="liquid-glass-subtle text-slate-500 font-semibold border-b border-white/20 dark:border-white/10">
                            <tr>
                              <th className="py-3 px-4">Username</th>
                              <th className="py-3 px-4">Full Name</th>
                              <th className="py-3 px-4">Role</th>
                              <th className="py-3 px-4">Registered Date</th>
                              <th className="py-3 px-4">Last Active</th>
                              <th className="py-3 px-4 text-right">Actions</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-white/10 dark:divide-white/5 font-mono">
                            {technicians
                              .filter((u) => u.role === "technician" && u.username !== "admin")
                              .map((u) => (
                                <tr key={u.id} className="hover:bg-white/30 dark:hover:bg-white/5 transition group">
                                  <td className="py-3 px-4 font-semibold">
                                    <button
                                      type="button"
                                      onClick={() => handleViewTechnicianLogs(u.username)}
                                      className="text-purple-600 dark:text-purple-400 hover:text-purple-500 hover:underline flex items-center space-x-1.5 transition cursor-pointer text-left"
                                      title={`Click to view audit logs for technician '${u.username}'`}
                                    >
                                      <span>{u.username}</span>
                                      <Clock className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                                    </button>
                                  </td>
                                  <td className="py-3 px-4 text-slate-800 dark:text-slate-200 font-sans">
                                    <button
                                      type="button"
                                      onClick={() => handleViewTechnicianLogs(u.username)}
                                      className="hover:underline hover:text-purple-600 dark:hover:text-purple-300 transition text-left cursor-pointer"
                                      title={`Click to view audit logs for technician '${u.username}'`}
                                    >
                                      {u.full_name}
                                    </button>
                                  </td>
                                  <td className="py-3 px-4">
                                    <span className="px-2 py-0.5 rounded-full text-[10px] uppercase font-bold bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30">
                                      {u.role}
                                    </span>
                                  </td>
                                  <td className="py-3 px-4 text-slate-500 whitespace-nowrap">
                                    {new Date(u.created_at).toLocaleDateString()}
                                  </td>
                                  <td className="py-3 px-4 whitespace-nowrap">
                                    {u.last_active ? (
                                      <span
                                        className="inline-flex items-center space-x-1.5 text-xs text-emerald-600 dark:text-emerald-400 font-sans font-medium"
                                        title={`Last activity recorded at: ${new Date(u.last_active).toLocaleString()}`}
                                      >
                                        <span className="relative flex h-2 w-2">
                                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                                        </span>
                                        <span>{formatLastActive(u.last_active)}</span>
                                      </span>
                                    ) : (
                                      <span className="text-slate-400 dark:text-neutral-500 text-xs font-sans">
                                        Never
                                      </span>
                                    )}
                                  </td>
                                  <td className="py-3 px-4 text-right space-x-2 whitespace-nowrap">
                                    <button
                                      type="button"
                                      onClick={() => handleViewTechnicianLogs(u.username)}
                                      className="px-2.5 py-1 text-xs rounded-lg bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20 hover:bg-purple-600 hover:text-white transition inline-flex items-center space-x-1 cursor-pointer"
                                      title={`View query history for technician '${u.username}'`}
                                    >
                                      <Clock className="h-3 w-3" />
                                      <span>View Logs</span>
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => handleDeleteTechnician(u.id, u.username)}
                                      disabled={deletingTechId === u.id}
                                      className="px-2.5 py-1 text-xs rounded-lg bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20 hover:bg-rose-500 hover:text-white transition inline-flex items-center space-x-1 disabled:opacity-50 cursor-pointer"
                                      title={`Delete technician ${u.username}`}
                                    >
                                      <Trash2 className="h-3 w-3" />
                                      <span>{deletingTechId === u.id ? "Deleting..." : "Delete"}</span>
                                    </button>
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* TAB 3: PENDING FIX APPROVALS */}
              {activeAdminTab === "pending" && (
                <div className="liquid-glass rounded-3xl p-6 shadow-xl space-y-6">
                  <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-4">
                    <div>
                      <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                        <ClipboardList className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                        <span>Pending Maintenance Fix Approvals Queue</span>
                      </h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        Review new squawks and fixes submitted by technicians. Approving permanently inserts the record into the fleet logbook and dynamically updates the hybrid index.
                      </p>
                    </div>
                    <button
                      onClick={fetchPendingFixes}
                      disabled={loadingPending}
                      className="px-3.5 py-1.5 liquid-glass-subtle hover:scale-105 active:scale-95 text-slate-700 dark:text-slate-300 rounded-xl text-xs font-medium flex items-center space-x-1.5 transition"
                    >
                      <RefreshCw className={`h-3.5 w-3.5 ${loadingPending ? "animate-spin" : ""}`} />
                      <span>Refresh Queue</span>
                    </button>
                  </div>

                  {adminActionStatus && (
                    <div className="p-3.5 rounded-2xl bg-blue-500/15 border border-blue-500/30 text-blue-800 dark:text-blue-200 text-xs font-medium">
                      {adminActionStatus}
                    </div>
                  )}

                  {loadingPending ? (
                    <div className="py-12 text-center text-xs text-slate-500">Loading pending requests...</div>
                  ) : pendingFixes.length === 0 ? (
                    <div className="py-16 text-center space-y-2.5">
                      <CheckCircle2 className="h-10 w-10 text-emerald-600 dark:text-emerald-400 mx-auto" />
                      <p className="text-sm font-semibold text-slate-800 dark:text-slate-300">Queue is Clear</p>
                      <p className="text-xs text-slate-500">
                        No technician fix requests are currently awaiting approval.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {pendingFixes.map((item) => (
                        <div
                          key={item.temp_id}
                          className="liquid-glass-subtle rounded-2xl p-5 space-y-4"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/20 dark:border-white/10 pb-3">
                            <div className="flex items-center space-x-2">
                              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30">
                                {item.temp_id}
                              </span>
                              <span className="text-xs text-slate-600 dark:text-slate-400">
                                Submitted by <strong className="text-slate-800 dark:text-slate-200">{item.submitted_by}</strong>
                              </span>
                            </div>
                            <span className="text-xs text-slate-500 font-mono">
                              {new Date(item.submitted_at).toLocaleString()}
                            </span>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                                Discrepancy (Raw Problem):
                              </span>
                              <p className="text-xs font-mono text-slate-800 dark:text-slate-200 liquid-glass-input p-3 rounded-xl">
                                {item.problem_raw}
                              </p>
                            </div>

                            <div>
                              <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                                Corrective Action (Raw Action):
                              </span>
                              <p className="text-xs font-mono text-emerald-700 dark:text-emerald-300 liquid-glass-input p-3 rounded-xl">
                                {item.action_raw}
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center justify-end space-x-2 pt-2">
                            <button
                              onClick={() => handleReject(item.temp_id)}
                              className="px-3.5 py-1.5 bg-rose-500/15 hover:bg-rose-500/25 text-rose-700 dark:text-rose-300 border border-rose-500/30 rounded-xl text-xs font-semibold flex items-center space-x-1 transition"
                            >
                              <XCircle className="h-3.5 w-3.5" />
                              <span>Reject Fix</span>
                            </button>

                            <button
                              onClick={() => handleApprove(item.temp_id)}
                              className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition shadow-lg shadow-emerald-600/30"
                            >
                              <CheckCircle className="h-3.5 w-3.5" />
                              <span>Approve & Index into Fleet Logbook</span>
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* TAB 4: TECHNICIAN QUERY HISTORY */}
              {activeAdminTab === "audit" && (
                <div className="liquid-glass rounded-3xl p-6 shadow-xl space-y-6">
                  <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-4">
                    <div>
                      <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                        <Clock className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                        <span>Technician Query History</span>
                      </h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        Audit trail of all queries executed by technicians, including evidence assessments and cited record IDs.
                      </p>
                    </div>
                    <div className="flex items-center space-x-2.5">
                      <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                        <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        <span>Live Updates</span>
                      </span>
                      <button
                        onClick={fetchAuditLogs}
                        disabled={loadingAudit}
                        className="px-3.5 py-1.5 liquid-glass-subtle hover:scale-105 active:scale-95 text-slate-700 dark:text-slate-300 rounded-xl text-xs font-medium flex items-center space-x-1.5 transition"
                      >
                        <RefreshCw className={`h-3.5 w-3.5 ${loadingAudit ? "animate-spin" : ""}`} />
                        <span>Refresh Logs</span>
                      </button>
                    </div>
                  </div>

                  {/* Pre-applied Technician Filter Banner */}
                  {auditTechFilter !== "all" && (
                    <div className="flex items-center justify-between px-4 py-2.5 rounded-2xl bg-purple-500/10 border border-purple-500/25 text-xs text-purple-800 dark:text-purple-300">
                      <div className="flex items-center space-x-2">
                        <User className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                        <span>
                          Filtered by technician: <strong className="font-mono font-bold text-purple-900 dark:text-purple-100">{auditTechFilter}</strong>
                        </span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-purple-500/20 text-purple-700 dark:text-purple-300">
                          {auditLogs.filter((log) => {
                            const target = auditTechFilter.toLowerCase().trim();
                            const logUser = log.user_name.toLowerCase().trim();
                            return logUser === target || logUser.replace(/\s+/g, "") === target.replace(/\s+/g, "") || logUser.includes(target);
                          }).length} log(s)
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setAuditTechFilter("all")}
                        className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-900 dark:text-purple-100 font-semibold transition cursor-pointer"
                      >
                        <X className="h-3 w-3" />
                        <span>Show All Technicians</span>
                      </button>
                    </div>
                  )}

                  {/* Multi-Criteria Filter Toolbar */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 p-3.5 rounded-2xl bg-white/40 dark:bg-white/[0.02] border border-white/20 dark:border-white/10">
                    {/* Technician Name Filter */}
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                        Technician
                      </label>
                      <div className="relative">
                        <User className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                        <select
                          value={auditTechFilter}
                          onChange={(e) => setAuditTechFilter(e.target.value)}
                          className="w-full pl-8 pr-3 py-1.5 rounded-xl liquid-glass-input text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-purple-500 cursor-pointer"
                        >
                          <option value="all">All Technicians</option>
                          {technicians
                            .filter((t) => t.role === "technician" && t.username !== "admin")
                            .map((t) => (
                              <option key={t.id} value={t.username}>
                                {t.username} ({t.full_name})
                              </option>
                            ))}
                        </select>
                      </div>
                    </div>

                    {/* Date Filter */}
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                        Date
                      </label>
                      <div className="relative">
                        <Calendar className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                        <input
                          type="date"
                          value={auditDateFilter}
                          onChange={(e) => setAuditDateFilter(e.target.value)}
                          className="w-full pl-8 pr-7 py-1.5 rounded-xl liquid-glass-input text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                        />
                        {auditDateFilter && (
                          <button
                            type="button"
                            onClick={() => setAuditDateFilter("")}
                            className="absolute right-2 top-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer"
                            title="Clear date filter"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Query Search Filter */}
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                        Query Keyword
                      </label>
                      <div className="relative">
                        <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                        <input
                          type="text"
                          value={auditQueryFilter}
                          onChange={(e) => setAuditQueryFilter(e.target.value)}
                          placeholder="Filter symptom or action..."
                          className="w-full pl-8 pr-7 py-1.5 rounded-xl liquid-glass-input text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                        />
                        {auditQueryFilter && (
                          <button
                            type="button"
                            onClick={() => setAuditQueryFilter("")}
                            className="absolute right-2 top-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer"
                            title="Clear keyword filter"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Evidence Status Filter */}
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                        Evidence Status
                      </label>
                      <div className="relative">
                        <Filter className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                        <select
                          value={auditStatusFilter}
                          onChange={(e) => setAuditStatusFilter(e.target.value)}
                          className="w-full pl-8 pr-3 py-1.5 rounded-xl liquid-glass-input text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-purple-500 cursor-pointer"
                        >
                          <option value="all">All Evidence Statuses</option>
                          <option value="sufficient">Sufficient Evidence</option>
                          <option value="insufficient">Insufficient (Abstained)</option>
                          <option value="needs_clarification">Needs Clarification</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Filter Active Indicator & Reset Button */}
                  {(auditTechFilter !== "all" || auditDateFilter || auditQueryFilter || auditStatusFilter !== "all") && (
                    <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 px-1">
                      <span>
                        Showing{" "}
                        <strong className="text-slate-900 dark:text-white">
                          {
                            auditLogs.filter((log) => {
                              if (auditTechFilter !== "all") {
                                const target = auditTechFilter.toLowerCase().trim();
                                const logUser = log.user_name.toLowerCase().trim();
                                const logUserNoSpaces = logUser.replace(/\s+/g, "");
                                const targetNoSpaces = target.replace(/\s+/g, "");
                                if (logUser !== target && logUserNoSpaces !== targetNoSpaces && !logUser.includes(target)) {
                                  return false;
                                }
                              }
                              if (auditDateFilter) {
                                const logDateStr = new Date(log.timestamp).toISOString().split("T")[0];
                                if (logDateStr !== auditDateFilter) return false;
                              }
                              if (auditQueryFilter.trim()) {
                                const q = auditQueryFilter.toLowerCase().trim();
                                if (!log.query.toLowerCase().includes(q)) return false;
                              }
                              if (auditStatusFilter !== "all" && log.evidence_status !== auditStatusFilter) {
                                return false;
                              }
                              return true;
                            }).length
                          }
                        </strong>{" "}
                        of {auditLogs.length} total query logs
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          setAuditTechFilter("all");
                          setAuditDateFilter("");
                          setAuditQueryFilter("");
                          setAuditStatusFilter("all");
                        }}
                        className="text-purple-600 dark:text-purple-400 hover:underline font-semibold flex items-center space-x-1 cursor-pointer"
                      >
                        <X className="h-3 w-3" />
                        <span>Reset All Filters</span>
                      </button>
                    </div>
                  )}

                  {loadingAudit ? (
                    <div className="py-12 text-center text-xs text-slate-500">Loading query history...</div>
                  ) : auditLogs.length === 0 ? (
                    <div className="py-16 text-center text-xs text-slate-500">
                      No activity has been recorded yet. Perform a query in the Copilot to generate audit trails.
                    </div>
                  ) : auditLogs.filter((log) => {
                      if (auditTechFilter !== "all") {
                        const target = auditTechFilter.toLowerCase().trim();
                        const logUser = log.user_name.toLowerCase().trim();
                        const logUserNoSpaces = logUser.replace(/\s+/g, "");
                        const targetNoSpaces = target.replace(/\s+/g, "");
                        if (logUser !== target && logUserNoSpaces !== targetNoSpaces && !logUser.includes(target)) {
                          return false;
                        }
                      }
                      if (auditDateFilter) {
                        const logDateStr = new Date(log.timestamp).toISOString().split("T")[0];
                        if (logDateStr !== auditDateFilter) return false;
                      }
                      if (auditQueryFilter.trim()) {
                        const q = auditQueryFilter.toLowerCase().trim();
                        if (!log.query.toLowerCase().includes(q)) return false;
                      }
                      if (auditStatusFilter !== "all" && log.evidence_status !== auditStatusFilter) {
                        return false;
                      }
                      return true;
                    }).length === 0 ? (
                    <div className="py-14 text-center text-xs text-slate-500 space-y-2">
                      <p>No queries match your current filter settings.</p>
                      <button
                        type="button"
                        onClick={() => {
                          setAuditTechFilter("all");
                          setAuditDateFilter("");
                          setAuditQueryFilter("");
                          setAuditStatusFilter("all");
                        }}
                        className="px-3 py-1.5 rounded-xl bg-purple-500/10 text-purple-600 dark:text-purple-400 hover:bg-purple-500/20 font-semibold transition"
                      >
                        Clear Filters
                      </button>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="liquid-glass-subtle text-slate-500 font-semibold border-b border-white/20 dark:border-white/10">
                          <tr>
                            <th className="py-3 px-4">Timestamp</th>
                            <th className="py-3 px-4">User</th>
                            <th className="py-3 px-4">Role</th>
                            <th className="py-3 px-4">Maintenance Query</th>
                            <th className="py-3 px-4">Evidence Status</th>
                            <th className="py-3 px-4">Cited IDs</th>
                            <th className="py-3 px-4">Latency</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/10 dark:divide-white/5 font-mono">
                          {auditLogs
                            .filter((log) => {
                              if (auditTechFilter !== "all") {
                                const target = auditTechFilter.toLowerCase().trim();
                                const logUser = log.user_name.toLowerCase().trim();
                                const logUserNoSpaces = logUser.replace(/\s+/g, "");
                                const targetNoSpaces = target.replace(/\s+/g, "");
                                if (logUser !== target && logUserNoSpaces !== targetNoSpaces && !logUser.includes(target)) {
                                  return false;
                                }
                              }
                              if (auditDateFilter) {
                                const logDateStr = new Date(log.timestamp).toISOString().split("T")[0];
                                if (logDateStr !== auditDateFilter) return false;
                              }
                              if (auditQueryFilter.trim()) {
                                const q = auditQueryFilter.toLowerCase().trim();
                                if (!log.query.toLowerCase().includes(q)) return false;
                              }
                              if (auditStatusFilter !== "all" && log.evidence_status !== auditStatusFilter) {
                                return false;
                              }
                              return true;
                            })
                            .map((log) => (
                              <tr key={log.id} className="hover:bg-white/30 dark:hover:bg-white/5 transition">
                                <td className="py-3 px-4 text-slate-500 whitespace-nowrap text-[11px]">
                                  {new Date(log.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })},{" "}
                                  {new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                </td>
                                <td className="py-3 px-4 text-slate-800 dark:text-white font-sans font-semibold">
                                  {log.user_name}
                                </td>
                                <td className="py-3 px-4">
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold ${
                                      log.user_role === "admin"
                                        ? "bg-purple-500/15 text-purple-700 dark:text-purple-300 border border-purple-500/30"
                                        : "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30"
                                    }`}
                                  >
                                    {log.user_role}
                                  </span>
                                </td>
                                <td className="py-3 px-4 text-slate-700 dark:text-slate-200 font-sans max-w-xs truncate" title={log.query}>
                                  {log.query}
                                </td>
                                <td className="py-3 px-4">
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                                      log.evidence_status === "sufficient"
                                        ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
                                        : log.evidence_status === "needs_clarification"
                                        ? "bg-sky-500/15 text-sky-700 dark:text-sky-400"
                                        : "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                                    }`}
                                  >
                                    {log.evidence_status}
                                  </span>
                                </td>
                                <td className="py-3 px-4 text-purple-600 dark:text-purple-400">
                                  {log.cited_record_ids.length > 0 ? (
                                    <span>{log.cited_record_ids.join(", ")}</span>
                                  ) : (
                                    <span className="text-slate-400 dark:text-slate-600">—</span>
                                  )}
                                </td>
                                <td className="py-3 px-4 text-slate-500">
                                  {log.execution_time_ms.toFixed(0)}ms
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* --- TECHNICIAN VIEW (Investigation or Personal History) --- */
          techView === "history" ? (
            <div className="border border-slate-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 rounded-2xl p-6 shadow-sm space-y-5">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-neutral-800 gap-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                    <Clock className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                    <span>My Query History</span>
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-neutral-400 mt-1">
                    Historical logbook investigations executed under your technician identity ({currentUser.full_name}).
                  </p>
                </div>

                <div className="flex items-center space-x-2">
                  <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span>Live Updates</span>
                  </span>
                  <button
                    onClick={() => setTechView("search")}
                    className="px-3.5 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold transition flex items-center space-x-1.5 shadow-sm"
                  >
                    <Search className="h-3.5 w-3.5" />
                    <span>Search Logbooks</span>
                  </button>
                  <button
                    onClick={fetchMyHistory}
                    disabled={loadingMyHistory}
                    className="px-3 py-1.5 rounded-xl border border-slate-200 dark:border-neutral-800 hover:bg-slate-50 dark:hover:bg-neutral-800 text-slate-700 dark:text-slate-300 text-xs font-medium flex items-center space-x-1.5 transition"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${loadingMyHistory ? "animate-spin" : ""}`} />
                    <span>Refresh</span>
                  </button>
                </div>
              </div>

              {/* Technician Query History Filters Toolbar */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3.5 rounded-xl border border-slate-200 dark:border-neutral-800 bg-slate-50/60 dark:bg-neutral-950/40">
                {/* Date Filter */}
                <div>
                  <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                    Date
                  </label>
                  <div className="relative">
                    <Calendar className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                    <input
                      type="date"
                      value={myHistoryDateFilter}
                      onChange={(e) => setMyHistoryDateFilter(e.target.value)}
                      className="w-full pl-8 pr-7 py-1.5 rounded-lg border border-slate-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    {myHistoryDateFilter && (
                      <button
                        type="button"
                        onClick={() => setMyHistoryDateFilter("")}
                        className="absolute right-2 top-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                        title="Clear date filter"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Query Keyword Search */}
                <div>
                  <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                    Query Search
                  </label>
                  <div className="relative">
                    <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                    <input
                      type="text"
                      value={myHistoryQueryFilter}
                      onChange={(e) => setMyHistoryQueryFilter(e.target.value)}
                      placeholder="Filter symptom or action..."
                      className="w-full pl-8 pr-7 py-1.5 rounded-lg border border-slate-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    {myHistoryQueryFilter && (
                      <button
                        type="button"
                        onClick={() => setMyHistoryQueryFilter("")}
                        className="absolute right-2 top-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                        title="Clear keyword filter"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Evidence Status Filter */}
                <div>
                  <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                    Evidence Status
                  </label>
                  <div className="relative">
                    <Filter className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                    <select
                      value={myHistoryStatusFilter}
                      onChange={(e) => setMyHistoryStatusFilter(e.target.value)}
                      className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-slate-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
                    >
                      <option value="all">All Evidence Statuses</option>
                      <option value="sufficient">Sufficient Evidence</option>
                      <option value="insufficient">Insufficient (Abstained)</option>
                      <option value="needs_clarification">Needs Clarification</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Active Filter Indicators & Reset Button */}
              {(myHistoryDateFilter || myHistoryQueryFilter || myHistoryStatusFilter !== "all") && (
                <div className="flex items-center justify-between text-xs text-slate-500 dark:text-neutral-400 px-1">
                  <span>
                    Showing{" "}
                    <strong className="text-slate-900 dark:text-white">
                      {
                        myQueryLogs.filter((log) => {
                          if (myHistoryDateFilter) {
                            const logDateStr = new Date(log.timestamp).toISOString().split("T")[0];
                            if (logDateStr !== myHistoryDateFilter) return false;
                          }
                          if (myHistoryQueryFilter.trim()) {
                            const q = myHistoryQueryFilter.toLowerCase().trim();
                            if (!log.query.toLowerCase().includes(q)) return false;
                          }
                          if (myHistoryStatusFilter !== "all" && log.evidence_status !== myHistoryStatusFilter) {
                            return false;
                          }
                          return true;
                        }).length
                      }
                    </strong>{" "}
                    of {myQueryLogs.length} your past queries
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setMyHistoryDateFilter("");
                      setMyHistoryQueryFilter("");
                      setMyHistoryStatusFilter("all");
                    }}
                    className="text-blue-600 dark:text-blue-400 hover:underline font-semibold flex items-center space-x-1 cursor-pointer"
                  >
                    <X className="h-3 w-3" />
                    <span>Reset All Filters</span>
                  </button>
                </div>
              )}

              {loadingMyHistory ? (
                <div className="py-16 text-center text-xs text-slate-400 flex flex-col items-center space-y-2">
                  <RefreshCw className="h-5 w-5 animate-spin text-blue-600" />
                  <span>Loading your query audit history...</span>
                </div>
              ) : myQueryLogs.length === 0 ? (
                <div className="py-16 text-center text-xs text-slate-400 space-y-3">
                  <p>You haven't run any queries yet.</p>
                  <button
                    onClick={() => setTechView("search")}
                    className="px-4 py-2 bg-blue-600 text-white rounded-xl text-xs font-semibold hover:bg-blue-700 transition"
                  >
                    Start an Investigation
                  </button>
                </div>
              ) : myQueryLogs.filter((log) => {
                  if (myHistoryDateFilter) {
                    const logDateStr = new Date(log.timestamp).toISOString().split("T")[0];
                    if (logDateStr !== myHistoryDateFilter) return false;
                  }
                  if (myHistoryQueryFilter.trim()) {
                    const q = myHistoryQueryFilter.toLowerCase().trim();
                    if (!log.query.toLowerCase().includes(q)) return false;
                  }
                  if (myHistoryStatusFilter !== "all" && log.evidence_status !== myHistoryStatusFilter) {
                    return false;
                  }
                  return true;
                }).length === 0 ? (
                <div className="py-14 text-center text-xs text-slate-500 space-y-2">
                  <p>No queries match your current filter criteria.</p>
                  <button
                    type="button"
                    onClick={() => {
                      setMyHistoryDateFilter("");
                      setMyHistoryQueryFilter("");
                      setMyHistoryStatusFilter("all");
                    }}
                    className="px-3 py-1.5 rounded-xl bg-blue-500/10 text-blue-600 dark:text-blue-400 hover:bg-blue-500/20 font-semibold transition"
                  >
                    Clear Filters
                  </button>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 dark:bg-neutral-950/60 text-slate-500 font-semibold border-b border-slate-200 dark:border-neutral-800">
                      <tr>
                        <th className="py-3 px-4">Timestamp</th>
                        <th className="py-3 px-4">Maintenance Query</th>
                        <th className="py-3 px-4">Evidence Status</th>
                        <th className="py-3 px-4">Cited IDs</th>
                        <th className="py-3 px-4">Latency</th>
                        <th className="py-3 px-4 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-neutral-800/60 font-mono">
                      {myQueryLogs
                        .filter((log) => {
                          if (myHistoryDateFilter) {
                            const logDateStr = new Date(log.timestamp).toISOString().split("T")[0];
                            if (logDateStr !== myHistoryDateFilter) return false;
                          }
                          if (myHistoryQueryFilter.trim()) {
                            const q = myHistoryQueryFilter.toLowerCase().trim();
                            if (!log.query.toLowerCase().includes(q)) return false;
                          }
                          if (myHistoryStatusFilter !== "all" && log.evidence_status !== myHistoryStatusFilter) {
                            return false;
                          }
                          return true;
                        })
                        .map((log) => (
                          <tr key={log.id} className="hover:bg-slate-50/70 dark:hover:bg-neutral-800/40 transition">
                            <td className="py-3 px-4 text-slate-500 whitespace-nowrap text-[11px]">
                              {new Date(log.timestamp).toLocaleDateString([], { month: "short", day: "numeric" })},{" "}
                              {new Date(log.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                            </td>
                            <td className="py-3 px-4 text-slate-900 dark:text-white font-sans max-w-xs truncate" title={log.query}>
                              {log.query}
                            </td>
                            <td className="py-3 px-4">
                              <span
                                className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold ${
                                  log.evidence_status === "sufficient"
                                    ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/30"
                                    : log.evidence_status === "needs_clarification"
                                    ? "bg-sky-500/15 text-sky-700 dark:text-sky-400 border border-sky-500/30"
                                    : "bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/30"
                                }`}
                              >
                                {log.evidence_status}
                              </span>
                            </td>
                            <td className="py-3 px-4 text-blue-600 dark:text-blue-400">
                              {log.cited_record_ids.length > 0 ? log.cited_record_ids.join(", ") : "—"}
                            </td>
                            <td className="py-3 px-4 text-slate-500">
                              {log.execution_time_ms.toFixed(0)}ms
                            </td>
                            <td className="py-3 px-4 text-right">
                              <button
                                onClick={() => {
                                  setQuery(log.query);
                                  setTechView("search");
                                }}
                                className="px-2.5 py-1 text-xs rounded-lg bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800 hover:bg-blue-600 hover:text-white transition inline-flex items-center space-x-1"
                                title="Load query into search bar"
                              >
                                <RotateCcw className="h-3 w-3" />
                                <span>Re-run</span>
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            /* --- TECHNICIAN VIEW (Full-width clean interface in Liquid Glass) --- */
            <>
            {/* Search Bar in Liquid Glass */}
            <div className="liquid-glass rounded-3xl p-6 shadow-xl">
              <label className="block text-sm font-semibold text-slate-800 dark:text-slate-300 mb-2.5 flex items-center justify-between">
                <span>Historical Logbook Query & Symptom Search</span>
                <span className="text-xs font-normal text-slate-500 dark:text-slate-400">
                  Hybrid BM25 + Dense Semantic RAG
                </span>
              </label>

              <div className="relative flex items-center">
                <Search className="absolute left-4 h-5 w-5 text-slate-400" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  placeholder="Describe maintenance squawk or discrepancy (e.g. 'cylinder 2 spark plug fouled', 'fuel servo leak')..."
                  className="w-full pl-12 pr-44 sm:pr-48 py-3.5 liquid-glass-input rounded-2xl text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 text-sm"
                />
                <button
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  className="absolute right-2 px-4 py-2 liquid-pill-blue hover:brightness-110 active:scale-95 disabled:opacity-50 text-white rounded-xl text-sm font-medium transition flex items-center space-x-1.5 shadow"
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
            </div>

            {/* Loading Skeleton */}
            {loading && (
              <div className="liquid-glass rounded-3xl p-8 flex flex-col items-center justify-center space-y-4 animate-pulse">
                <div className="h-12 w-12 rounded-2xl liquid-pill-blue flex items-center justify-center text-white">
                  <RefreshCw className="h-6 w-6 animate-spin" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">Executing Bounded Agentic Search...</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Matching across logbook database • Enforcing Evidence Gate • Validating Citations
                  </p>
                </div>
              </div>
            )}

            {/* Error Alert */}
            {error && (
              <div className="bg-rose-500/15 border border-rose-500/30 backdrop-blur-md rounded-2xl p-4 flex items-start space-x-3 text-rose-800 dark:text-rose-300">
                <ShieldAlert className="h-5 w-5 flex-shrink-0 mt-0.5 text-rose-500" />
                <div>
                  <p className="text-sm font-semibold">Investigation Error</p>
                  <p className="text-xs mt-1 text-rose-700 dark:text-rose-200">{error}</p>
                </div>
              </div>
            )}

            {/* Results Presentation Area */}
            {response && !loading && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                  {/* Evidence Status Card */}
                  <div
                    className={`rounded-2xl p-5 border shadow-md backdrop-blur-xl ${
                      response.evidence_status === "sufficient"
                        ? "bg-emerald-500/10 border-emerald-500/30"
                        : response.evidence_status === "needs_clarification"
                        ? "bg-sky-500/10 border-sky-500/30"
                        : "bg-amber-500/10 border-amber-500/30"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {response.evidence_status === "sufficient" ? (
                          <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                        ) : response.evidence_status === "needs_clarification" ? (
                          <HelpCircle className="h-5 w-5 text-sky-600 dark:text-sky-400" />
                        ) : (
                          <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                        )}
                        <span
                          className={`text-sm font-bold uppercase tracking-wider ${
                            response.evidence_status === "sufficient"
                              ? "text-emerald-700 dark:text-emerald-400"
                              : response.evidence_status === "needs_clarification"
                              ? "text-sky-700 dark:text-sky-400"
                              : "text-amber-700 dark:text-amber-400"
                          }`}
                        >
                          Evidence Assessment: {response.evidence_status.replace("_", " ")}
                        </span>
                      </div>

                      <span className="text-xs text-slate-500 font-mono">
                        Latency: {response.total_execution_time_ms.toFixed(1)} ms
                      </span>
                    </div>
                    <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">{response.evidence_reason}</p>
                  </div>

                  {/* Synthesized Answer Card */}
                  <div className="liquid-glass rounded-3xl p-6 shadow-xl space-y-4">
                    <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-3">
                      <h2 className="text-base font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
                        <Wrench className="h-4 w-4 text-cyan-600 dark:text-cyan-400" />
                        <span>Historical Maintenance Analysis</span>
                      </h2>
                      {response.is_fallback && (
                        <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30">
                          Fallback Mode
                        </span>
                      )}
                    </div>

                    <div className="max-w-none">
                      <StructuredAnswerRenderer
                        answer={response.answer}
                        citations={response.citations}
                        onSelectRecord={setSelectedRecord}
                        accentColor="cyan"
                      />
                    </div>

                    <div className="mt-4 p-4 rounded-2xl bg-amber-500/10 border border-amber-500/25 text-xs text-amber-800 dark:text-amber-200/90 flex items-start space-x-3">
                      <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="font-semibold text-amber-900 dark:text-amber-300">Descriptive Safety Notice: </span>
                        Historical logbook entries describe past actions recorded by technicians; they are NOT manufacturer-approved maintenance instructions. Always refer to the FAA-approved AMM and applicable Service Bulletins.
                      </div>
                    </div>

                    <div className="pt-3 border-t border-white/20 dark:border-white/10 flex items-center justify-between text-xs text-slate-500">
                      <div className="flex items-center space-x-1.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                        <span>Citation Verification: {response.policy_audit.policy_summary}</span>
                      </div>
                      <span>{response.cited_record_ids.length} Citing Record(s)</span>
                    </div>
                  </div>

                  {/* Tool Trace Accordion */}
                  <div className="liquid-glass rounded-2xl overflow-hidden shadow-md">
                    <button
                      onClick={() => setTraceOpen(!traceOpen)}
                      className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/40 dark:hover:bg-white/5 transition text-left text-sm font-semibold text-slate-700 dark:text-slate-300"
                    >
                      <div className="flex items-center space-x-2">
                        <Terminal className="h-4 w-4 text-cyan-600 dark:text-cyan-400" />
                        <span>Agent Execution Trace ({response.tool_trace.length} Calls)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-mono text-slate-400">{response.total_execution_time_ms.toFixed(1)} ms total</span>
                        {traceOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </div>
                    </button>

                    {traceOpen && (
                      <div className="p-6 border-t border-white/20 dark:border-white/10 space-y-4">
                        {response.tool_trace.map((trace, idx) => (
                          <div key={idx} className="liquid-glass-subtle rounded-xl p-3.5 text-xs font-mono">
                            <div className="flex items-center justify-between text-slate-800 dark:text-slate-300 mb-1.5">
                              <span className="font-bold text-cyan-600 dark:text-cyan-400">Step {idx + 1}: {trace.tool_name}()</span>
                              <span className="text-slate-500">{trace.execution_time_ms.toFixed(1)} ms</span>
                            </div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">
                              <span className="text-slate-400 dark:text-slate-500">Arguments: </span>
                              {JSON.stringify(trace.arguments)}
                            </div>
                            <div className="text-slate-700 dark:text-slate-300">
                              <span className="text-slate-400 dark:text-slate-500">Summary: </span>
                              {trace.output_summary}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Right: Source Logbook Inspector */}
                <div className="space-y-6">
                  <div className="liquid-glass rounded-3xl p-5 shadow-xl">
                    <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-3 mb-4">
                      <h3 className="text-sm font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
                        <Database className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                        <span>Source Logbook Inspector</span>
                      </h3>
                      <span className="text-xs text-slate-500 font-mono">
                        {response.citations.length} Verified Sources
                      </span>
                    </div>

                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {response.citations.map((c) => (
                        <button
                          key={c.record_id}
                          onClick={() => setSelectedRecord(c)}
                          className={`px-2.5 py-1 text-xs font-mono rounded-xl border transition ${
                            selectedRecord?.record_id === c.record_id
                              ? "liquid-pill-blue text-white font-bold"
                              : "liquid-glass-subtle text-slate-700 dark:text-slate-300 hover:scale-105"
                          }`}
                        >
                          [ID: {c.record_id}]
                        </button>
                      ))}
                    </div>

                    {selectedRecord ? (
                      <div className="liquid-glass-subtle rounded-2xl p-4 space-y-4">
                        <div className="flex items-center justify-between border-b border-white/20 dark:border-white/10 pb-2">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-slate-500">IDENT:</span>
                            <span className="text-sm font-mono font-bold text-cyan-600 dark:text-cyan-400">{selectedRecord.record_id}</span>
                          </div>
                          <button
                            onClick={() => copyToClipboard(selectedRecord.record_id)}
                            className="p-1 rounded-lg hover:bg-white/40 dark:hover:bg-white/10 text-slate-500 dark:text-slate-400 transition"
                            title="Copy Record ID"
                          >
                            {copiedId === selectedRecord.record_id ? (
                              <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                            ) : (
                              <Copy className="h-3.5 w-3.5" />
                            )}
                          </button>
                        </div>

                        <div>
                          <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                            Reported Problem (Raw):
                          </span>
                          <p className="text-xs font-mono text-slate-800 dark:text-slate-200 liquid-glass-input p-2.5 rounded-xl leading-relaxed">
                            {selectedRecord.problem_raw}
                          </p>
                        </div>

                        <div>
                          <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                            Action Taken by Technician (Raw):
                          </span>
                          <p className="text-xs font-mono text-emerald-700 dark:text-emerald-300 liquid-glass-input p-2.5 rounded-xl leading-relaxed">
                            {selectedRecord.action_raw}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="p-8 text-center text-xs text-slate-400">
                        No citation selected.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )
      )}
      </main>

      {/* Bottom Status Deck: Database Active (Left) & Aviation Glossary (Right) */}
      <section className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-2 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
          {/* Left Card: Database Active */}
          <div className="border border-slate-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 rounded-2xl p-5 shadow-sm transition">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-neutral-800 pb-3">
              <div className="flex items-center space-x-2.5">
                <div className="p-2 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400">
                  <Database className="h-4 w-4" />
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-white">
                      Database Active
                    </h4>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/25">
                      {stats ? `${stats.total_records.toLocaleString()} Records` : "Loading..."}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 dark:text-neutral-400 mt-0.5">
                    Canonical fleet maintenance logbooks & live indices
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowDatabase(!showDatabase)}
                className="px-3.5 py-1.5 rounded-xl border border-slate-200 dark:border-neutral-800 hover:bg-slate-50 dark:hover:bg-neutral-800 text-xs font-semibold text-slate-700 dark:text-slate-300 transition flex items-center space-x-1.5"
              >
                <span>{showDatabase ? "Hide Details" : "View Details"}</span>
                {showDatabase ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              </button>
            </div>

            {showDatabase && (
              <div className="pt-4 space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-3 rounded-xl border border-slate-100 dark:border-neutral-800 bg-slate-50/70 dark:bg-neutral-950/60">
                    <span className="text-[10px] uppercase font-bold text-slate-400">Total Fleet Records</span>
                    <div className="text-base font-bold text-slate-900 dark:text-white mt-0.5">
                      {stats ? stats.total_records.toLocaleString() : "..."}
                    </div>
                    <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">Approved & Searchable</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-100 dark:border-neutral-800 bg-slate-50/70 dark:bg-neutral-950/60">
                    <span className="text-[10px] uppercase font-bold text-slate-400">Pending Approvals</span>
                    <div className="text-base font-bold text-slate-900 dark:text-white mt-0.5">
                      {stats ? stats.pending_fixes : 0}
                    </div>
                    <span className="text-[10px] text-amber-600 dark:text-amber-400 font-medium">Awaiting Review</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-100 dark:border-neutral-800 bg-slate-50/70 dark:bg-neutral-950/60">
                    <span className="text-[10px] uppercase font-bold text-slate-400">Registered Technicians</span>
                    <div className="text-base font-bold text-slate-900 dark:text-white mt-0.5">
                      {stats ? stats.total_technicians : 0}
                    </div>
                    <span className="text-[10px] text-blue-600 dark:text-blue-400 font-medium">Active Staff Identities</span>
                  </div>

                  <div className="p-3 rounded-xl border border-slate-100 dark:border-neutral-800 bg-slate-50/70 dark:bg-neutral-950/60">
                    <span className="text-[10px] uppercase font-bold text-slate-400">Query Audit Ledger</span>
                    <div className="text-base font-bold text-slate-900 dark:text-white mt-0.5">
                      {stats ? stats.total_queries : 0}
                    </div>
                    <span className="text-[10px] text-purple-600 dark:text-purple-400 font-medium">Live Inquiries Logged</span>
                  </div>
                </div>

                <div className="p-2.5 rounded-xl border border-emerald-500/20 bg-emerald-500/5 text-[11px] text-slate-600 dark:text-slate-400 flex items-center justify-between">
                  <span>Dynamic Re-Indexing: Enabled on Admin Approval</span>
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400">Status: Operational</span>
                </div>
              </div>
            )}
          </div>

          {/* Right Card: Aviation Acronym Glossary */}
          <div className="border border-slate-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 rounded-2xl p-5 shadow-sm transition">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-neutral-800 pb-3">
              <div className="flex items-center space-x-2.5">
                <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400">
                  <BookOpen className="h-4 w-4" />
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-white">
                      Aviation Glossary
                    </h4>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-blue-500/15 text-blue-700 dark:text-blue-400 border border-blue-500/25">
                      {Object.keys(glossary).length} Terms
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 dark:text-neutral-400 mt-0.5">
                    Standard abbreviations & dynamic logbook learning
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowGlossary(!showGlossary)}
                className="px-3.5 py-1.5 rounded-xl border border-slate-200 dark:border-neutral-800 hover:bg-slate-50 dark:hover:bg-neutral-800 text-xs font-semibold text-slate-700 dark:text-slate-300 transition flex items-center space-x-1.5"
              >
                <span>{showGlossary ? "Hide Glossary" : "Explore Glossary"}</span>
                {showGlossary ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              </button>
            </div>

            {showGlossary && (
              <div className="pt-4 space-y-3">
                <div className="flex items-center">
                  <div className="relative w-full">
                    <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
                    <input
                      type="text"
                      value={glossaryFilter}
                      onChange={(e) => setGlossaryFilter(e.target.value)}
                      placeholder="Filter acronym (e.g. AMM, PFT, A/C)..."
                      className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-slate-200 dark:border-neutral-800 bg-slate-50 dark:bg-neutral-950 text-xs text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-72 overflow-y-auto pr-1">
                  {Object.entries(glossary)
                    .filter(([term, info]) =>
                      !glossaryFilter ||
                      term.toLowerCase().includes(glossaryFilter.toLowerCase()) ||
                      info.expansion.toLowerCase().includes(glossaryFilter.toLowerCase())
                    )
                    .map(([term, info]: [string, GlossaryItem]) => (
                      <div key={term} className="border border-slate-100 dark:border-neutral-800 rounded-xl p-2.5 bg-slate-50/50 dark:bg-neutral-950/50 text-xs hover:border-blue-400 dark:hover:border-blue-600 transition">
                        <div className="font-mono font-bold text-blue-600 dark:text-blue-400">{term}</div>
                        <div className="text-slate-700 dark:text-slate-300 truncate text-[11px]" title={info.expansion}>
                          {info.expansion}
                        </div>
                        <div className="text-[10px] text-slate-400 mt-0.5">Occurrences: {info.frequency}</div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-200/80 dark:border-neutral-800/80 py-6 text-center text-xs text-slate-500 dark:text-neutral-400 transition-colors space-y-1.5">
        <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-slate-700 dark:text-white font-medium">
          <span>Maintenance Copilot</span>
          <span>•</span>
          <span>Aviation Fleet Knowledge & Incident Intelligence Engine</span>
          <span>•</span>
          <span>System Operational</span>
        </div>
        <p className="text-[11px] text-slate-400 dark:text-neutral-500 max-w-2xl mx-auto px-4">
          Safety-grounded decision support for certified aircraft maintenance engineers and airport maintenance control. Logbook records reflect historical recorded actions and do not supersede FAA-approved Aircraft Maintenance Manuals (AMM).
        </p>
      </footer>
    </div>
  );
}
