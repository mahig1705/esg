"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { BrainCircuit, CheckCircle2, Download, Loader2, RefreshCw, TerminalSquare } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

type AppState = "input" | "processing" | "results";

type LogEntry = {
  message: string;
  level: "info" | "warn" | "error" | "success";
  source: "stdout" | "stderr";
  ts: string;
};

type AnalysisResult = {
  status: string;
  company_name: string;
  claim: string;
  industry: string;
  risk_level: string;
  confidence: number;
  summary: string;
  report_markdown: string;
  report_file_name: string;
  report_created_at: string;
  parsed_main_report?: {
    company: string;
    ticker: string;
    industry: string;
    claim: string;
    reportDate: string;
    reportConfidence: string;
    riskScore: string;
    esgRating: string;
    riskBand: string;
    confidence: string;
    keyDetails: Array<{ key: string; value: string }>;
    narrative: string[];
  };
  json_report?: Record<string, unknown> | null;
  json_file_name?: string;
};

function badgeClass(level: LogEntry["level"]): string {
  if (level === "error") return "bg-red-100 text-red-700";
  if (level === "warn") return "bg-amber-100 text-amber-700";
  if (level === "success") return "bg-emerald-100 text-emerald-700";
  return "bg-neutral-100 text-neutral-700";
}

function parsePercentText(value?: string): number {
  if (!value) return 0;
  const match = value.match(/(\d+(?:\.\d+)?)/);
  if (!match) return 0;
  return Number(match[1]) / 100;
}

function isUsefulText(value?: string): boolean {
  if (!value) return false;
  const normalized = value.trim();
  return !!normalized && !/^n\/?a$/i.test(normalized) && !/^unknown$/i.test(normalized);
}

function getDisplayRisk(results: AnalysisResult): string {
  const risk = results.risk_level;
  if (isUsefulText(risk)) return risk;
  const fallback = results.parsed_main_report?.riskBand;
  return isUsefulText(fallback) ? (fallback || "") : "";
}

function getDisplayConfidence(results: AnalysisResult): string {
  const numeric = (results.confidence > 0 ? results.confidence : parsePercentText(results.parsed_main_report?.confidence)) || 0;
  if (numeric > 0) {
    return `${Math.round(numeric * 100)}%`;
  }

  const textFallback = results.parsed_main_report?.reportConfidence;
  return isUsefulText(textFallback) ? (textFallback || "") : "";
}

function saveHistory(result: AnalysisResult): void {
  if (typeof window === "undefined") return;
  const key = "esg-analysis-history";
  const existingRaw = window.localStorage.getItem(key);
  const existing = existingRaw ? (JSON.parse(existingRaw) as AnalysisResult[]) : [];
  const next = [result, ...existing].slice(0, 30);
  window.localStorage.setItem(key, JSON.stringify(next));
}

export default function AnalyzePage() {
  const [appState, setAppState] = useState<AppState>("input");
  const [company, setCompany] = useState("Unilever");
  const [claim, setClaim] = useState("Unilever aims to achieve net-zero emissions across its value chain by 2039.");
  const [industry, setIndustry] = useState("Consumer Goods");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [errorText, setErrorText] = useState<string>("");
  const [results, setResults] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    const defaultIndustry = window.localStorage.getItem("esg-default-industry");
    if (defaultIndustry) {
      setIndustry(defaultIndustry);
    }
  }, []);

  const runStatusLabel = useMemo(() => {
    if (appState === "processing") {
      return "Live run in progress";
    }
    if (appState === "results") {
      return "Run completed";
    }
    return "Ready";
  }, [appState]);

  const downloadFile = (name?: string) => {
    if (!name) return;
    window.open(`/api/reports/download?name=${encodeURIComponent(name)}`, "_blank");
  };

  const downloadPdf = (name?: string) => {
    if (!name) return;
    window.open(`/api/reports/pdf?name=${encodeURIComponent(name)}`, "_blank");
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorText("");
    setResults(null);
    setLogs([]);
    setAppState("processing");

    try {
      const response = await fetch("/api/analyze-company", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company_name: company, claim, industry }),
      });

      if (!response.ok || !response.body) {
        const bodyText = await response.text();
        throw new Error(bodyText || "Unable to start analysis.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let runResult: AnalysisResult | null = null;
      let runError = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
          const lines = chunk.split("\n");
          const eventLine = lines.find((line) => line.startsWith("event:"));
          const dataLine = lines.find((line) => line.startsWith("data:"));
          if (!eventLine || !dataLine) continue;

          const event = eventLine.replace("event:", "").trim();
          const dataJson = dataLine.replace("data:", "").trim();
          const payload = JSON.parse(dataJson) as Record<string, unknown>;

          if (event === "log") {
            setLogs((prev) => {
              const next = [
                ...prev,
                {
                  message: String(payload.message || ""),
                  level: (payload.level as LogEntry["level"]) || "info",
                  source: (payload.source as LogEntry["source"]) || "stdout",
                  ts: String(payload.ts || new Date().toISOString()),
                },
              ];
              return next.slice(-120);
            });
          }

          if (event === "error") {
            runError = String(payload.message || "Analysis failed.");
            setErrorText(runError);
          }

          if (event === "result") {
            const nextResult = payload as unknown as AnalysisResult;
            runResult = nextResult;
            setResults(nextResult);
            saveHistory(nextResult);
          }

          if (event === "end") {
            if (payload.ok && runResult !== null) {
              setAppState("results");
            } else if (!payload.ok) {
              setAppState("input");
            }
          }
        }
      }

      if (!runError && runResult) {
        setAppState("results");
      }
      if (!runResult && !runError) {
        setErrorText("Run completed but no report payload was returned.");
        setAppState("input");
      }
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Analysis request failed");
      setAppState("input");
    }
  };

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
        <h2 className="text-2xl font-bold font-heading text-neutral-900 tracking-tight">Analyze ESG Claim</h2>
        <p className="text-neutral-500 text-sm mt-1">Submit company, claim, and industry to trigger the main LangGraph backend service.</p>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <Card className="xl:col-span-4 border-neutral-200 bg-white shadow-sm h-fit">
          <CardHeader>
            <CardTitle>Analysis Input</CardTitle>
            <CardDescription>Backend run starts only when this form is submitted.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleAnalyze} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="company">Company Name</Label>
                <Input id="company" value={company} onChange={(e) => setCompany(e.target.value)} required />
              </div>

              <div className="space-y-2">
                <Label htmlFor="industry">Industry</Label>
                <Input id="industry" value={industry} onChange={(e) => setIndustry(e.target.value)} required />
              </div>

              <div className="space-y-2">
                <Label htmlFor="claim">Claim</Label>
                <textarea
                  id="claim"
                  value={claim}
                  onChange={(e) => setClaim(e.target.value)}
                  required
                  className="flex min-h-[140px] w-full rounded-md border border-neutral-300 bg-transparent px-3 py-2 text-sm placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                />
              </div>

              <Button type="submit" className="w-full" disabled={appState === "processing"}>
                {appState === "processing" ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Running Analysis
                  </>
                ) : (
                  <>
                    <BrainCircuit className="w-4 h-4 mr-2" /> Run Main Pipeline
                  </>
                )}
              </Button>

              {errorText && <p className="text-sm text-red-600">{errorText}</p>}
            </form>
          </CardContent>
        </Card>

        <div className="xl:col-span-8 space-y-6">
          <Card className="border-neutral-200 bg-white shadow-sm">
            <CardHeader className="pb-2 border-b border-neutral-100">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <TerminalSquare className="w-5 h-5 text-primary-700" /> Live Backend Dashboard
                  </CardTitle>
                  <CardDescription>Important execution logs from main LangGraph run.</CardDescription>
                </div>
                <span className="text-xs font-semibold text-primary-700 bg-primary-50 px-2.5 py-1 rounded-md">
                  {runStatusLabel}
                </span>
              </div>
            </CardHeader>
            <CardContent className="p-4 md:p-5">
              <div className="h-[260px] overflow-auto rounded-lg border border-neutral-200 bg-neutral-950 text-neutral-100 p-3 space-y-2">
                {!logs.length && <div className="text-sm text-neutral-400">No logs yet. Start an analysis to stream events.</div>}
                {logs.map((log, idx) => (
                  <div key={`${log.ts}-${idx}`} className="text-xs leading-relaxed flex gap-2 items-start">
                    <span className={`px-1.5 py-0.5 rounded uppercase tracking-wide text-[10px] font-semibold ${badgeClass(log.level)}`}>
                      {log.level}
                    </span>
                    <span className="text-neutral-300/80">{new Date(log.ts).toLocaleTimeString()}</span>
                    <span className="break-words text-neutral-100">{log.message}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {results && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <Card className="border-neutral-200 bg-white shadow-sm">
                <CardHeader className="pb-3 border-b border-neutral-100">
                  <div className="flex justify-between gap-4 flex-wrap">
                    <div>
                      <CardTitle className="text-xl">{results.company_name} ESG Intelligence Report</CardTitle>
                      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) ? `Risk Level: ${getDisplayRisk(results)}` : "", getDisplayConfidence(results) ? `Confidence: ${getDisplayConfidence(results)}` : ""]
                          .filter(Boolean)
                          .join(" | ") || "Report generated successfully"}
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => downloadFile(results.report_file_name)}>
                        <Download className="w-4 h-4 mr-2" /> Download TXT
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => downloadPdf(results.report_file_name)}>
                        <Download className="w-4 h-4 mr-2" /> Download PDF
                      </Button>
                      <Button size="sm" onClick={() => setAppState("input")}>
                        <RefreshCw className="w-4 h-4 mr-2" /> New Run
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-5">
                  <div className="mb-5 rounded-lg border border-primary-200 bg-primary-50 p-4">
                    <div className="text-sm font-medium text-primary-800">Executive Summary</div>
                    <p className="text-sm text-primary-900 mt-1">{results.summary || "Summary unavailable"}</p>
                  </div>

                  <div className="rounded-lg border border-neutral-200 overflow-hidden">
                    <div className="px-4 py-3 bg-neutral-100 border-b border-neutral-200">
                      <h4 className="font-semibold text-neutral-900">Main ESG Scoring Report</h4>
                    </div>

                    <div className="p-4 md:p-5 space-y-5">
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                        {isUsefulText(results.parsed_main_report?.riskScore) && (
                          <div className="rounded-md border border-neutral-200 p-3 bg-white">
                            <div className="text-xs text-neutral-500">Risk Score</div>
                            <div className="text-lg font-semibold text-neutral-900">{results.parsed_main_report?.riskScore}</div>
                          </div>
                        )}
                        {isUsefulText(results.parsed_main_report?.esgRating) && (
                          <div className="rounded-md border border-neutral-200 p-3 bg-white">
                            <div className="text-xs text-neutral-500">ESG Rating</div>
                            <div className="text-lg font-semibold text-neutral-900">{results.parsed_main_report?.esgRating}</div>
                          </div>
                        )}
                        {isUsefulText(results.parsed_main_report?.riskBand) && (
                          <div className="rounded-md border border-neutral-200 p-3 bg-white">
                            <div className="text-xs text-neutral-500">Risk Band</div>
                            <div className="text-lg font-semibold text-neutral-900">{results.parsed_main_report?.riskBand}</div>
                          </div>
                        )}
                        {isUsefulText(results.parsed_main_report?.confidence) && (
                          <div className="rounded-md border border-neutral-200 p-3 bg-white">
                            <div className="text-xs text-neutral-500">Confidence</div>
                            <div className="text-lg font-semibold text-neutral-900">{results.parsed_main_report?.confidence}</div>
                          </div>
                        )}
                      </div>

                      <div className="rounded-md border border-neutral-200 bg-white overflow-x-auto">
                        <table className="w-full text-sm border-collapse">
                          <thead>
                            <tr className="bg-neutral-100 text-left">
                              <th className="border-b border-neutral-200 px-3 py-2 font-semibold">Field</th>
                              <th className="border-b border-neutral-200 px-3 py-2 font-semibold">Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {[
                              { key: "Company", value: results.parsed_main_report?.company || results.company_name },
                              { key: "Ticker", value: results.parsed_main_report?.ticker },
                              { key: "Industry", value: results.parsed_main_report?.industry || results.industry },
                              { key: "Claim Analyzed", value: results.parsed_main_report?.claim || results.claim },
                              { key: "Report Date", value: results.parsed_main_report?.reportDate || new Date(results.report_created_at).toLocaleString() },
                              { key: "Report Confidence", value: results.parsed_main_report?.reportConfidence },
                            ]
                              .filter((row) => isUsefulText(String(row.value || "")))
                              .map((row, idx, arr) => (
                                <tr key={row.key}>
                                  <td className={`${idx !== arr.length - 1 ? "border-b" : ""} border-neutral-200 px-3 py-2 text-neutral-600`}>{row.key}</td>
                                  <td className={`${idx !== arr.length - 1 ? "border-b" : ""} border-neutral-200 px-3 py-2 text-neutral-900`}>{row.value}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>

                      {!!results.parsed_main_report?.narrative?.length && (
                        <div className="space-y-2">
                          <h5 className="font-semibold text-neutral-900">Narrative</h5>
                          {results.parsed_main_report.narrative.map((paragraph, idx) => (
                            <p key={idx} className="text-sm leading-7 text-neutral-700">{paragraph}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-neutral-200 bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Run Metadata
                  </CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                  <div className="rounded-md border border-neutral-200 p-3">
                    <div className="text-neutral-500">Industry</div>
                    <div className="font-medium text-neutral-900">{results.industry}</div>
                  </div>
                  <div className="rounded-md border border-neutral-200 p-3">
                    <div className="text-neutral-500">Report Generated</div>
                    <div className="font-medium text-neutral-900">{new Date(results.report_created_at).toLocaleString()}</div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
