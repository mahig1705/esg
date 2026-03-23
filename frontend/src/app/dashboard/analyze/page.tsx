"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { BrainCircuit, CheckCircle2, Download, Loader2, RefreshCw, TerminalSquare } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { type AnalysisResult, type LogEntry, useAnalysisRun } from "@/components/providers/analysis-run-provider";

type AgentResult = {
  agent: string;
  status?: string;
  confidence?: number | null;
  key_findings?: Record<string, unknown>;
};

type AgentContribution = {
  agent: string;
  status: string;
  confidence: number;
  contributionPct: number;
  keyFindingsCount: number;
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

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function toObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function titleCaseAgent(agent: string): string {
  return agent
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function getAgentResults(results: AnalysisResult | null): AgentResult[] {
  if (!results?.json_report) return [];

  const reportObj = toObject(results.json_report);
  const raw = reportObj.agent_results ?? reportObj.agent_outputs;
  if (!Array.isArray(raw)) return [];

  return raw
    .map((item) => {
      const row = toObject(item);
      const agent = String(row.agent || "").trim();
      if (!agent) return null;

      return {
        agent,
        status: String(row.status || "UNKNOWN"),
        confidence: toNumber(row.confidence),
        key_findings: toObject(row.key_findings),
      } as AgentResult;
    })
    .filter((item): item is AgentResult => item !== null);
}

function toAgentContributions(agentResults: AgentResult[]): AgentContribution[] {
  const withConfidence = agentResults.map((item) => ({
    ...item,
    confidence: item.confidence ?? 0,
  }));
  const total = withConfidence.reduce((sum, item) => sum + Math.max(0, item.confidence || 0), 0);

  return withConfidence
    .map((item) => ({
      agent: item.agent,
      status: item.status || "UNKNOWN",
      confidence: Math.max(0, item.confidence || 0),
      contributionPct: total > 0 ? (Math.max(0, item.confidence || 0) / total) * 100 : 0,
      keyFindingsCount: Object.keys(item.key_findings || {}).length,
    }))
    .sort((a, b) => b.contributionPct - a.contributionPct);
}

function summarizeClaimExtraction(agentResults: AgentResult[]): {
  extractedClaims: string[];
  totalReportClaims: number;
  chunksProcessed: number;
  chunksSkipped: number;
  yearsDetected: string[];
  fallbackMessage: string;
} {
  const extractorA = agentResults.find((item) => item.agent === "claim_extraction")?.key_findings || {};
  const extractorB = agentResults.find((item) => item.agent === "claim_extractor")?.key_findings || {};

  const claimsRaw = extractorA.claims;
  const extractedClaims = Array.isArray(claimsRaw)
    ? claimsRaw
      .map((entry) => {
        if (typeof entry === "string") return entry.trim();
        const claimFromObj = toObject(entry).claim;
        return typeof claimFromObj === "string" ? claimFromObj.trim() : "";
      })
      .filter((text) => !!text)
    : [];

  const yearsDetectedRaw = extractorB.years_detected;
  const yearsDetected = Array.isArray(yearsDetectedRaw)
    ? yearsDetectedRaw.map((year) => String(year)).filter(Boolean)
    : [];

  return {
    extractedClaims,
    totalReportClaims: toNumber(extractorB.total_report_claims) || 0,
    chunksProcessed: toNumber(extractorB.chunks_processed) || 0,
    chunksSkipped: toNumber(extractorB.chunks_skipped) || 0,
    yearsDetected,
    fallbackMessage: "No extracted claims available from this run. Showing fallback metadata.",
  };
}

export default function AnalyzePage() {
  const [company, setCompany] = useState("Unilever");
  const [claim, setClaim] = useState("Unilever aims to achieve net-zero emissions across its value chain by 2039.");
  const [industry, setIndustry] = useState(() => {
    if (typeof window === "undefined") return "Consumer Goods";
    return window.localStorage.getItem("esg-default-industry") || "Consumer Goods";
  });
  const { appState, logs, results, runStatusLabel, startAnalysis, resetRun } = useAnalysisRun();

  const agentResults = useMemo(() => getAgentResults(results), [results]);
  const agentContributions = useMemo(() => toAgentContributions(agentResults), [agentResults]);
  const claimExtractionTrace = useMemo(() => summarizeClaimExtraction(agentResults), [agentResults]);

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
    await startAnalysis({ company, claim, industry });
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
                      <Button size="sm" onClick={resetRun}>
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

              <Card className="border-neutral-200 bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="text-base">Agent Contribution Breakdown</CardTitle>
                  <CardDescription>Normalized contribution based on each agent confidence from backend output.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {!agentContributions.length && (
                    <p className="text-sm text-neutral-500">No agent contribution data found in json_report.</p>
                  )}
                  {agentContributions.map((agent) => (
                    <div key={agent.agent} className="rounded-md border border-neutral-200 p-3 bg-white">
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <div className="font-medium text-neutral-900">{titleCaseAgent(agent.agent)}</div>
                        <div className="text-neutral-600">
                          {agent.contributionPct.toFixed(1)}% contribution | conf {(agent.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="mt-2 h-2 w-full rounded-full bg-neutral-200 overflow-hidden">
                        <div className="h-full rounded-full bg-primary-600" style={{ width: `${Math.min(100, Math.max(0, agent.contributionPct))}%` }} />
                      </div>
                      <div className="mt-2 text-xs text-neutral-500 flex items-center justify-between">
                        <span>Status: {agent.status}</span>
                        <span>Key findings: {agent.keyFindingsCount}</span>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="border-neutral-200 bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="text-base">Claim Extraction Trace</CardTitle>
                  <CardDescription>How claim extraction agents processed and structured claim evidence.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
                    <div className="rounded-md border border-neutral-200 p-3">
                      <div className="text-neutral-500">Claims Extracted</div>
                      <div className="font-semibold text-neutral-900">{claimExtractionTrace.extractedClaims.length}</div>
                    </div>
                    <div className="rounded-md border border-neutral-200 p-3">
                      <div className="text-neutral-500">Report Claims</div>
                      <div className="font-semibold text-neutral-900">{claimExtractionTrace.totalReportClaims}</div>
                    </div>
                    <div className="rounded-md border border-neutral-200 p-3">
                      <div className="text-neutral-500">Chunks Processed</div>
                      <div className="font-semibold text-neutral-900">{claimExtractionTrace.chunksProcessed}</div>
                    </div>
                    <div className="rounded-md border border-neutral-200 p-3">
                      <div className="text-neutral-500">Chunks Skipped</div>
                      <div className="font-semibold text-neutral-900">{claimExtractionTrace.chunksSkipped}</div>
                    </div>
                  </div>

                  {!!claimExtractionTrace.yearsDetected.length && (
                    <p className="text-sm text-neutral-700">
                      Years detected: {claimExtractionTrace.yearsDetected.join(", ")}
                    </p>
                  )}

                  {claimExtractionTrace.extractedClaims.length > 0 ? (
                    <div className="space-y-2">
                      {claimExtractionTrace.extractedClaims.map((extractedClaim, idx) => (
                        <div key={`${idx}-${extractedClaim.slice(0, 24)}`} className="rounded-md border border-neutral-200 bg-neutral-50 p-3 text-sm text-neutral-800">
                          {extractedClaim}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                      {claimExtractionTrace.fallbackMessage}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="border-neutral-200 bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="text-base">Agent Deep Dive</CardTitle>
                  <CardDescription>Per-agent execution status and compact key findings preview.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {!agentResults.length && <p className="text-sm text-neutral-500">No agent result rows available in report payload.</p>}
                  {agentResults.map((agent) => {
                    const findingEntries = Object.entries(agent.key_findings || {}).filter(([, value]) => {
                      const t = typeof value;
                      return t === "string" || t === "number" || t === "boolean";
                    });

                    return (
                      <div key={agent.agent} className="rounded-md border border-neutral-200 p-3">
                        <div className="flex items-center justify-between gap-3 flex-wrap">
                          <div className="font-medium text-neutral-900">{titleCaseAgent(agent.agent)}</div>
                          <div className="text-xs text-neutral-600">
                            Status: {agent.status || "UNKNOWN"} | Confidence: {agent.confidence !== null && agent.confidence !== undefined ? `${(agent.confidence * 100).toFixed(0)}%` : "N/A"}
                          </div>
                        </div>
                        <div className="mt-2 space-y-1 text-sm text-neutral-700">
                          {findingEntries.length > 0 ? (
                            findingEntries.slice(0, 4).map(([key, value]) => (
                              <div key={`${agent.agent}-${key}`}>
                                <span className="text-neutral-500">{key}:</span> {String(value)}
                              </div>
                            ))
                          ) : (
                            <div className="text-neutral-500">No scalar preview fields in key findings.</div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
