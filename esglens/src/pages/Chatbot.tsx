import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Send, Sparkles, Loader2 } from "lucide-react";
import { PageWrapper } from "@/components/layout/PageWrapper";
import { ArcGauge } from "@/components/charts/ArcGauge";
import { RiskBadge } from "@/components/cards/RiskBadge";
import { useChatStore } from "@/stores/chatStore";
import { useAnalysisStore } from "@/stores/analysisStore";

const SUGGESTIONS = [
  "What is the greenwashing risk score?",
  "Explain the top contradictions found",
  "Summarise the regulatory compliance gaps",
  "What are the carbon emissions figures?",
  "What is the ESG rating and why?",
  "Give me an executive summary",
];

export default function Chatbot() {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const { messages, isTyping, sendMessage } = useChatStore();
  const report = useAnalysisStore((s) => s.currentReport);

  const send = async (text: string) => {
    if (!text.trim() || isTyping) return;
    setInput("");
    await sendMessage(text);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <PageWrapper>
      <div className="h-screen flex flex-col">
        <div className="px-10 py-3 border-b border-bg-border bg-bg-surface/40">
          <div className="font-mono text-[11px] text-text-secondary flex items-center gap-2 flex-wrap">
            <Sparkles className="h-3 w-3 text-teal-bright" />
            Grounded in your latest ESG report · Cites evidence · TCFD / FCA aligned
            {report && <span className="text-teal-bright ml-2">· Active: {report.company}</span>}
          </div>
        </div>

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_380px] overflow-hidden">
          <div className="flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto scrollbar-thin px-10 py-8">
              {messages.length === 0 ? (
                <div className="max-w-3xl mx-auto">
                  <div className="text-center mb-10">
                    <div className="label-eyebrow text-teal-bright mb-3">ESGLENS AI</div>
                    <h1 className="font-display text-5xl">How can I help?</h1>
                    <p className="text-text-secondary mt-3">Ask about any analysed company, claim, or framework.</p>
                    {!report && <p className="text-amber-bright text-xs mt-2 font-mono">⚠ No active report — run an analysis first for grounded answers.</p>}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {SUGGESTIONS.map((s, i) => (
                      <motion.button key={s} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
                        onClick={() => send(s)}
                        className="text-left p-4 rounded-xl bg-bg-surface border border-bg-border hover:border-teal-dim hover:bg-teal-bright/5 transition group">
                        <div className="text-sm text-text-primary group-hover:text-teal-bright">{s}</div>
                      </motion.button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="max-w-3xl mx-auto space-y-6">
                  {messages.map((m) => (
                    <motion.div key={m.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                      className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[80%] px-5 py-3 rounded-2xl ${
                        m.role === "user"
                          ? "bg-gradient-to-br from-teal-mid to-teal-bright text-bg-void rounded-br-sm"
                          : "bg-bg-surface border-l-2 border-teal-bright text-text-primary rounded-bl-sm"
                      }`}>
                        <div className="text-sm leading-relaxed whitespace-pre-wrap"
                          dangerouslySetInnerHTML={{ __html: m.content.replace(/\*\*(.+?)\*\*/g, '<strong class="text-teal-bright">$1</strong>') }} />
                        {m.citations && m.citations.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-bg-border/30 text-[10px] font-mono text-text-muted">
                            Sources: {m.citations.slice(0, 3).join(" · ")}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                  {isTyping && (
                    <div className="flex justify-start">
                      <div className="bg-bg-surface border-l-2 border-teal-bright px-5 py-3 rounded-2xl rounded-bl-sm">
                        <Loader2 className="h-4 w-4 text-teal-bright animate-spin" />
                      </div>
                    </div>
                  )}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            <div className="border-t border-bg-border p-6 bg-bg-deep/60">
              <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="max-w-3xl mx-auto">
                <div className="flex items-center gap-3 px-4 py-2 rounded-xl glass border-bg-border focus-within:border-teal-bright transition">
                  <input id="chat-input" value={input} onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask ESGLens about any company, claim, or framework..."
                    className="flex-1 bg-transparent outline-none text-sm py-2" />
                  <button id="chat-send-btn" type="submit" disabled={isTyping || !input.trim()}
                    className="h-9 w-9 rounded-lg bg-teal-bright text-bg-void flex items-center justify-center hover:scale-105 transition disabled:opacity-50">
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </form>
            </div>
          </div>

          <aside className="border-l border-bg-border bg-bg-deep p-6 overflow-y-auto scrollbar-thin">
            <div className="label-eyebrow text-teal-bright mb-2">ACTIVE REPORT →</div>
            {report ? (
              <>
                <div className="font-display text-xl mb-6">{report.company} · {report.ticker}</div>
                <div className="space-y-5">
                  <div className="rounded-xl bg-bg-surface border border-bg-border p-4 text-center">
                    <ArcGauge value={report.esg_score} size={140} label="ESG SCORE" />
                  </div>
                  <div className="rounded-xl bg-bg-surface border border-bg-border p-4">
                    <div className="label-eyebrow mb-2">RISK</div>
                    <RiskBadge level={report.risk_level as "HIGH" | "MEDIUM" | "LOW"} />
                  </div>
                  <div className="rounded-xl bg-bg-surface border border-bg-border p-4">
                    <div className="label-eyebrow mb-2">CARBON · TOTAL</div>
                    <div className="font-display text-3xl text-amber-bright">{(report.carbon.total / 1e9).toFixed(2)}B</div>
                    <div className="font-mono text-[11px] text-text-secondary">tCO2e Scope 1+2+3</div>
                  </div>
                  <div className="rounded-xl bg-risk-high/5 border border-risk-high/30 p-4">
                    <div className="label-eyebrow text-risk-high mb-1">CONTRADICTIONS</div>
                    <div className="font-display text-3xl text-risk-high">
                      {report.contradictions.filter(c => c.severity === "HIGH").length} HIGH
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-text-muted text-sm font-mono">
                No report loaded.<br />
                Run an analysis to enable grounded chatbot responses.
              </div>
            )}
          </aside>
        </div>
      </div>
    </PageWrapper>
  );
}
