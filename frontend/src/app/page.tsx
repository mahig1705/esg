"use client";

import React from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { 
  ArrowRight, 
  BrainCircuit, 
  Leaf, 
  ShieldAlert, 
  Activity, 
  Database, 
  FileSearch,
  CheckCircle2,
  Globe,
  Layers,
  Terminal,
  Zap,
  LineChart,
  Lock,
  Workflow,
  Search,
  Settings,
  ShieldCheck,
  FileText,
  BarChart3
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";

// Animation Variants
const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.6 }
};

const staggerContainer = {
  initial: {},
  whileInView: { transition: { staggerChildren: 0.1 } }
};

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-slate-50 selection:bg-emerald-100 font-sans text-slate-900">
      <Navbar />

      {/* --- HERO SECTION --- */}
      <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 overflow-hidden bg-white">
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
          <div className="absolute top-[-10%] right-[-5%] w-[50%] h-[50%] rounded-full bg-emerald-50 blur-[120px]" />
          <div className="absolute bottom-[20%] left-[-10%] w-[40%] h-[60%] rounded-full bg-blue-50 blur-[150px]" />
        </div>

        <div className="container mx-auto px-4 text-center">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            className="max-w-4xl mx-auto space-y-8"
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm font-semibold mb-4">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              Next-Gen Environmental Intelligence
            </div>
            
            <h1 className="text-6xl md:text-8xl font-bold tracking-tighter leading-[1.1] text-slate-950">
              Audit the <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-blue-700">
                Future of ESG
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-slate-600 max-w-2xl mx-auto leading-relaxed font-light">
              We developed a multi-agent AI ecosystem to verify sustainability claims, uncover hidden risks, and bring radical transparency to corporate disclosures.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-8">
              <Button size="lg" className="h-14 px-10 text-lg bg-slate-900 hover:bg-slate-800 text-white group shadow-xl transition-all">
                Analyze ESG Claims <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button variant="outline" size="lg" className="h-14 px-10 text-lg border-slate-200 hover:bg-slate-50">
                Explore Our Science
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* --- MISSION SECTION --- */}
      <section className="py-24 border-y border-slate-100 bg-slate-50/50">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-20 items-center">
            <motion.div {...fadeInUp} className="space-y-8">
              <div className="space-y-4">
                <h2 className="text-4xl font-bold text-slate-900 tracking-tight">
                  Our Mission: Bridging the <br />
                  <span className="text-emerald-600">Integrity Gap</span>
                </h2>
                <p className="text-lg text-slate-600 leading-relaxed">
                  We built this platform to transition ESG evaluation from marketing-led narratives to <strong>evidence-backed certainties</strong>. By leveraging distributed AI agents, we provide the granular verification that institutional stakeholders require.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { title: "Verification", desc: "We cross-check disclosures against 1M+ external news & regulatory data points." },
                  { title: "Risk Mitigation", desc: "We flag greenwashing and contradictory patterns using advanced LLM reasoning." }
                ].map((item, i) => (
                  <div key={i} className="p-5 bg-white rounded-xl border border-slate-200 shadow-sm">
                    <h4 className="font-bold text-slate-900 mb-1">{item.title}</h4>
                    <p className="text-sm text-slate-500">{item.desc}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: 40 }}
              whileInView={{ opacity: 1, x: 0 }}
              className="relative bg-slate-900 rounded-[2.5rem] p-10 text-white shadow-2xl overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-8 opacity-10"><Globe size={180} /></div>
              <h3 className="text-2xl font-bold mb-8 flex items-center gap-3 text-emerald-400">
                <Zap className="w-6 h-6" /> Why We Built This
              </h3>
              <ul className="space-y-8">
                {[
                  "To automate the analysis of complex, unstructured corporate reports.",
                  "To apply specialized AI agents that act as independent auditors.",
                  "To provide explainable SHAP-based scores for audit-ready compliance."
                ].map((text, i) => (
                  <li key={i} className="flex gap-4 items-start">
                    <div className="h-6 w-6 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 mt-1 font-mono text-xs">
                      {i + 1}
                    </div>
                    <p className="text-slate-300 text-lg leading-snug">{text}</p>
                  </li>
                ))}
              </ul>
            </motion.div>
          </div>
        </div>
      </section>

      {/* --- TECH STACK SECTION --- */}
      <section className="py-24 bg-white overflow-hidden">
        <div className="container mx-auto px-4 text-center mb-16">
          <motion.div {...fadeInUp}>
            <h2 className="text-4xl font-bold mb-4 tracking-tight text-slate-900">The Intelligence Architecture</h2>
            <p className="text-slate-500 max-w-2xl mx-auto">We designed a modular stack optimized for high-throughput ESG verification and transparency.</p>
          </motion.div>
        </div>

        <div className="container mx-auto px-4 max-w-6xl">
          <motion.div 
            variants={staggerContainer}
            initial="initial"
            whileInView="whileInView"
            className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            <TechCard 
              icon={<Workflow className="text-emerald-600" />} 
              title="Orchestration" 
              items={["Python Logic Layer", "LangGraph State Control", "Dynamic Agent Routing"]} 
            />
            <TechCard 
              icon={<BrainCircuit className="text-blue-600" />} 
              title="AI & NLP Engine" 
              items={["LLM Claims Extraction", "Sentiment Discrepancy", "Confidence Calibration"]} 
            />
            <TechCard 
              icon={<LineChart className="text-purple-600" />} 
              title="Predictive Models" 
              items={["XGBoost / LightGBM", "SHAP Explainability", "Temporal Risk Forecasting"]} 
            />
            <TechCard 
              icon={<Database className="text-orange-600" />} 
              title="Data Retrieval" 
              items={["Vector RAG Architecture", "Pinecone Evidence Storage", "High-Performance Caching"]} 
            />
            <TechCard 
              icon={<Terminal className="text-slate-600" />} 
              title="Interface" 
              items={["Next.js / TypeScript", "Tailwind + Framer Motion", "Real-time Visualization"]} 
            />
            <TechCard 
              icon={<Lock className="text-red-600" />} 
              title="Safety & Validation" 
              items={["Hallucination Detection", "Bias Mitigation", "Audit-Ready Logs"]} 
            />
          </motion.div>
        </div>
      </section>

      {/* --- HOW IT WORKS SECTION --- */}
      <section className="py-24 bg-slate-900 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-emerald-500/5 blur-[120px] pointer-events-none" />
        <div className="container mx-auto px-4 relative z-10">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold mb-4">How the Pipeline Operates</h2>
            <p className="text-slate-400">We engineered a multi-stage flow to transform raw reports into verifiable intelligence.</p>
          </div>

          <div className="grid md:grid-cols-4 gap-4">
            {[
              { icon: <FileText />, step: "01", title: "Ingestion", desc: "We ingest raw PDF reports, URLs, and financial statements." },
              { icon: <Search />, step: "02", title: "Retrieval", desc: "Our RAG engine pulls contextual evidence from global data sources." },
              { icon: <Settings />, step: "03", title: "Reasoning", desc: "14 autonomous agents stress-test every sustainability claim." },
              { icon: <BarChart3 />, step: "04", title: "Synthesis", desc: "Ensemble models compute a final risk score with SHAP evidence." }
            ].map((item, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="relative p-8 rounded-3xl bg-slate-800/40 border border-slate-700/50 backdrop-blur-sm group hover:border-emerald-500/50 transition-colors"
              >
                <div className="text-emerald-500 mb-4 group-hover:scale-110 transition-transform">{item.icon}</div>
                <div className="text-xs font-mono text-emerald-500/50 mb-2">STAGE {item.step}</div>
                <h4 className="text-xl font-bold mb-3">{item.title}</h4>
                <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
                {i < 3 && <ArrowRight className="hidden lg:block absolute -right-6 top-1/2 -translate-y-1/2 text-slate-700" />}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* --- DASHBOARD PREVIEW --- */}
      <section className="py-24 bg-slate-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4 tracking-tight">Enterprise Visibility</h2>
            <p className="text-slate-500">We provide real-time risk scoring and deep evidence mapping.</p>
          </div>

          <motion.div 
            initial={{ opacity: 0, scale: 0.98, y: 20 }}
            whileInView={{ opacity: 1, scale: 1, y: 0 }}
            className="max-w-5xl mx-auto bg-white rounded-[2.5rem] border border-slate-200 shadow-2xl overflow-hidden"
          >
            <div className="bg-slate-100 px-6 py-4 flex items-center gap-2 border-b">
               <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-400" />
                <div className="w-3 h-3 rounded-full bg-amber-400" />
                <div className="w-3 h-3 rounded-full bg-emerald-400" />
               </div>
               <div className="mx-auto text-[10px] font-mono tracking-widest text-slate-400 uppercase">System Status: Active</div>
            </div>
            
            <div className="p-10 grid md:grid-cols-3 gap-10">
              <div className="space-y-6">
                <div className="p-6 rounded-[2rem] bg-red-50 border border-red-100 shadow-inner">
                  <div className="text-xs uppercase font-bold text-red-600 mb-2">Critical Anomaly</div>
                  <div className="text-4xl font-black text-red-700 tracking-tighter">89% RISK</div>
                  <div className="mt-4 h-2 w-full bg-red-200 rounded-full overflow-hidden">
                    <div className="h-full bg-red-600 w-[89%]" />
                  </div>
                  <p className="mt-4 text-xs text-red-800 font-medium leading-relaxed italic">"Detection of misleading narrative regarding Scope 3 carbon neutrality targets."</p>
                </div>
              </div>

              <div className="md:col-span-2 p-8 rounded-[2rem] border border-slate-100 bg-slate-50/50">
                <h4 className="font-bold mb-8 flex items-center gap-2 text-slate-800">
                  <Activity size={18} className="text-emerald-500" /> Analytical Component Scoring
                </h4>
                <div className="space-y-10">
                  <ScoreBar label="Environmental Integrity" score={42} color="bg-amber-500" />
                  <ScoreBar label="Social & Ethical Disclosure" score={78} color="bg-emerald-500" />
                  <ScoreBar label="Governance Transparency" score={85} color="bg-blue-500" />
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* --- CALL TO ACTION --- */}
      <section className="py-24 bg-white px-4">
        <div className="container mx-auto">
          <Card className="bg-slate-950 border-0 rounded-[3.5rem] overflow-hidden relative p-12 md:p-24 text-center">
            <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-emerald-500 via-transparent to-transparent"></div>
            <CardContent className="relative z-10 space-y-8">
              <h2 className="text-4xl md:text-6xl font-bold text-white tracking-tight leading-tight">
                Ready to transform <br /> ESG from <span className="text-emerald-400">Risk to Reality?</span>
              </h2>
              <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
                We empower teams with the evidence needed to make informed, sustainable decisions.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button size="lg" className="h-16 px-12 bg-emerald-500 hover:bg-emerald-600 text-slate-950 text-lg font-bold shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                  Get Started
                </Button>
                <Button size="lg" variant="outline" className="h-16 px-12 border-slate-700 text-white hover:bg-slate-900 text-lg font-bold">
                  View Case Studies
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      <Footer />
    </main>
  );
}

// --- SUB-COMPONENTS ---

function TechCard({ icon, title, items }: { icon: React.ReactNode, title: string, items: string[] }) {
  return (
    <motion.div 
      variants={fadeInUp}
      whileHover={{ y: -8, boxShadow: "0 20px 25px -5px rgb(0 0 0 / 0.1)" }}
      className="p-8 rounded-[2rem] border border-slate-200 bg-white transition-all group"
    >
      <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-inner">
        {React.cloneElement(icon as React.ReactElement, { size: 24 } as any)}
      </div>
      <h3 className="text-xl font-bold text-slate-900 mb-4 tracking-tight">{title}</h3>
      <ul className="space-y-3">
        {items.map((item, i) => (
          <li key={i} className="text-sm text-slate-500 flex items-center gap-3">
            <ShieldCheck size={14} className="text-emerald-500 shrink-0" />
            {item}
          </li>
        ))}
      </ul>
    </motion.div>
  );
}

function ScoreBar({ label, score, color }: { label: string, score: number, color: string }) {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center text-sm">
        <span className="font-bold text-slate-700 tracking-tight">{label}</span>
        <span className="font-mono font-black text-slate-900">{score}/100</span>
      </div>
      <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden shadow-inner">
        <motion.div 
          initial={{ width: 0 }}
          whileInView={{ width: `${score}%` }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          className={`h-full ${color} shadow-[0_0_10px_rgba(0,0,0,0.1)]`}
        />
      </div>
    </div>
  );
}