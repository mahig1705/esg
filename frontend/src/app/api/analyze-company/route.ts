import { spawn } from "node:child_process";
import {
  classifyLogLevel,
  deriveResult,
  getAnalysisScriptPath,
  getLatestStoredReport,
  getPythonExecutable,
  getRepoRoot,
  isImportantLogLine,
  parseMainReport,
} from "@/lib/server/report-utils";

export const runtime = "nodejs";

type StreamEvent = {
  event: "status" | "log" | "result" | "error" | "end";
  payload: Record<string, unknown>;
};

function toSseChunk(event: StreamEvent): string {
  return `event: ${event.event}\ndata: ${JSON.stringify(event.payload)}\n\n`;
}

export async function POST(req: Request) {
  const { company_name, claim, industry } = await req.json();

  if (!company_name || !claim) {
    return new Response(JSON.stringify({ error: "Company name and claim are required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const pushEvent = (event: StreamEvent) => {
        controller.enqueue(encoder.encode(toSseChunk(event)));
      };

      const pythonExecutable = getPythonExecutable();
      const scriptPath = getAnalysisScriptPath();
      const repoRoot = getRepoRoot();
      const analysisStartedAt = Date.now();

      pushEvent({
        event: "status",
        payload: {
          state: "started",
          message: "Analysis started. Running LangGraph pipeline...",
        },
      });

      const child = spawn(
        pythonExecutable,
        [scriptPath, "--company", company_name, "--claim", claim, ...(industry ? ["--industry", industry] : [])],
        {
          cwd: repoRoot,
          env: {
            ...process.env,
            PYTHONIOENCODING: "utf-8",
          },
        },
      );

      let stderrText = "";
      let stdoutBuffer = "";
      let stderrBuffer = "";

      const handleBuffer = (buffer: string, source: "stdout" | "stderr") => {
        const lines = buffer.split(/\r?\n/);
        const remainder = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) {
            continue;
          }

          if (source === "stderr") {
            stderrText += `${trimmed}\n`;
          }

          if (!isImportantLogLine(trimmed)) {
            continue;
          }

          pushEvent({
            event: "log",
            payload: {
              level: classifyLogLevel(trimmed),
              source,
              message: trimmed,
              ts: new Date().toISOString(),
            },
          });
        }

        return remainder;
      };

      child.stdout.on("data", (chunk: Buffer) => {
        stdoutBuffer += chunk.toString("utf8");
        stdoutBuffer = handleBuffer(stdoutBuffer, "stdout");
      });

      child.stderr.on("data", (chunk: Buffer) => {
        stderrBuffer += chunk.toString("utf8");
        stderrBuffer = handleBuffer(stderrBuffer, "stderr");
      });

      const heartbeat = setInterval(() => {
        pushEvent({
          event: "status",
          payload: {
            state: "running",
            message: "Pipeline still running...",
            ts: new Date().toISOString(),
          },
        });
      }, 4500);

      child.on("error", (error) => {
        clearInterval(heartbeat);
        pushEvent({
          event: "error",
          payload: {
            message: `Failed to start Python process: ${error.message}`,
          },
        });
        pushEvent({ event: "end", payload: { ok: false } });
        controller.close();
      });

      child.on("close", async (exitCode) => {
        clearInterval(heartbeat);

        if (exitCode !== 0) {
          pushEvent({
            event: "error",
            payload: {
              message: "Analysis failed. Check logs for details.",
              exitCode,
              details: stderrText.slice(-2000),
            },
          });
          pushEvent({ event: "end", payload: { ok: false } });
          controller.close();
          return;
        }

        const latestReport = await getLatestStoredReport(company_name, analysisStartedAt - 60_000);
        if (!latestReport) {
          pushEvent({
            event: "error",
            payload: {
              message: "Analysis finished, but no report file was found in reports directory.",
              diagnostics: {
                repoRoot,
                company_name,
              },
            },
          });
          pushEvent({ event: "end", payload: { ok: false } });
          controller.close();
          return;
        }

        const derived = deriveResult(latestReport.reportMarkdown, latestReport.jsonReport);
        const parsedMainReport = parseMainReport(latestReport.reportMarkdown);
        pushEvent({
          event: "result",
          payload: {
            status: "success",
            company_name,
            claim,
            industry: industry || "Auto-detected",
            risk_level: derived.riskLevel,
            confidence: derived.confidence,
            summary: derived.summary,
            report_markdown: latestReport.reportMarkdown,
            report_file_name: latestReport.txtFileName,
            report_created_at: latestReport.createdAt,
            parsed_main_report: parsedMainReport,
            json_report: latestReport.jsonReport,
            json_file_name: latestReport.jsonFileName,
          },
        });

        pushEvent({ event: "end", payload: { ok: true } });
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
