import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { getPythonExecutable, getRepoRoot } from "@/lib/server/report-utils";

export const runtime = "nodejs";

function extractJsonPayload(stdout: string): Record<string, unknown> | null {
    const trimmed = stdout.trim();

    try {
        return JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
        // Continue to fallback extraction.
    }

    const first = trimmed.indexOf("{");
    const last = trimmed.lastIndexOf("}");
    if (first >= 0 && last > first) {
        const candidate = trimmed.slice(first, last + 1);
        try {
            return JSON.parse(candidate) as Record<string, unknown>;
        } catch {
            return null;
        }
    }

    return null;
}

function cacheCandidates(repoRoot: string, company: string): string[] {
    const normalized = company.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
    return [
        path.join(repoRoot, "cache", "esg_analysis", `${normalized}.json`),
        path.join(repoRoot, "cache", "esg_analysis", `${company.toLowerCase()}.json`),
    ];
}

export async function POST(req: Request) {
    const { company } = (await req.json()) as { company?: string };

    if (!company || !company.trim()) {
        return Response.json({ error: "Company is required" }, { status: 400 });
    }

    const repoRoot = getRepoRoot();
    const python = getPythonExecutable();
    const args = ["-m", "features.esg_mismatch_detector.pipeline", company.trim()];

    const runResult = await new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
        const child = spawn(python, args, {
            cwd: repoRoot,
            env: {
                ...process.env,
                PYTHONIOENCODING: "utf-8",
            },
        });

        let stdout = "";
        let stderr = "";

        child.stdout.on("data", (chunk: Buffer) => {
            stdout += chunk.toString("utf-8");
        });

        child.stderr.on("data", (chunk: Buffer) => {
            stderr += chunk.toString("utf-8");
        });

        child.on("close", (code) => {
            resolve({ code: code ?? 1, stdout, stderr });
        });
    });

    if (runResult.code !== 0) {
        return Response.json(
            {
                error: "Mismatch detection failed",
                details: runResult.stderr.slice(-2000),
                command: `${python} ${args.join(" ")}`,
            },
            { status: 500 },
        );
    }

    let parsed = extractJsonPayload(runResult.stdout);

    if (!parsed) {
        for (const candidate of cacheCandidates(repoRoot, company.trim())) {
            try {
                const raw = await fs.readFile(candidate, "utf-8");
                parsed = JSON.parse(raw) as Record<string, unknown>;
                break;
            } catch {
                // Try next candidate.
            }
        }
    }

    if (!parsed) {
        return Response.json(
            {
                error: "Mismatch pipeline completed but no JSON payload was detected.",
                command: `${python} ${args.join(" ")}`,
            },
            { status: 500 },
        );
    }

    return Response.json({
        status: "success",
        company: company.trim(),
        command: `python -m features.esg_mismatch_detector.pipeline \"${company.trim()}\"`,
        result: parsed,
    });
}
