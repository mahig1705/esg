import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";

export type AnalysisRequest = {
    companyName: string;
    claim: string;
    industry?: string;
};

export type StoredReport = {
    txtFileName: string;
    jsonFileName?: string;
    txtPath: string;
    jsonPath?: string;
    reportMarkdown: string;
    jsonReport: Record<string, unknown> | null;
    createdAt: string;
};

export function getRepoRoot(): string {
    const cwd = process.cwd();
    const candidates = [
        cwd,
        path.resolve(cwd, ".."),
        path.resolve(cwd, "..", ".."),
    ];

    for (const candidate of candidates) {
        if (fsSync.existsSync(path.join(candidate, "main_langgraph.py"))) {
            return candidate;
        }
    }

    return path.resolve(cwd, "..");
}

export function getReportsDir(): string {
    return path.join(getRepoRoot(), "reports");
}

export function getPythonExecutable(): string {
    const configured = process.env.ESG_PYTHON_EXECUTABLE;
    if (configured) {
        return configured;
    }

    const repoRoot = getRepoRoot();

    const candidates = [
        process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, "Scripts", "python.exe") : "",
        path.join(repoRoot, "venv", "Scripts", "python.exe"),
        path.join(repoRoot, ".venv", "Scripts", "python.exe"),
    ].filter(Boolean);

    for (const candidate of candidates) {
        if (fsSync.existsSync(candidate)) {
            return candidate;
        }
    }

    return "python";
}

export function getAnalysisScriptPath(): string {
    return path.join(getRepoRoot(), "main_langgraph.py");
}

export function sanitizeCompanyForFileName(company: string): string {
    return company.trim().replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function isImportantLogLine(line: string): boolean {
    const normalized = line.trim();
    if (!normalized) {
        return false;
    }

    return /(ANALYZING|Running|COMPLETE|FAILED|ERROR|WARNING|Risk Level|Reports saved|EXECUTIVE SUMMARY|Agent|workflow|timed out|saved|\u26a0\ufe0f|\u274c|\u2705|\ud83d\udd0d|\ud83d\udcca|\ud83d\udcbe|\ud83d\ude80)/i.test(normalized);
}

export function classifyLogLevel(line: string): "info" | "warn" | "error" | "success" {
    if (/(error|failed|traceback|\u274c)/i.test(line)) {
        return "error";
    }
    if (/(warning|timed out|\u26a0\ufe0f)/i.test(line)) {
        return "warn";
    }
    if (/(complete|saved|success|\u2705)/i.test(line)) {
        return "success";
    }
    return "info";
}

async function readIfExists(filePath: string): Promise<string | null> {
    try {
        return await fs.readFile(filePath, "utf-8");
    } catch {
        return null;
    }
}

function tokenizeCompany(companyName: string): string[] {
    return sanitizeCompanyForFileName(companyName)
        .toLowerCase()
        .split("_")
        .filter((token) => token.length >= 3);
}

export async function getLatestStoredReport(companyName: string, minCreatedAtMs?: number): Promise<StoredReport | null> {
    const reportsDir = getReportsDir();
    const companySlug = sanitizeCompanyForFileName(companyName);
    const companyTokens = tokenizeCompany(companyName);

    const fileNames = await fs.readdir(reportsDir).catch(() => [] as string[]);
    const allTxtCandidates = fileNames
        .filter((name) => name.startsWith("ESG_Report_") && name.endsWith(".txt"))
        .map((name) => ({
            name,
            fullPath: path.join(reportsDir, name),
        }));

    const exactCandidates = allTxtCandidates.filter((item) => item.name.startsWith(`ESG_Report_${companySlug}_`) && item.name.endsWith(".txt"));

    const tokenCandidates = allTxtCandidates.filter((item) => {
        const lower = item.name.toLowerCase();
        if (!companyTokens.length) {
            return false;
        }
        const matchedTokens = companyTokens.filter((token) => lower.includes(token));
        return matchedTokens.length >= Math.min(2, companyTokens.length);
    });

    const selectedCandidates = exactCandidates.length
        ? exactCandidates
        : tokenCandidates.length
            ? tokenCandidates
            : allTxtCandidates;

    if (!selectedCandidates.length) {
        return null;
    }

    const withStats = await Promise.all(
        selectedCandidates.map(async (entry) => ({
            ...entry,
            stats: await fs.stat(entry.fullPath),
        })),
    );

    withStats.sort((a, b) => b.stats.mtimeMs - a.stats.mtimeMs);

    let latestTxt = withStats[0];
    if (minCreatedAtMs) {
        const recent = withStats.find((item) => item.stats.mtimeMs >= minCreatedAtMs);
        if (recent) {
            latestTxt = recent;
        }
    }

    const baseName = latestTxt.name.replace(/\.txt$/, "");

    const jsonFileName = `${baseName}.json`;
    const jsonPath = path.join(reportsDir, jsonFileName);
    const reportMarkdown = (await readIfExists(latestTxt.fullPath)) || "";
    const jsonRaw = await readIfExists(jsonPath);

    let jsonReport: Record<string, unknown> | null = null;
    if (jsonRaw) {
        try {
            jsonReport = JSON.parse(jsonRaw) as Record<string, unknown>;
        } catch {
            jsonReport = null;
        }
    }

    return {
        txtFileName: latestTxt.name,
        jsonFileName: jsonRaw ? jsonFileName : undefined,
        txtPath: latestTxt.fullPath,
        jsonPath: jsonRaw ? jsonPath : undefined,
        reportMarkdown,
        jsonReport,
        createdAt: latestTxt.stats.mtime.toISOString(),
    };
}

export type DerivedResult = {
    riskLevel: string;
    confidence: number;
    summary: string;
};

function parsePercentageValue(value: string): number {
    const match = value.match(/(\d+(?:\.\d+)?)/);
    if (!match) {
        return 0;
    }
    const numeric = Number(match[1]);
    if (!Number.isFinite(numeric)) {
        return 0;
    }
    return numeric > 1 ? numeric / 100 : numeric;
}

export function deriveResult(reportMarkdown: string, jsonReport: Record<string, unknown> | null): DerivedResult {
    const finalVerdict = (jsonReport?.final_verdict ?? {}) as Record<string, unknown>;
    const parsedMain = parseMainReport(reportMarkdown);

    const riskFromJson = (finalVerdict.risk_level as string) || (jsonReport?.risk_level as string) || parsedMain.riskBand || "UNKNOWN";
    const confidenceRaw =
        (finalVerdict.final_confidence as number | undefined) ??
        (jsonReport?.confidence as number | undefined) ??
        parsePercentageValue(parsedMain.confidence || parsedMain.reportConfidence);

    const confidence = confidenceRaw > 1 ? confidenceRaw / 100 : confidenceRaw;

    const summaryFromJson =
        (finalVerdict.executive_summary as string) ||
        (jsonReport?.summary as string) ||
        "";

    if (summaryFromJson) {
        return {
            riskLevel: riskFromJson,
            confidence,
            summary: summaryFromJson,
        };
    }

    const markdownLines = reportMarkdown
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);

    const firstParagraph = markdownLines.find((line) => !line.startsWith("#") && line.length > 40) || "";

    return {
        riskLevel: riskFromJson,
        confidence,
        summary: firstParagraph || parsedMain.narrative[0] || "",
    };
}

export type ReportListItem = {
    fileName: string;
    company: string;
    createdAt: string;
    type: "txt" | "json";
    sizeBytes: number;
};

export type ParsedMainReport = {
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

function parseInlineFields(line: string): Array<{ key: string; value: string }> {
    const fields: Array<{ key: string; value: string }> = [];
    const regex = /([A-Za-z][A-Za-z /()-]+):\s*([^:]+?)(?=(?:\s+[A-Za-z][A-Za-z /()-]+:\s)|$)/g;
    let match: RegExpExecArray | null = null;

    while ((match = regex.exec(line)) !== null) {
        fields.push({
            key: match[1].trim(),
            value: match[2].trim(),
        });
    }

    return fields;
}

export function parseMainReport(reportText: string): ParsedMainReport {
    const rawLines = reportText.split(/\r?\n/);
    const lines = rawLines.map((line) => line.trim());

    const fieldMap = new Map<string, string>();
    const detailsMap = new Map<string, string>();
    let activeMultilineKey: string | null = null;

    for (const rawLine of rawLines) {
        const trimmed = rawLine.trim();
        if (!trimmed) {
            continue;
        }

        if (/^={5,}|^[-─]{5,}|^\*{3,}/.test(trimmed)) {
            activeMultilineKey = null;
            continue;
        }

        const fields = parseInlineFields(trimmed);
        if (fields.length > 0) {
            for (const item of fields) {
                const normalizedKey = item.key.toLowerCase();
                fieldMap.set(normalizedKey, item.value);
                detailsMap.set(item.key, item.value);
            }

            const lastKey = fields[fields.length - 1].key.toLowerCase();
            if (lastKey === "claim analyzed") {
                activeMultilineKey = "claim analyzed";
            } else {
                activeMultilineKey = null;
            }
            continue;
        }

        if (activeMultilineKey) {
            const previous = fieldMap.get(activeMultilineKey) || "";
            fieldMap.set(activeMultilineKey, `${previous} ${trimmed}`.trim());
            detailsMap.set("Claim Analyzed", `${previous} ${trimmed}`.trim());
            continue;
        }
    }

    const section1Start = lines.findIndex((line) => /^SECTION\s+1:\s+EXECUTIVE SUMMARY/i.test(line));
    const section2Start = lines.findIndex((line) => /^SECTION\s+2:/i.test(line));
    const narrative: string[] = [];

    if (section1Start >= 0) {
        const end = section2Start > section1Start ? section2Start : lines.length;
        for (let i = section1Start + 1; i < end; i += 1) {
            const line = lines[i];
            if (!line || /^={5,}|^[-─]{5,}/.test(line)) {
                continue;
            }
            narrative.push(line);
        }
    } else {
        for (const line of lines) {
            if (
                line.length > 50 &&
                !line.startsWith("Pipeline:") &&
                !/agent/i.test(line) &&
                !/^REPORT METADATA$/i.test(line)
            ) {
                narrative.push(line);
            }
        }
    }

    const reportDate =
        fieldMap.get("date") ||
        fieldMap.get("analysis date") ||
        "N/A";

    const keyDetails = [...detailsMap.entries()]
        .map(([key, value]) => ({ key, value }))
        .filter((item) => !/pipeline|warnings?/i.test(item.key) && !/agent/i.test(item.key));

    return {
        company: fieldMap.get("company") || "N/A",
        ticker: fieldMap.get("ticker") || "N/A",
        industry: fieldMap.get("industry") || "N/A",
        claim: fieldMap.get("claim analyzed") || "N/A",
        reportDate,
        reportConfidence: fieldMap.get("report confidence") || "N/A",
        riskScore: fieldMap.get("greenwashing risk score") || "N/A",
        esgRating: fieldMap.get("esg rating") || "N/A",
        riskBand: fieldMap.get("risk band") || "N/A",
        confidence: fieldMap.get("confidence") || "N/A",
        keyDetails,
        narrative: narrative.slice(0, 8),
    };
}

export async function listReports(companyName?: string): Promise<ReportListItem[]> {
    const reportsDir = getReportsDir();
    const fileNames = await fs.readdir(reportsDir).catch(() => [] as string[]);

    const filtered = fileNames.filter((name) =>
        /^ESG_Report_.*\.(txt|json)$/i.test(name) && (!companyName || name.includes(sanitizeCompanyForFileName(companyName))),
    );

    const items = await Promise.all(
        filtered.map(async (name) => {
            const fullPath = path.join(reportsDir, name);
            const stat = await fs.stat(fullPath);
            const companyMatch = name.match(/^ESG_Report_(.+)_\d{8}_\d{6}/);
            const company = companyMatch ? companyMatch[1].replace(/_/g, " ") : "Unknown";

            return {
                fileName: name,
                company,
                createdAt: stat.mtime.toISOString(),
                type: name.endsWith(".json") ? "json" : "txt",
                sizeBytes: stat.size,
            } as ReportListItem;
        }),
    );

    return items.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
}
