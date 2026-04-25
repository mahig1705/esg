import { cn } from "@/lib/utils";

const map = {
  CRITICAL: "bg-risk-critical/12 text-risk-critical border-risk-critical/30",
  HIGH: "bg-risk-high/12 text-risk-high border-risk-high/30",
  MEDIUM: "bg-risk-medium/12 text-risk-medium border-risk-medium/30",
  LOW: "bg-risk-low/12 text-risk-low border-risk-low/30",
  MINIMAL: "bg-risk-minimal/12 text-risk-minimal border-risk-minimal/30",
} as const;

export function RiskBadge({ level, className }: { level: keyof typeof map; className?: string }) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-mono font-medium tracking-wider",
      map[level], className
    )}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {level}
    </span>
  );
}
