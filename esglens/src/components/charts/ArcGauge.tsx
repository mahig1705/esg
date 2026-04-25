import { motion } from "framer-motion";

export function ArcGauge({ value, size = 140, label }: { value: number; size?: number; label?: string }) {
  const r = size / 2 - 12;
  const cx = size / 2;
  const cy = size / 2;
  const startAngle = Math.PI;
  const endAngle = 2 * Math.PI;
  const valueAngle = startAngle + (value / 100) * Math.PI;

  const arc = (a1: number, a2: number) => {
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy + r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2);
    const y2 = cy + r * Math.sin(a2);
    const large = a2 - a1 > Math.PI ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  const color =
    value >= 60 ? "hsl(var(--risk-low))" :
    value >= 40 ? "hsl(var(--risk-medium))" :
    value >= 25 ? "hsl(var(--risk-high))" :
    "hsl(var(--risk-critical))";

  return (
    <div className="relative inline-block" style={{ width: size, height: size / 2 + 30 }}>
      <svg width={size} height={size / 2 + 12} className="overflow-visible">
        <defs>
          <linearGradient id={`arc-${size}`} x1="0%" x2="100%">
            <stop offset="0%" stopColor="hsl(var(--risk-critical))" />
            <stop offset="50%" stopColor="hsl(var(--risk-medium))" />
            <stop offset="100%" stopColor="hsl(var(--risk-low))" />
          </linearGradient>
        </defs>
        <path d={arc(startAngle, endAngle)} fill="none" stroke="hsl(var(--bg-border))" strokeWidth={6} strokeLinecap="round" />
        <motion.path
          d={arc(startAngle, valueAngle)}
          fill="none"
          stroke={`url(#arc-${size})`}
          strokeWidth={6}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-x-0 top-1/2 -translate-y-1 text-center">
        <div className="font-display text-3xl leading-none" style={{ color }}>{value.toFixed(1)}</div>
        {label && <div className="label-eyebrow mt-1">{label}</div>}
      </div>
    </div>
  );
}
