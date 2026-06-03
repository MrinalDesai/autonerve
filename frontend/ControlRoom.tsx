"use client";
// AutoNerve — Control Room (Mumbai Plant 1)
// Pattern-setter for Day 1: fetch /scenario, render KPIs + critical alert,
// and fire the live closed loop on /loop. Build the other screens the same way.
// Drop into a Next.js (app router) project. Tailwind assumed. See README.

import { useEffect, useState } from "react";

const API = "http://localhost:8000";

type Loop = {
  recommendation: { headline: string; costPct: number; chinaDepReductionPct: number; cpkImpactMax: number; cta: string };
  risk: { topSupplier: string; score: number; exposureCr: number };
};

export default function ControlRoom() {
  const [s, setS] = useState<any>(null);
  const [loop, setLoop] = useState<Loop | null>(null);
  const [firing, setFiring] = useState(false);

  useEffect(() => {
    fetch(`${API}/scenario`).then((r) => r.json()).then(setS).catch(() => {});
  }, []);

  async function fireLoop() {
    setFiring(true);
    try {
      const r = await fetch(`${API}/loop`, { method: "POST" });
      setLoop(await r.json());
    } finally {
      setFiring(false);
    }
  }

  if (!s) return <div className="p-8 text-zinc-400">Connecting to AutoNerve…</div>;

  const k = s.kpis;
  const kpiCards = [
    { label: "Supply Risk Index", value: `${k.supplyRiskIndex.value}/100`, sub: `▲ ${k.supplyRiskIndex.delta} ${k.supplyRiskIndex.deltaLabel}` },
    { label: "Open POs at Risk", value: `${k.openPOsAtRisk.value} of ${k.openPOsAtRisk.of}`, sub: `₹${k.openPOsAtRisk.exposureCr} Cr exposure` },
    { label: "Plant FPY (today)", value: `${k.plantFPY.value}%`, sub: `▼ ${Math.abs(k.plantFPY.delta)} ${k.plantFPY.deltaLabel}` },
    { label: "kWh / Unit", value: `${k.kwhPerUnit.value} kWh`, sub: `▲ ${k.kwhPerUnit.delta} ${k.kwhPerUnit.deltaLabel}` },
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-8 space-y-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Control Room — {s.plant.name}</h1>
        <span className="text-emerald-400 text-sm">Live · last refresh {s.plant.lastRefreshSec}s ago</span>
      </header>

      <div className="grid grid-cols-4 gap-4">
        {kpiCards.map((c) => (
          <div key={c.label} className="rounded-xl bg-zinc-900 border border-zinc-800 p-5">
            <div className="text-xs uppercase tracking-wide text-zinc-500">{c.label}</div>
            <div className="text-3xl font-bold mt-2">{c.value}</div>
            <div className="text-sm text-zinc-400 mt-1">{c.sub}</div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-red-900/60 bg-red-950/40 p-5">
        <div className="text-red-400 text-xs font-semibold tracking-wide">
          🚨 CRITICAL ALERT · {s.event.ageMinutes} minutes ago
        </div>
        <div className="text-lg font-medium mt-1">{s.event.headline}</div>
        <div className="text-sm text-zinc-400 mt-1">
          Source: {s.event.source} · cross-ref {s.event.crossRef}
        </div>
        <div className="text-sm mt-3 text-zinc-300">
          {s.event.linkedImpact.bomPartsAffected} BOM parts ·{" "}
          {s.event.linkedImpact.tier1SuppliersAffected.join(", ")} · ₹
          {s.event.linkedImpact.exposureCr} Cr in next {s.event.linkedImpact.horizonDays} days
        </div>
        <button
          onClick={fireLoop}
          disabled={firing}
          className="mt-4 rounded-lg bg-red-600 hover:bg-red-500 px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {firing ? "Reasoning…" : "Find Alternates → run closed loop"}
        </button>
      </div>

      {loop && (
        <div className="rounded-xl border border-emerald-900/60 bg-emerald-950/30 p-5">
          <div className="text-emerald-400 text-xs font-semibold tracking-wide">AI SUGGESTED ACTION</div>
          <div className="text-lg font-medium mt-1">{loop.recommendation.headline}</div>
          <div className="text-sm text-zinc-300 mt-2">
            Projected: +{loop.recommendation.costPct}% cost · −
            {loop.recommendation.chinaDepReductionPct}% China dep · Cpk impact ≤{" "}
            {loop.recommendation.cpkImpactMax}
          </div>
          <button className="mt-4 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-4 py-2 text-sm font-medium">
            {loop.recommendation.cta}
          </button>
        </div>
      )}
    </div>
  );
}
