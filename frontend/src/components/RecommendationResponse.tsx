"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { RecommendationResponse as RecommendationResponseType } from "@/lib/types";
import { cn } from "@/lib/utils";

interface RecommendationResponseProps {
  data: RecommendationResponseType;
}

const PROGRAM_BADGE_CLASS: Record<string, string> = {
  CPE: "bg-ku-green text-white",
  CS: "bg-blue-600 text-white",
  SKE: "bg-amber-500 text-slate-900",
};

export function RecommendationResponse({ data }: RecommendationResponseProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const detected = Object.entries(data.interests)
    .filter(([, weight]) => weight > 0)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="max-w-3xl space-y-3">
      <Card className="border-ku-mint-border bg-white/95">
        <CardContent className="p-4 text-sm text-slate-700">
          <span className="font-medium">Detected interests:</span>{" "}
          {detected.length > 0
            ? detected.map(([topic, weight]) => `${topic} (${weight.toFixed(2)})`).join(", ")
            : "None"}
        </CardContent>
      </Card>

      {data.programs.map((program, index) => {
        const rank = index + 1;
        const pct = Math.round(program.fit_score * 100);
        const rankClass =
          rank === 1
            ? "bg-ku-green text-white"
            : rank === 2
              ? "bg-ku-gold text-slate-900"
              : "bg-slate-200 text-slate-700";

        return (
          <Card
            key={program.program_id}
            className={cn(
              "border bg-white/95",
              rank === 1 ? "border-[1.5px] border-ku-green" : "border-ku-mint-border"
            )}
          >
            <CardContent className="space-y-3 p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div
                    className={cn(
                      "mt-0.5 flex h-8 w-8 items-center justify-center rounded-full text-sm font-semibold",
                      rankClass
                    )}
                  >
                    #{rank}
                  </div>
                  <div>
                    <p className="text-base font-semibold text-slate-900">{program.program_name_th}</p>
                    <Badge className={cn("mt-1", PROGRAM_BADGE_CLASS[program.program_id] ?? "bg-slate-300")}>
                      {program.program_id}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm font-semibold text-ku-green">{pct}%</p>
              </div>

              <div className="relative h-1.5 overflow-hidden rounded-full bg-gray-100">
                <div
                  className="h-full rounded-full bg-[#006633] transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>

              <p className="text-sm leading-relaxed text-slate-700">{program.explanation}</p>

              <div className="flex flex-wrap gap-2">
                {program.top_careers.map((career, idx) => (
                  <Badge key={`${program.program_id}-${idx}`} className="border border-ku-mint-border bg-ku-mint text-ku-green">
                    {career.title_th} · {career.salary_entry}
                  </Badge>
                ))}
              </div>

              <div>
                <button
                  type="button"
                  onClick={() =>
                    setExpanded((prev) => ({
                      ...prev,
                      [program.program_id]: !prev[program.program_id],
                    }))
                  }
                  className="text-xs font-medium text-ku-green underline underline-offset-2"
                >
                  PLO highlights
                </button>
                {expanded[program.program_id] ? (
                  <ul className="mt-2 space-y-1 text-sm text-slate-700">
                    {program.plo_highlights.map((item, idx) => (
                      <li key={`${program.program_id}-plo-${idx}`}>• {item}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </CardContent>
          </Card>
        );
      })}

      <p className="text-xs text-slate-500">generation {Math.round(data.generation_ms)}ms</p>
    </div>
  );
}
