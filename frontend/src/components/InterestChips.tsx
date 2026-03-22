"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const INTEREST_OPTIONS = [
  { key: "programming", label: "programming" },
  { key: "ai_data", label: "ai / data" },
  { key: "design_ux", label: "design / UX" },
  { key: "systems_hardware", label: "hardware" },
  { key: "math_theory", label: "math / theory" },
  { key: "business_product", label: "business" },
  { key: "security_networks", label: "security" },
  { key: "research_science", label: "research" },
] as const;

interface InterestChipsProps {
  selected: string[];
  onToggle: (topic: string) => void;
  prominent?: boolean;
}

export function InterestChips({
  selected,
  onToggle,
  prominent = false,
}: InterestChipsProps) {
  return (
    <div className={cn("flex flex-wrap gap-2", prominent && "gap-2.5") }>
      {INTEREST_OPTIONS.map((option) => {
        const isSelected = selected.includes(option.key);
        return (
          <button
            key={option.key}
            type="button"
            onClick={() => onToggle(option.key)}
            className="text-left"
          >
            <Badge
              className={cn(
                "cursor-pointer rounded-full border border-ku-mint-border bg-white px-3 py-1.5 text-xs text-slate-700 transition",
                isSelected && "border-ku-green bg-ku-mint text-ku-green",
                prominent && "px-4 py-2 text-sm"
              )}
            >
              {option.label}
            </Badge>
          </button>
        );
      })}
    </div>
  );
}
