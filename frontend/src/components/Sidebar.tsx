"use client";

import type { Message } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SidebarProps {
  programFilter: "CPE" | "CS" | "SKE" | null;
  onFilterChange: (filter: "CPE" | "CS" | "SKE" | null) => void;
  messages: Message[];
}

const FILTERS: Array<{ label: string; value: "CPE" | "CS" | "SKE" | null }> = [
  { label: "CPE", value: "CPE" },
  { label: "CS", value: "CS" },
  { label: "SKE", value: "SKE" },
  { label: "All", value: null },
];

export function Sidebar({ programFilter, onFilterChange, messages }: SidebarProps) {
  const history = messages
    .filter((message) => message.role === "assistant" && message.type !== "loading")
    .map((message, index) => {
      if (message.type === "rag") {
        return {
          id: `rag-${index}`,
          color: "bg-ku-green",
          title: message.data.question,
        };
      }
      return {
        id: `rec-${index}`,
        color: "bg-ku-gold",
        title: message.data.programs[0]?.program_name_th ?? "Recommendation",
      };
    });

  return (
    <aside className="flex h-full w-full flex-col bg-ku-green-dark text-white lg:w-[260px]">
      <div className="border-b border-white/10 p-5">
        <h1 className="text-[22px] font-bold tracking-tight">KUru</h1>
        <p className="mt-1 text-[11px] text-ku-gold">PLO-to-Career Navigator · KU</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {FILTERS.map((filter) => {
            const active = filter.value === programFilter;
            return (
              <button
                key={filter.label}
                type="button"
                onClick={() => onFilterChange(filter.value)}
                className={cn(
                  "rounded-full border px-3 py-1 text-xs transition",
                  active
                    ? "border-ku-gold bg-ku-gold text-ku-green-dark"
                    : "border-white/20 bg-white/5 text-white/90 hover:border-ku-gold/50"
                )}
              >
                {filter.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3">
        <p className="mb-2 text-xs uppercase tracking-wide text-white/60">Chat History</p>
        <ul className="space-y-2">
          {history.length === 0 ? (
            <li className="rounded-lg border border-white/10 bg-white/5 p-2 text-xs text-white/60">
              No messages yet
            </li>
          ) : (
            history.map((item) => (
              <li key={item.id} className="rounded-lg border border-white/10 bg-white/5 p-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className={cn("h-2.5 w-2.5 rounded-full", item.color)} />
                  <span className="line-clamp-2 text-white/90">{item.title}</span>
                </div>
              </li>
            ))
          )}
        </ul>
      </div>

      <footer className="border-t border-white/10 p-4 text-xs text-white/70">KU · มก. · AY 2568</footer>
    </aside>
  );
}
