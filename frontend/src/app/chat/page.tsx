"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Bot } from "lucide-react";

import { ChatMessage } from "@/components/ChatMessage";
import { InterestChips } from "@/components/InterestChips";
import { MessageInput } from "@/components/MessageInput";
import { Sidebar } from "@/components/Sidebar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  askRAG,
  BackendUnavailableError,
  recommend,
  RateLimitError,
} from "@/lib/api";
import type { Message, RecommendationResponse, RAGResponse } from "@/lib/types";

type Tab = "chat" | "recommend" | "tcas";
type ProgramFilter = "CPE" | "CS" | "SKE" | null;

type InterestTopic =
  | "programming"
  | "ai_data"
  | "design_ux"
  | "systems_hardware"
  | "math_theory"
  | "business_product"
  | "security_networks"
  | "research_science";

const TAB_ITEMS: Array<{ id: Tab; label: string }> = [
  { id: "chat", label: "Chat" },
  { id: "recommend", label: "Recommend" },
  { id: "tcas", label: "TCAS" },
];

const TCAS_QUERIES: Array<{ id: ProgramFilter extends infer T ? T : never; label: string; query: string }> = [
  { id: "CPE", label: "CPE TCAS", query: "ข้อมูลการรับเข้า TCAS ของ CPE" },
  { id: "CS", label: "CS TCAS", query: "ข้อมูลการรับเข้า TCAS ของ CS" },
  { id: "SKE", label: "SKE TCAS", query: "ข้อมูลการรับเข้า TCAS ของ SKE" },
];

function detectIntent(message: string, selectedChips: string[]): "rag" | "recommend" {
  if (selectedChips.length > 0) {
    return "recommend";
  }
  const recKeywords = [
    "อยากเรียน",
    "แนะนำ",
    "ชอบ",
    "สนใจ",
    "เหมาะ",
    "ถนัด",
    "recommend",
    "suggest",
    "interested",
    "i like",
    "i enjoy",
    "which program",
    "what should i study",
  ];
  const lower = message.toLowerCase();
  if (recKeywords.some((kw) => lower.includes(kw))) {
    return "recommend";
  }
  return "rag";
}

function extractInterests(message: string, selectedChips: string[]) {
  const text = message.toLowerCase();
  const map: Record<InterestTopic, number> = {
    programming: 0,
    ai_data: 0,
    design_ux: 0,
    systems_hardware: 0,
    math_theory: 0,
    business_product: 0,
    security_networks: 0,
    research_science: 0,
  };

  const keywords: Array<[InterestTopic, string[]]> = [
    ["programming", ["code", "coding", "programming", "develop", "เขียนโปรแกรม"]],
    ["ai_data", ["ai", "machine learning", "data", "analytics", "ปัญญาประดิษฐ์"]],
    ["design_ux", ["design", "ux", "ui", "creative", "ออกแบบ"]],
    ["systems_hardware", ["hardware", "embedded", "iot", "systems", "อุปกรณ์"]],
    ["math_theory", ["math", "theory", "algorithm", "logic", "คณิต"]],
    ["business_product", ["business", "product", "startup", "management", "ธุรกิจ"]],
    ["security_networks", ["security", "network", "cyber", "pentest", "ความปลอดภัย"]],
    ["research_science", ["research", "science", "lab", "thesis", "วิจัย"]],
  ];

  keywords.forEach(([topic, words]) => {
    if (words.some((word) => text.includes(word))) {
      map[topic] = Math.max(map[topic], 0.7);
    }
  });

  selectedChips.forEach((chip) => {
    if (chip in map) {
      map[chip as InterestTopic] = Math.max(map[chip as InterestTopic], 0.8);
    }
  });

  return map;
}

function replaceLoadingMessage(messages: Message[], next: Message): Message[] {
  const idx = messages.findIndex(
    (message) => message.role === "assistant" && message.type === "loading"
  );
  if (idx === -1) {
    return [...messages, next];
  }
  const copied = [...messages];
  copied[idx] = next;
  return copied;
}

export default function ChatPage() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [programFilter, setProgramFilter] = useState<ProgramFilter>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedChips, setSelectedChips] = useState<string[]>([]);
  const [modelName, setModelName] = useState("KUru");
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [retryCountdown, setRetryCountdown] = useState<number | null>(null);

  const retryActionRef = useRef<(() => Promise<void>) | null>(null);

  useEffect(() => {
    if (retryCountdown === null) {
      return;
    }

    if (retryCountdown <= 0) {
      const action = retryActionRef.current;
      retryActionRef.current = null;
      setRetryCountdown(null);
      if (action) {
        void action();
      }
      return;
    }

    const timer = window.setTimeout(() => {
      setRetryCountdown((prev) => (prev === null ? null : prev - 1));
    }, 1000);

    return () => window.clearTimeout(timer);
  }, [retryCountdown]);

  const isLoading = useMemo(
    () => messages.some((message) => message.role === "assistant" && message.type === "loading"),
    [messages]
  );

  const toggleChip = (topic: string) => {
    setSelectedChips((prev) =>
      prev.includes(topic) ? prev.filter((item) => item !== topic) : [...prev, topic]
    );
  };

  const executeWithRetry = async (action: () => Promise<void>) => {
    try {
      setErrorBanner(null);
      await action();
    } catch (error) {
      if (error instanceof RateLimitError) {
        const seconds = Math.max(1, Math.round(error.retryAfterSeconds));
        retryActionRef.current = () => executeWithRetry(action);
        setRetryCountdown(seconds);
        setErrorBanner(`Rate limit reached — retrying in ${seconds}s...`);
        return;
      }

      setMessages((prev) =>
        prev.filter(
          (message) => !(message.role === "assistant" && message.type === "loading")
        )
      );

      if (error instanceof BackendUnavailableError) {
        setErrorBanner(
          "Cannot connect to backend. Make sure the FastAPI server is running: uvicorn main:app --reload --port 8000"
        );
        return;
      }

      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorBanner(message);
    }
  };

  useEffect(() => {
    if (retryCountdown !== null) {
      setErrorBanner(`Rate limit reached — retrying in ${retryCountdown}s...`);
    }
  }, [retryCountdown]);

  const submitWithText = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) {
      return;
    }

    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed },
      { role: "assistant", type: "loading" },
    ]);

    const callIntent = async () => {
      const mode =
        activeTab === "recommend"
          ? "recommend"
          : activeTab === "tcas"
            ? "rag"
            : detectIntent(trimmed, selectedChips);

      if (mode === "recommend") {
        const interests = extractInterests(trimmed, selectedChips);
        const data: RecommendationResponse = await recommend(interests);
        const next: Message = { role: "assistant", type: "recommendation", data };
        setMessages((prev) => replaceLoadingMessage(prev, next));
        setModelName("RecommendationPipeline");
        return;
      }

      const data: RAGResponse = await askRAG(trimmed, programFilter ?? undefined);
      const next: Message = { role: "assistant", type: "rag", data };
      setMessages((prev) => replaceLoadingMessage(prev, next));
      setModelName(data.model_used);
    };

    await executeWithRetry(callIntent);
  };

  const handleSubmit = async () => {
    await submitWithText(input);
    setInput("");
  };

  const runTcasShortcut = async (filter: ProgramFilter, query: string) => {
    setActiveTab("tcas");
    setProgramFilter(filter);
    await submitWithText(query);
  };

  return (
    <main className="flex h-screen overflow-hidden">
      <div className="hidden lg:block lg:w-[260px]">
        <Sidebar
          programFilter={programFilter}
          onFilterChange={setProgramFilter}
          messages={messages}
        />
      </div>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-ku-mint-border/80 bg-white/85 px-4 py-3 backdrop-blur">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              {TAB_ITEMS.map((tab) => (
                <Button
                  key={tab.id}
                  variant={activeTab === tab.id ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </Button>
              ))}
            </div>
            <Badge className="border border-ku-mint-border bg-ku-mint text-ku-green">
              <Bot className="mr-1 h-3.5 w-3.5" />
              {modelName}
            </Badge>
          </div>
        </header>

        {errorBanner ? (
          <div className="mx-4 mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            <span className="inline-flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              {errorBanner}
            </span>
          </div>
        ) : null}

        {activeTab === "tcas" ? (
          <div className="mx-4 mt-3 flex flex-wrap gap-2">
            {TCAS_QUERIES.map((item) => (
              <Button
                key={item.label}
                variant="outline"
                onClick={() => void runTcasShortcut(item.id as ProgramFilter, item.query)}
              >
                {item.label}
              </Button>
            ))}
          </div>
        ) : null}

        <div className="min-h-0 flex-1 p-4">
          <ScrollArea className="h-full pr-2">
            <div className="space-y-3 pb-4">
              {messages.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-ku-mint-border bg-white/70 p-6 text-sm text-slate-600">
                  Ask KUru anything about curriculum outcomes, PLO alignment, or what program fits you best.
                </div>
              ) : (
                messages.map((message, index) => (
                  <ChatMessage key={`${message.role}-${index}`} message={message} />
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        <div className="space-y-3 border-t border-ku-mint-border/80 bg-white/70 p-4">
          <InterestChips
            selected={selectedChips}
            onToggle={toggleChip}
            prominent={activeTab === "recommend"}
          />
          <MessageInput
            value={input}
            onChange={setInput}
            onSubmit={() => void handleSubmit()}
            disabled={isLoading}
          />
        </div>
      </section>
    </main>
  );
}
