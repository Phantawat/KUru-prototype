"use client";

import type { Message } from "@/lib/types";

import { RAGResponse } from "@/components/RAGResponse";
import { RecommendationResponse } from "@/components/RecommendationResponse";

interface ChatMessageProps {
  message: Message;
}

function LoadingBubble() {
  return (
    <div className="inline-flex items-center gap-1 rounded-full border border-ku-mint-border bg-white px-4 py-2">
      <span className="typing-dot h-2 w-2 rounded-full bg-ku-green" />
      <span className="typing-dot h-2 w-2 rounded-full bg-ku-green" />
      <span className="typing-dot h-2 w-2 rounded-full bg-ku-green" />
    </div>
  );
}

export function ChatMessage({ message }: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl rounded-2xl rounded-br-sm bg-ku-green px-4 py-3 text-sm text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.type === "loading") {
    return (
      <div className="flex justify-start">
        <LoadingBubble />
      </div>
    );
  }

  if (message.type === "rag") {
    return (
      <div className="flex justify-start">
        <RAGResponse data={message.data} />
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <RecommendationResponse data={message.data} />
    </div>
  );
}
