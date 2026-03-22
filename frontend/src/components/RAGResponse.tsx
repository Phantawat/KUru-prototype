"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { RAGResponse as RAGResponseType } from "@/lib/types";

interface RAGResponseProps {
  data: RAGResponseType;
}

export function RAGResponse({ data }: RAGResponseProps) {
  return (
    <Card className="max-w-3xl border-ku-mint-border bg-white/95 shadow-sm">
      <CardContent className="space-y-4 p-4">
        <div className="prose prose-sm max-w-none text-slate-800 prose-p:my-2">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.answer}</ReactMarkdown>
        </div>

        <div className="flex flex-wrap gap-2">
          {data.citations.map((citation) => {
            const chunk = data.retrieved.find((item) => item.chunk_id === citation.chunk_id);
            return (
              <Popover key={citation.chunk_id}>
                <PopoverTrigger asChild>
                  <button type="button">
                    <Badge className="cursor-pointer border border-ku-mint-border bg-ku-mint text-ku-green">
                      {`${citation.program_id}-${citation.plo_id}`}
                    </Badge>
                  </button>
                </PopoverTrigger>
                <PopoverContent align="start" className="w-96 space-y-2">
                  <p className="font-mono text-xs text-slate-600">
                    {citation.chunk_id} · {citation.chunk_type}
                  </p>
                  <p className="text-xs text-slate-700">
                    similarity: {(citation.similarity * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs text-slate-700">{chunk?.text.slice(0, 300) ?? "No chunk text"}</p>
                  <Badge className="border border-ku-mint-border bg-ku-mint text-ku-green">
                    {citation.program_id}
                  </Badge>
                </PopoverContent>
              </Popover>
            );
          })}
        </div>

        <p className="text-xs text-slate-500">
          retrieval {Math.round(data.retrieval_ms)}ms · generation {Math.round(data.generation_ms)}ms · {data.retrieved.length} chunks
        </p>
      </CardContent>
    </Card>
  );
}
