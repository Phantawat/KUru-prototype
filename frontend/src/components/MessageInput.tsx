"use client";

import { Loader2, SendHorizonal } from "lucide-react";
import { KeyboardEvent, useRef } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export function MessageInput({
  value,
  onChange,
  onSubmit,
  disabled = false,
}: MessageInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const resize = () => {
    const el = textareaRef.current;
    if (!el) {
      return;
    }
    el.style.height = "auto";
    const maxHeight = 4 * 24 + 16;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!disabled) {
        onSubmit();
      }
    }
  };

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-ku-mint-border bg-white p-2 shadow-sm">
      <Textarea
        ref={textareaRef}
        rows={1}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          onChange(event.target.value);
          resize();
        }}
        onKeyDown={handleKeyDown}
        placeholder="พิมพ์คำถามเกี่ยวกับหลักสูตร KU หรือ Tell KUru what you want to study"
        className="min-h-10 max-h-28 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0"
      />
      <Button
        type="button"
        size="icon"
        disabled={disabled || value.trim().length === 0}
        onClick={onSubmit}
        className="h-10 w-10 rounded-xl"
      >
        {disabled ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <SendHorizonal className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}
