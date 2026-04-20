"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  toolCalls?: { function: string; arguments: Record<string, unknown>; result: unknown }[];
}

export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const res = await apiFetch("/agent/chat", {
        method: "POST",
        body: JSON.stringify({ message: userMsg }),
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.response,
          toolCalls: res.tool_calls,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err instanceof Error ? err.message : "Unknown error"}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const suggestedQueries = [
    "Summarize the experiment results",
    "Which arms should we retire?",
    "Are there signs of non-stationarity?",
    "What's the optimal incentive for Zone 0?",
    "Which Dasher segments are most responsive?",
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-3 p-4 min-h-[300px] max-h-[500px]">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <Bot className="mx-auto h-10 w-10 text-[var(--muted-foreground)] mb-3" />
            <p className="text-sm text-[var(--muted-foreground)] mb-4">
              Ask the experiment agent about bandit performance, optimal incentives, or causal insights.
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {suggestedQueries.map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setInput(q);
                  }}
                  className="rounded-full border border-[var(--border)] bg-[var(--muted)] px-3 py-1.5 text-xs text-[var(--muted-foreground)] hover:bg-[var(--border)] transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "justify-end" : ""}`}>
            {msg.role === "assistant" && (
              <div className="flex-shrink-0 w-7 h-7 rounded-full bg-purple-500/20 flex items-center justify-center">
                <Bot className="h-4 w-4 text-purple-400" />
              </div>
            )}
            <div
              className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-[var(--primary)] text-white"
                  : "bg-[var(--muted)]"
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div className="mt-2 pt-2 border-t border-white/10">
                  <div className="text-[10px] text-[var(--muted-foreground)] font-medium mb-1">
                    Tool calls:
                  </div>
                  {msg.toolCalls.map((tc, j) => (
                    <div key={j} className="text-[10px] text-[var(--muted-foreground)] font-mono">
                      {tc.function}({JSON.stringify(tc.arguments)})
                    </div>
                  ))}
                </div>
              )}
            </div>
            {msg.role === "user" && (
              <div className="flex-shrink-0 w-7 h-7 rounded-full bg-[var(--primary)]/20 flex items-center justify-center">
                <User className="h-4 w-4 text-[var(--primary)]" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-2.5">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-purple-500/20 flex items-center justify-center">
              <Loader2 className="h-4 w-4 text-purple-400 animate-spin" />
            </div>
            <div className="rounded-xl bg-[var(--muted)] px-4 py-2.5 text-sm text-[var(--muted-foreground)]">
              Analyzing experiment data...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-[var(--border)] p-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask about experiment results..."
            className="flex-1 rounded-lg bg-[var(--muted)] border border-[var(--border)] px-3 py-2 text-sm outline-none focus:border-[var(--primary)] transition-colors placeholder:text-[var(--muted-foreground)]"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="rounded-lg bg-[var(--primary)] px-3 py-2 text-white hover:bg-[var(--primary)]/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
