import type { RAGResponse, RecommendationResponse } from "@/lib/types";

const BASE = "http://localhost:8000";

export class RateLimitError extends Error {
  readonly retryAfterSeconds: number;

  constructor(retryAfterSeconds = 60) {
    super("Rate limit reached");
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export class BackendUnavailableError extends Error {
  constructor() {
    super(
      "Cannot connect to backend. Make sure the FastAPI server is running: uvicorn main:app --reload --port 8000"
    );
  }
}

async function apiFetch<T>(path: string, body: unknown): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch {
    throw new BackendUnavailableError();
  }

  if (response.status === 429) {
    const retryAfter = Number(response.headers.get("Retry-After") ?? "60");
    throw new RateLimitError(Number.isFinite(retryAfter) ? retryAfter : 60);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API error: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function askRAG(
  question: string,
  programFilter?: string
): Promise<RAGResponse> {
  return apiFetch<RAGResponse>("/ask", {
    question,
    program_filter: programFilter ?? null,
  });
}

export async function recommend(
  interests: Record<string, number>
): Promise<RecommendationResponse> {
  return apiFetch<RecommendationResponse>("/recommend", {
    interests,
  });
}
