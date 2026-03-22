export interface RetrievedChunk {
  chunk_id: string;
  text: string;
  program_id: string;
  program_name_th: string;
  chunk_type: string;
  plo_id: string;
  plo_number: number;
  domain: string;
  language: string;
  similarity: number;
}

export interface Citation {
  chunk_id: string;
  program_id: string;
  plo_id: string;
  chunk_type: string;
  similarity: number;
}

export interface RAGResponse {
  question: string;
  answer: string;
  citations: Citation[];
  retrieved: RetrievedChunk[];
  model_used: string;
  retrieval_ms: number;
  generation_ms: number;
}

export interface ProgramMatch {
  program_id: string;
  program_name_th: string;
  program_name_en: string;
  fit_score: number;
  vector_score: number;
  graph_score: number;
  matched_skills: string[];
  top_careers: { title_th: string; coverage: number; salary_entry: string }[];
  plo_highlights: string[];
  explanation: string;
}

export interface RecommendationResponse {
  interests: Record<string, number>;
  skill_vector: Record<string, number>;
  programs: ProgramMatch[];
  generation_ms: number;
}

export type Message =
  | { role: "user"; content: string }
  | { role: "assistant"; type: "rag"; data: RAGResponse }
  | { role: "assistant"; type: "recommendation"; data: RecommendationResponse }
  | { role: "assistant"; type: "loading" };
