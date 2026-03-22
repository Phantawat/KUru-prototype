from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from rag_pipeline import RAGPipeline
from recommendation_pipeline import RecommendationPipeline

app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000"],
	allow_methods=["*"],
	allow_headers=["*"],
)

rag = RAGPipeline()
rec = RecommendationPipeline()


class AskRequest(BaseModel):
	question: str
	program_filter: str | None = None


class RecommendRequest(BaseModel):
	interests: dict[str, float]


@app.post("/ask")
def ask(req: AskRequest):
	resp = rag.ask(req.question, program_filter=req.program_filter)
	return {
		"question": resp.question,
		"answer": resp.answer,
		"citations": resp.citations,
		"retrieved": [
			{
				"chunk_id": c.chunk_id,
				"text": c.text,
				"program_id": c.program_id,
				"program_name_th": c.program_name_th,
				"chunk_type": c.chunk_type,
				"plo_id": c.plo_id,
				"plo_number": c.plo_number,
				"domain": c.domain,
				"language": c.language,
				"similarity": c.similarity,
			}
			for c in resp.retrieved
		],
		"model_used": resp.model_used,
		"retrieval_ms": resp.retrieval_ms,
		"generation_ms": resp.generation_ms,
	}


@app.post("/recommend")
def recommend_programs(req: RecommendRequest):
	resp = rec.recommend(req.interests)
	return {
		"interests": resp.interests,
		"skill_vector": resp.skill_vector,
		"programs": [
			{
				"program_id": p.program_id,
				"program_name_th": p.program_name_th,
				"program_name_en": p.program_name_en,
				"fit_score": p.fit_score,
				"vector_score": p.vector_score,
				"graph_score": p.graph_score,
				"matched_skills": p.matched_skills,
				"top_careers": p.top_careers,
				"plo_highlights": p.plo_highlights,
				"explanation": p.explanation,
			}
			for p in resp.programs
		],
		"generation_ms": resp.generation_ms,
	}
