from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence

from generation.llm_client import LLMClient
from generation.prompt_builder import build_grounded_prompt
from retrieval.retriever import Retriever


REFUSAL_TEXT = "Not enough information in the uploaded documents/data."


@dataclass
class AnswerResult:
    answer: str
    citations: List[Dict]
    contexts: List[Dict]
    route: str


def detect_intent(question: str) -> str:
    q = question.lower()
    finance_keywords = [
        "volatility",
        "drawdown",
        "returns",
        "rolling",
        "trend",
        "stock",
        "price",
        "financial",
        "portfolio",
        "transaction",
    ]
    pdf_keywords = [
        "pdf",
        "document",
        "report",
        "page",
        "section",
        "what does",
        "summarize",
        "risk",
    ]

    has_finance = any(k in q for k in finance_keywords)
    has_pdf = any(k in q for k in pdf_keywords)

    if has_finance and has_pdf:
        return "both"
    if has_finance:
        return "finance"
    if has_pdf:
        return "pdf"
    return "both"


def _filter_by_score(contexts: List[Dict], min_score: float = 0.0) -> List[Dict]:
    return [c for c in contexts if c.get("score", 0.0) >= min_score]


def _build_extract_answer(contexts: List[Dict]) -> str:
    if not contexts:
        return REFUSAL_TEXT

    lines = []
    for ctx in contexts[:3]:
        lines.append(ctx["text"])
    if not lines:
        return REFUSAL_TEXT
    return "\n".join(lines)


class GroundedAnswerer:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient | None = None,
        enable_llm: bool = False,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client or LLMClient()
        self.enable_llm = enable_llm

    @staticmethod
    def _question_terms(question: str) -> set[str]:
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "who",
            "where",
            "when",
            "how",
            "does",
            "do",
            "did",
            "about",
            "from",
            "with",
            "and",
            "for",
            "this",
            "that",
            "these",
            "those",
            "pdf",
            "document",
            "data",
        }
        words = re.findall(r"[a-zA-Z0-9]+", question.lower())
        return {w for w in words if len(w) > 2 and w not in stop_words}

    def _extractive_answer(self, question: str, contexts: List[Dict]) -> tuple[str, List[Dict]]:
        if not contexts:
            return REFUSAL_TEXT, []

        question_lower = question.lower()
        if "abstract" in question_lower:
            for context in contexts:
                text = context.get("text", "")
                if not text:
                    continue
                lower = text.lower()
                idx = lower.find("abstract")
                if idx != -1:
                    snippet = text[idx:]
                    sentence_candidates = re.split(r"(?<=[.!?])\s+", snippet)
                    selected = [s.strip() for s in sentence_candidates if s.strip()]
                    if selected:
                        best = " ".join(selected[:3])
                        return best, [context]

        terms = self._question_terms(question)
        if not terms:
            terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))

        scored_sentences: List[tuple[float, str, Dict]] = []

        for context in contexts:
            text = context.get("text", "")
            if not text:
                continue
            sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentence_candidates:
                clean = sentence.strip()
                if len(clean) < 25:
                    continue
                lower = clean.lower()
                overlap = sum(1 for term in terms if term in lower)
                if overlap == 0:
                    continue
                score = overlap + max(float(context.get("score", 0.0)), 0.0)
                scored_sentences.append((score, clean, context))

        if not scored_sentences:
            fallback_contexts = contexts[:2]
            fallback_sentences: List[str] = []
            for context in fallback_contexts:
                text = context.get("text", "").strip()
                if not text:
                    continue
                sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
                picked = None
                for sentence in sentence_candidates:
                    clean = sentence.strip()
                    if len(clean) >= 25:
                        picked = clean
                        break
                if picked is None:
                    picked = text[:240]
                fallback_sentences.append(picked)

            if not fallback_sentences:
                return REFUSAL_TEXT, []

            return " ".join(fallback_sentences), fallback_contexts

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected_sentences: List[str] = []
        used_chunk_ids: set[str] = set()

        for _, sentence, context in scored_sentences:
            if sentence in selected_sentences:
                continue
            selected_sentences.append(sentence)
            used_chunk_ids.add(str(context["metadata"].get("chunk_id")))
            if len(selected_sentences) >= 3:
                break

        used_contexts = [
            c for c in contexts if str(c["metadata"].get("chunk_id")) in used_chunk_ids
        ]

        return " ".join(selected_sentences), used_contexts

    def answer(self, question: str, k: int = 5) -> AnswerResult:
        route = detect_intent(question)

        source_filter: Sequence[str] | None
        if route == "pdf":
            source_filter = ["pdf"]
        elif route == "finance":
            source_filter = None
        else:
            source_filter = None

        contexts = self.retriever.retrieve(question, k=k, source_types=source_filter)
        contexts = _filter_by_score(contexts, min_score=0.0)

        if not contexts:
            return AnswerResult(answer=REFUSAL_TEXT, citations=[], contexts=[], route=route)

        final_contexts = contexts
        if self.enable_llm and self.llm_client.is_available():
            prompt = build_grounded_prompt(question, contexts)
            response = self.llm_client.complete(prompt)
            if not response:
                response, final_contexts = self._extractive_answer(question, contexts)
        else:
            response, final_contexts = self._extractive_answer(question, contexts)

        if "Not enough information" in response and contexts:
            return AnswerResult(answer=REFUSAL_TEXT, citations=[], contexts=contexts, route=route)

        citations = [
            {
                "file": c["metadata"].get("file_name"),
                "page": c["metadata"].get("page_number"),
                "chunk_id": c["metadata"].get("chunk_id"),
            }
            for c in final_contexts
        ]

        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation.get("file"), citation.get("page"), citation.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            unique_citations.append(citation)

        return AnswerResult(answer=response, citations=unique_citations, contexts=final_contexts, route=route)
