from __future__ import annotations

from typing import Dict, List


def format_context_blocks(contexts: List[Dict]) -> str:
    if not contexts:
        return ""

    lines: List[str] = []
    for idx, item in enumerate(contexts, start=1):
        md = item["metadata"]
        lines.append(
            f"[{idx}] source_type={md.get('source_type')} | "
            f"file={md.get('file_name')} | page={md.get('page_number')} | "
            f"chunk_id={md.get('chunk_id')}"
        )
        lines.append(item["text"])
        lines.append("")
    return "\n".join(lines)


def build_grounded_prompt(question: str, contexts: List[Dict]) -> str:
    context_blob = format_context_blocks(contexts)
    return (
        "You are a grounded assistant. Use only the context below. "
        "If the answer is not present in the context, respond exactly with: "
        "Not enough information in the uploaded documents/data. "
        "Do not use external knowledge. Include citations using this format: "
        "(file=<name>, page=<number_or_None>, chunk_id=<id>).\n\n"
        f"Context:\n{context_blob}\n"
        f"Question: {question}\n"
        "Answer:"
    )
