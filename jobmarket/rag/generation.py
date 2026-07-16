"""Answer generation: Claude, grounded in retrieved evidence, with a deterministic mock fallback."""

import anthropic

from jobmarket.config import ANTHROPIC_MODEL

SYSTEM_PROMPT = (
    "You are a job-market intelligence assistant. Answer the user's question using ONLY the "
    "numbered evidence provided below — do not use outside knowledge. Cite postings inline using "
    "their bracket number, e.g. [2]. If the evidence is insufficient to fully answer, say so explicitly."
)


def format_evidence(evidence: list[dict]) -> str:
    blocks = []
    for i, e in enumerate(evidence, start=1):
        blocks.append(
            f"[{i}] {e['title']} — {e['company']} ({e['location']}) — job_id: {e['job_id']}\n"
            f"Excerpt: {e['excerpt'][:600]}"
        )
    return "\n\n".join(blocks)


def mock_answer(query: str, evidence: list[dict]) -> str:
    if not evidence:
        return "[Mock mode] No relevant postings were retrieved for this question."
    lines = [
        "[Mock mode — no Anthropic credentials available; showing retrieved evidence directly.]",
        f"Question: {query}",
        "",
        "Top matching postings:",
    ]
    for i, e in enumerate(evidence, start=1):
        snippet = e["excerpt"][:220].rsplit(" ", 1)[0] + "…"
        lines.append(f"[{i}] {e['title']} at {e['company']} ({e['location']}): {snippet}")
    return "\n".join(lines)


def generate_answer(
    query: str,
    evidence: list[dict],
    anthropic_client: anthropic.Anthropic,
    model: str = ANTHROPIC_MODEL,
) -> str:
    if not evidence:
        return mock_answer(query, evidence)

    user_prompt = f"Question: {query}\n\nEvidence:\n{format_evidence(evidence)}"
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
    except (anthropic.APIError, TypeError) as e:
        # anthropic.APIError covers auth/rate-limit/server/network failures once a request is
        # sent; the SDK raises a plain TypeError earlier, at request-build time, when no
        # credentials (API key, auth token, or profile) can be resolved at all — that's the
        # "no API key available" case this fallback exists for.
        print(f"[Anthropic API call failed ({type(e).__name__}); falling back to mock mode.]")
        return mock_answer(query, evidence)
