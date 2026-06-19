"""Open-source LLM (Ollama) backed summarize / compare / digest builders.

Uses a local Ollama server (https://ollama.com) — free, no API key. Swap the
model with the ARDIVE_MODEL env var; point at a remote server with OLLAMA_HOST.
"""

from __future__ import annotations

import os
import re

import ollama

from .arxiv import Paper

MODEL = os.environ.get("ARDIVE_MODEL", "llama3.2")
# Context window. Papers are long; raise this (and your RAM) for big papers.
NUM_CTX = int(os.environ.get("ARDIVE_NUM_CTX", "8192"))
# How long Ollama keeps the model loaded after a call, so back-to-back runs
# skip the cold-start reload (Ollama's own default is only 5m).
KEEP_ALIVE = os.environ.get("ARDIVE_KEEP_ALIVE", "15m")
# Char budget for supplied paper text, leaving room for the prompt + response so
# the model sees the instruction and the start of the paper instead of truncating.
INPUT_CHARS = max(2000, (NUM_CTX - 1500) * 4)

SYSTEM = (
    "You are arDive, a research-paper explainer. You read arXiv papers and "
    "respond in clear Markdown bullet points. Be accurate and concise; do not "
    "invent results that are not in the provided text."
)


def _eli5_clause(eli5: bool) -> str:
    if not eli5:
        return ""
    return (
        " Explain like I'm 5: use plain language and everyday analogies, and "
        "avoid jargon and equations."
    )


def _bullets_clause(max_bullets: int | None) -> str:
    return f" Use at most {max_bullets} bullet points." if max_bullets else ""


# A top-level list item: "* ", "- ", "+ ", or "1. " at the start of a line.
_TOP_BULLET = re.compile(r"^(?:[*\-+]|\d+\.)\s")


def _cap_bullets(text: str, max_bullets: int | None) -> str:
    """Hard-trim to the first ``max_bullets`` top-level list items.

    The prompt also asks the model to self-limit, but local models don't obey
    counts reliably, so this guarantees the cap. Intro lines and nested/indented
    sub-bullets are kept; everything from the (max_bullets+1)-th item on is cut.
    """
    if not max_bullets:
        return text
    kept: list[str] = []
    count = 0
    for line in text.splitlines():
        if _TOP_BULLET.match(line):
            count += 1
            if count > max_bullets:
                break
        kept.append(line)
    return "\n".join(kept).rstrip()


def _ask(user: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            options={"num_ctx": NUM_CTX},
            keep_alive=KEEP_ALIVE,
        )
    except ConnectionError:
        raise RuntimeError(
            "could not reach Ollama. Install it from https://ollama.com, then "
            "run `ollama serve` and `ollama pull " + MODEL + "`."
        )
    except ollama.ResponseError as exc:
        raise RuntimeError(f"Ollama error: {exc}. Try `ollama pull {MODEL}`.")
    return response["message"]["content"].strip()


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... text truncated to fit the context window ...]"


def _paper_block(paper: Paper, body: str) -> str:
    return f"Title: {paper.title}\nAuthors: {paper.authors}\n\n{body}"


def _bare_id(paper_id: str) -> str:
    """arXiv id without the trailing version (e.g. '1706.03762v7' -> '1706.03762')."""
    return re.sub(r"v\d+$", "", paper_id)


def _clip_title(title: str, n: int = 40) -> str:
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def _parse_id_blocks(text: str) -> dict[str, list[str]]:
    """Split model output into blocks keyed by a leading ``[marker]`` line."""
    blocks: dict[str, list[str]] = {}
    key: str | None = None
    for line in text.splitlines():
        marker = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if marker:
            key = marker.group(1).strip()
            blocks[key] = []
        elif key is not None:
            blocks[key].append(line)
    return blocks


def _block_bullets(lines: list[str] | None) -> list[str]:
    """Keep only bullet lines from a block, normalized to '- '."""
    out = []
    for line in lines or []:
        s = line.strip()
        if s.startswith(("-", "*", "•")):
            out.append("- " + s.lstrip("-*• ").strip())
    return out


def summarize(
    paper: Paper,
    section: str | None,
    max_bullets: int | None,
    eli5: bool,
) -> str:
    focus = f" Focus only on the {section} section of the paper." if section else ""
    instruction = (
        "Summarize the following paper in bullet-point form."
        + focus
        + _bullets_clause(max_bullets)
        + _eli5_clause(eli5)
    )
    body = _clip(paper.full_text or paper.abstract, INPUT_CHARS)
    return _cap_bullets(_ask(f"{instruction}\n\n{_paper_block(paper, body)}"), max_bullets)


def compare(papers: list[Paper], eli5: bool) -> str:
    """Two sections — Similarities (bullets) and Differences (table) — under a
    code-generated 'Title A vs Title B' header."""
    per_paper = INPUT_CHARS // max(1, len(papers))
    blocks = "\n\n---\n\n".join(
        f"{p.title} ({_bare_id(p.id)}):\n{_clip(p.full_text or p.abstract, per_paper)}"
        for p in papers
    )
    columns = ", ".join(f'"{_clip_title(p.title)}"' for p in papers)
    instruction = (
        "Compare the following papers. Output EXACTLY these two sections, in this "
        "order, with no title and no other text:\n\n"
        "## Similarities\n"
        "- 3 to 6 concise bullet points covering what the papers share.\n\n"
        "## Differences\n"
        "A Markdown table whose first column header is 'Aspect' and which then has one "
        f"column per paper with these exact headers: {columns}. Each row is one point of "
        "difference, kept to a short phrase per cell." + _eli5_clause(eli5)
    )
    body = _ask(f"{instruction}\n\n{blocks}")
    header = " vs ".join(p.title for p in papers)
    return f"# {header}\n\n{body}"


def digest(query: str, papers: list[Paper], eli5: bool) -> str:
    """One concise entry per fetched paper (title + arXiv id + 2 bullets), then a
    short themes synthesis. The paper list/count is built in code so `-n` always holds."""
    listing = "\n\n".join(
        f"[{_bare_id(p.id)}] {p.title}\n{p.abstract}" for p in papers
    )
    instruction = (
        f"Below are {len(papers)} arXiv papers on '{query}', each with a bracketed id. "
        "Output ONLY blocks in this exact format and nothing else:\n"
        "[<id>]\n- one short sentence\n- one short sentence\n"
        "(one block per paper, reusing the exact bracketed id shown), then finally:\n"
        "[THEMES]\n- short cross-cutting theme or open question\n- ...\n"
        "Keep every bullet to a single concise sentence." + _eli5_clause(eli5)
    )
    parsed = _parse_id_blocks(_ask(f"{instruction}\n\n{listing}"))
    themes_key = next((k for k in parsed if k.upper() == "THEMES"), None)

    out = [f"# Digest: {query} ({len(papers)} papers)", ""]
    for p in papers:
        sid = _bare_id(p.id)
        out.append(f"### {p.title} ({sid})")
        out += _block_bullets(parsed.get(sid)) or ["- _(no summary returned)_"]
        out.append("")
    themes = _block_bullets(parsed.get(themes_key)) if themes_key else []
    if themes:
        out += ["## Themes", *themes]
    return "\n".join(out).rstrip()
