"""Fetch paper metadata and text from arXiv."""

from __future__ import annotations

import tempfile
import urllib.request
from dataclasses import dataclass

import arxiv

_client = arxiv.Client()


@dataclass
class Paper:
    id: str
    title: str
    authors: str
    abstract: str
    full_text: str = ""


def _result_to_paper(result: arxiv.Result, full_text: str = "") -> Paper:
    return Paper(
        id=result.get_short_id(),
        title=result.title.strip(),
        authors=", ".join(a.name for a in result.authors),
        abstract=result.summary.strip(),
        full_text=full_text,
    )


def _pdf_text(url: str) -> str:
    from pypdf import PdfReader

    req = urllib.request.Request(url, headers={"User-Agent": "arDive"})
    with urllib.request.urlopen(req) as resp, tempfile.NamedTemporaryFile(
        suffix=".pdf"
    ) as tmp:
        tmp.write(resp.read())
        tmp.flush()
        reader = PdfReader(tmp.name)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def fetch_paper(arxiv_id: str) -> Paper:
    """Fetch one paper by arXiv ID, including full text from its PDF."""
    try:
        result = next(_client.results(arxiv.Search(id_list=[arxiv_id])))
    except StopIteration:
        raise LookupError(f"No arXiv paper found with id '{arxiv_id}'.")

    text = _pdf_text(result.pdf_url)

    # Fall back to the abstract if text extraction came up empty.
    return _result_to_paper(result, full_text=text or result.summary.strip())


def search_topic(query: str, n: int) -> list[Paper]:
    """Search arXiv by topic; returns metadata + abstracts (no PDF download)."""
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = [_result_to_paper(r) for r in _client.results(search)]
    if not papers:
        raise LookupError(f"No arXiv papers found for topic '{query}'.")
    return papers
