"""Fetch paper metadata and text from arXiv."""

from __future__ import annotations

import contextlib
import re
import socket
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass

import arxiv

# Few retries + small pages so a busy/rate-limiting arXiv fails fast instead of
# hanging on the library's default long backoff.
_client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=1)


@contextlib.contextmanager
def _bounded():
    """Cap per-request network time and turn arXiv outages into a clean error."""
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(20)
    try:
        yield
    except arxiv.ArxivError as exc:
        raise RuntimeError(
            "arXiv is busy or rate-limiting right now — please try again in a moment."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach arXiv: {exc.reason}") from exc
    finally:
        socket.setdefaulttimeout(old)


@dataclass
class Paper:
    id: str
    title: str
    authors: str
    abstract: str
    primary_category: str = ""
    full_text: str = ""


def _result_to_paper(result: arxiv.Result, full_text: str = "") -> Paper:
    return Paper(
        id=result.get_short_id(),
        title=result.title.strip(),
        authors=", ".join(a.name for a in result.authors),
        abstract=result.summary.strip(),
        primary_category=result.primary_category or "",
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


def fetch_paper(arxiv_id: str, with_text: bool = True) -> Paper:
    """Fetch one paper by arXiv ID.

    With ``with_text`` (default), download the PDF and extract its full text.
    Pass ``with_text=False`` to skip the PDF entirely (metadata + abstract only)
    — much faster when the abstract is all that's needed.
    """
    try:
        with _bounded():
            result = next(_client.results(arxiv.Search(id_list=[arxiv_id])))
    except StopIteration:
        raise LookupError(f"No arXiv paper found with id '{arxiv_id}'.")

    if not with_text:
        return _result_to_paper(result)

    # Fall back to the abstract if text extraction came up empty.
    text = _pdf_text(result.pdf_url)
    return _result_to_paper(result, full_text=text or result.summary.strip())


def search_topic(query: str, n: int) -> list[Paper]:
    """Search arXiv by topic; returns metadata + abstracts (no PDF download)."""
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    with _bounded():
        papers = [_result_to_paper(r) for r in _client.results(search)]
    if not papers:
        raise LookupError(f"No arXiv papers found for topic '{query}'.")
    return papers


def bare_id(paper_id: str) -> str:
    """arXiv id without the trailing version (e.g. '1706.03762v7' -> '1706.03762')."""
    return re.sub(r"v\d+$", "", paper_id)


def find_similar(arxiv_id: str, n: int) -> tuple[Paper, list[Paper]]:
    """Return the source paper plus up to ``n`` similar papers (no PDF, no model).

    Similarity is an arXiv relevance search on the source paper's title, scoped to
    its primary category to avoid cross-domain matches, with the source removed.
    Falls back to a title-only search if the category-scoped one finds nothing.
    """
    src = fetch_paper(arxiv_id, with_text=False)
    # Strip punctuation so the title can't break arXiv's query syntax (e.g. ':').
    terms = re.sub(r"[^\w\s]", " ", src.title).strip()

    def _search(query: str) -> list[Paper]:
        results = search_topic(query, n + 1)
        return [p for p in results if bare_id(p.id) != bare_id(src.id)][:n]

    if src.primary_category:
        try:
            similar = _search(f"cat:{src.primary_category} AND {terms}")
        except LookupError:
            similar = []
        if similar:
            return src, similar
    return src, _search(terms)
