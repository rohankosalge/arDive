"""arDive command-line interface."""

from __future__ import annotations

import argparse
import sys

SECTIONS = {
    "abstract": "abstract",
    "intro": "introduction",
    "methodology": "methodology",
    "related": "related works",
    "citations": "citations / references",
}


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--eli5", action="store_true", help="explain like I'm 5"
    )

    parser = argparse.ArgumentParser(
        prog="ardive",
        description="Pull papers from arXiv and summarize / explain them with a local open-source model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser(
        "sum", parents=[common], help="summarize a paper in bullet points"
    )
    p_sum.add_argument("arxiv_id", help="arXiv id, e.g. 2410.12345")
    p_sum.add_argument(
        "--section", choices=list(SECTIONS), help="focus on one section"
    )
    p_sum.add_argument(
        "--max-bullets", type=positive_int, help="cap the number of bullets"
    )

    p_comp = sub.add_parser(
        "comp", parents=[common], help="compare two or more papers"
    )
    p_comp.add_argument("arxiv_ids", nargs="+", help="two or more arXiv ids")

    p_dig = sub.add_parser(
        "dig", parents=[common], help="digest a topic of papers"
    )
    p_dig.add_argument("topic", help="topic / search query")
    p_dig.add_argument(
        "-n", "--num", type=positive_int, default=8, help="papers to pull (default 8)"
    )

    return parser


def _run(args: argparse.Namespace) -> str:
    # Imported lazily so `ardive --help` works without the model deps running.
    from . import arxiv, llm

    if args.command == "sum":
        paper = arxiv.fetch_paper(args.arxiv_id)
        section = SECTIONS[args.section] if args.section else None
        return llm.summarize(paper, section, args.max_bullets, args.eli5)

    if args.command == "comp":
        if len(args.arxiv_ids) < 2:
            raise ValueError("comp needs at least two arXiv ids")
        papers = [arxiv.fetch_paper(i) for i in args.arxiv_ids]
        return llm.compare(papers, args.eli5)

    if args.command == "dig":
        papers = arxiv.search_topic(args.topic, args.num)
        return llm.digest(args.topic, papers, args.eli5)

    raise ValueError(f"unknown command: {args.command}")


def main() -> None:
    args = _build_parser().parse_args()
    try:
        print(_run(args))
    except Exception as exc:  # clean message, no traceback for expected failures
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
