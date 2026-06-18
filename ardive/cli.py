"""arDive command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import sys
import threading
import time

SECTIONS = {
    "abstract": "abstract",
    "intro": "introduction",
    "methodology": "methodology",
    "related": "related works",
    "citations": "citations / references",
}


@contextlib.contextmanager
def _status(message: str):
    """Show an animated status on stderr while a blocking task runs.

    No-op when stderr isn't a TTY, so piped/redirected output stays clean.
    """
    if not sys.stderr.isatty():
        yield
        return

    done = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if done.is_set():
                break
            sys.stderr.write(f"\r{frame} {message}")
            sys.stderr.flush()
            time.sleep(0.1)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
        sys.stderr.write("\r\033[K")  # return to start of line and clear it
        sys.stderr.flush()


def _status_message(args: argparse.Namespace) -> str:
    model = os.environ.get("ARDIVE_MODEL", "llama3.2")
    if args.command == "sum":
        what = f"Summarizing {args.arxiv_id}"
    elif args.command == "comp":
        what = f"Comparing {len(args.arxiv_ids)} papers"
    elif args.command == "dig":
        what = f"Digesting '{args.topic}'"
    else:
        what = "Working"
    return f"{what}… (model: {model})"


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
        # Abstract-only requests skip the PDF download entirely (much faster).
        paper = arxiv.fetch_paper(args.arxiv_id, with_text=args.section != "abstract")
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
        with _status(_status_message(args)):
            result = _run(args)
        print(result)
    except Exception as exc:  # clean message, no traceback for expected failures
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
