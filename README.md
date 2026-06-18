# arDive: a simple dive into your ArXiv

A small command-line agent that pulls papers from arXiv and uses Llama3.2 to
summarize, explain, compare, and digest them. Anyone can easily install and use without the need of a paid plan.

## Install

```bash
# 1. Install Ollama (free, runs models locally): https://ollama.com
ollama pull llama3.2        # or any open model you like

# 2. Install arDive
pip install -e .
```

That's it. arDive talks to the local Ollama server. Pick a
different model with `ARDIVE_MODEL` (e.g. `export ARDIVE_MODEL=qwen2.5`), or
point at a remote Ollama with `OLLAMA_HOST`.

## Usage

```bash
# Summarize a paper as bullet points
ardive sum 1234.56789

# Focus on one section, cap the bullets
ardive sum 1234.56789 --section methodology --max-bullets 5

# Explain like I'm 5 (works on every command)
ardive sum 1234.56789 --eli5

# Compare two or more papers
ardive comp 1234.56789 9876.54321

# Digest a topic (searches arXiv, default 8 papers)
ardive dig "diffusion models for protein folding"
ardive dig "graph neural networks" -n 12
```

### Commands

| Command | What it does |
| --- | --- |
| `sum <id>` | Bullet-point summary of one paper (full PDF text). |
| `comp <id> <id> [...]` | Compare two or more papers. |
| `dig <topic>` | Search arXiv by topic and digest the top results. |

### Flags

- `--eli5` — explain in plain, jargon-free language (all commands).
- `--section {abstract,intro,methodology,related,citations}` — `sum` only; focus on one section.
- `--max-bullets N` — `sum` only; cap the number of bullets (positive integer).
- `-n/--num N` — `dig` only; how many papers to pull (default 8).

## How it works

`sum` and `comp` download each paper's PDF and extract its full text; `dig`
searches arXiv and works from abstracts. The text is sent to a local
open-source model via Ollama (default `llama3.2`) with a prompt tailored to the
command, and the bullet-point response is printed to stdout.

Long papers can exceed the model's context window and be truncated. arDive asks
Ollama for an 8192-token window by default; raise it (at the cost of more RAM)
with `export ARDIVE_NUM_CTX=16384`.
