# arDive: a simple dive into your ArXiv

A small command-line agent that pulls papers from arXiv and uses Llama3.2 to
summarize, explain, compare, and digest them. Anyone can easily install and use without the need of a paid plan.

## Install

```bash
# 1. Install Ollama (free, runs models locally): https://ollama.com
ollama pull llama3.2:1b     # the fast default; or any open model you like

# 2. Install arDive
pip install ardive
```

That's it. arDive talks to the local Ollama server. Pick a
different model with `ARDIVE_MODEL` (e.g. `export ARDIVE_MODEL=qwen2.5`), or
point at a remote Ollama with `OLLAMA_HOST`.

### From source

```bash
git clone https://github.com/rohankosalge/arDive
cd arDive
pip install -e .
```

## Usage

```bash
# Summarize a paper as bullet points
ardive sum 1234.56789

# Focus on one section, cap the bullets
ardive sum 1234.56789 --section methodology --max-bullets 5

# Explain like I'm 5 (works on every command)
ardive sum 1234.56789 --eli5

# Compare papers (similarities + a differences table)
ardive comp 1234.56789 9876.54321

# Digest a topic (searches arXiv, default 3 papers)
ardive dig "diffusion models for protein folding"
ardive dig "graph neural networks" -n 5

# List papers similar to a given one (fast; no model)
ardive sim 1234.56789 -n 5
```

### Commands

| Command | What it does |
| --- | --- |
| `sum <id>` | Bullet-point summary of one paper (full PDF text). |
| `comp <id> <id> [...]` | Compare papers: a `Title A vs Title B` header, a **Similarities** bullet list, and a **Differences** table. |
| `dig <topic>` | Search arXiv by topic; one concise entry per paper (title + arXiv id + bullets) plus a **Themes** synthesis. |
| `sim <id>` | List papers similar to the given one (title + arXiv id only). Pure arXiv lookup — no model, very fast. |

### Flags

- `--eli5` — explain in plain, jargon-free language (`sum`, `comp`, `dig`).
- `--section {abstract,intro,methodology,related,citations}` — `sum` only; focus on one section.
- `--max-bullets N` — `sum` only; hard cap on the number of bullets (positive integer).
- `-n/--num N` — `dig` and `sim`; how many papers to pull/list (default 3).

## How it works

`sum` and `comp` download each paper's PDF and extract its full text; `dig`
searches arXiv and works from abstracts. The text is sent to a local
open-source model via Ollama (default `llama3.2:1b`) with a prompt tailored to
the command. (`sim` uses no model — it's a pure arXiv lookup.) In a terminal the
response is rendered as formatted Markdown inside a box; when piped or redirected
(e.g. `ardive sum 1234.56789 > out.md`) it's written as plain Markdown so the
file stays clean.

Long papers can exceed the model's context window and be truncated. arDive asks
Ollama for an 8192-token window by default; raise it (at the cost of more RAM)
with `export ARDIVE_NUM_CTX=16384`.

## Speed

Summaries run entirely on your machine, so wall-clock time is dominated by the
model. A few levers:

- **Model choice is the biggest one.** The default `llama3.2:1b` is fast but
  modest in quality; for better summaries (slower) try `export
  ARDIVE_MODEL=llama3.2` (~3B) or `export ARDIVE_MODEL=qwen2.5:3b`. 7B+ models
  are noticeably slower on full papers.
- **Abstract is near-instant.** `ardive sum <id> --section abstract` skips the
  PDF download and summarizes just the abstract.
- **First run is slowest.** It loads the model into memory; arDive keeps it warm
  for 15 min afterward (tune with `ARDIVE_KEEP_ALIVE`), so repeat runs are quicker.
- **Smaller asks finish sooner.** `--max-bullets N` shortens the output, and a
  lower `ARDIVE_NUM_CTX` trades context for speed.
