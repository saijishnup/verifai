<p align="center">
  <img src="assets/logo.svg" alt="VerifAI" width="270" />
</p>

# VerifAI

*For the report you wrote yourself,*
*and the detector that still has to agree.*

[![MIT License](https://img.shields.io/github/license/saijishnup/verifai?style=flat-square)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-f6f5f3?style=flat-square&labelColor=1c1b19)](https://www.python.org/)
[![Runs locally](https://img.shields.io/badge/runs-100%25%20local-c2410c?style=flat-square&labelColor=1c1b19)](#privacy)
[![Two detectors](https://img.shields.io/badge/detectors-Desklib%20%2B%20GPTTrace-e7e5e1?style=flat-square&labelColor=1c1b19)](#how-it-scores)
[![No build step](https://img.shields.io/badge/build%20step-none-d9d6d0?style=flat-square&labelColor=1c1b19)](#run-it-locally)

---

## What it is

Checks a PDF report for passages an AI-text detector would flag, before you submit. Scores the whole report, each section and each passage, and marks them in the margin. Runs on your laptop; the PDF never leaves it.

## Features

- **Two detectors.** Desklib academic and GPTTrace. One switch flips every chart; a line says if they agree.
- **Margin heatmap.** Each passage gets a mark tinted by score. Flagged ones get a Copy button to rewrite and re-check.
- **Section bars.** Sections ranked by score. Click to jump.
- **Document map.** One cell per passage, start to end.
- **Word split.** Share of text scoring low, borderline, high.
- **Sentence rhythm.** Words per sentence per section. Flat rhythm often reads as AI. Plain stats, not the model.
- **Threshold slider.** 0.05 to 0.95, re-tints instantly.
- **Live results.** Passages fill in as scored.
- **Paste mode.** Check one paragraph, no PDF.
- **Cleanup.** Drops tables, figure labels, captions, page numbers, title block and references. "Show skipped" lists what went.

## How it scores

```
PDF ──► extract text (pypdf)
    ──► drop tables, figure labels, captions, references, title block
    ──► split by section, pack sentences into passages of ≤ 700 tokens
    ──► score each passage with both detectors
    ──► section and overall = token-weighted averages
```

| | Desklib academic | GPTTrace (optional) |
|---|---|---|
| Model | [desklib/ai-text-detector-academic-v1.01](https://huggingface.co/desklib/ai-text-detector-academic-v1.01) | [nakas/gpttrace-ettin150-text-v5e](https://huggingface.co/nakas/gpttrace-ettin150-text-v5e) |
| Base | DeBERTa-v3-large, 435M | Ettin encoder, 150M, int8 ONNX |
| Tuned for | Academic writing | General text, paraphrase attacks |
| [RAID](https://raid-bench.xyz) accuracy* | not submitted (general sibling: 91.2%) | 99.8% |
| Download | 1.74 GB | 155 MB |
| RAM | ~1.9 GB | ~0.3 GB |
| Licence | MIT | CC BY-SA 4.0 |

*AI text caught at 5% false positives, all attacks. Most top entries trained on RAID data; real-world accuracy is lower.

**Memory.** Desklib's 128k-word embedding table is stored in bf16, the rest in fp32: −0.3 GB, score change 0.0001. Full bf16 was 8× slower on CPU; int8 broke the model (everything ~0.99).

### Measured on a 17-page report

| | Desklib | GPTTrace |
|---|---|---|
| Whole report | 4.3% | 0.1% |
| AI-style test paragraph | 100% | 100% |
| Run time, both (i5-1245U) | 41–86 s | |

## Run it locally

Needs Python 3.12+ and ~4 GB disk.

```bash
git clone https://github.com/saijishnup/verifai.git
cd verifai
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download models (~1.9 GB, once):

```bash
hf download desklib/ai-text-detector-academic-v1.01 --local-dir models/academic-v1.01
hf download nakas/gpttrace-ettin150-text-v5e --local-dir models/gpttrace-150m
```

Second line optional: without it, Desklib only. Skip both and Desklib downloads on first run.

```bash
python app.py
```

Opens http://127.0.0.1:8000. Model loads in 15–30 s; upload unlocks when ready.

| Option | Does |
|---|---|
| `--fp32` | Full fp32 Desklib (+0.35 GB RAM) |
| `--port 8001` | Another port |
| `--no-browser` | No browser tab |
| `--selftest` | Test the text pipeline, no model |

```
verifai/
  app.py             server, cleanup, both detectors
  index.html         interface, plain HTML/CSS/JS
  requirements.txt   torch (CPU), transformers, sentencepiece, pypdf, onnxruntime
  models/            downloaded models, not in git
```

## What the score means

AI-likelihood from a detector, not proof of authorship.

- Low score: no AI style found. Edited or paraphrased AI text can also score low.
- Formal technical prose can score high even when human-written.
- Two detectors agreeing beats one. VerifAI flags disagreement.
- Studies show commercial detectors flag 9–15% of real human abstracts. Never use a score alone to accuse anyone.

## Privacy

Runs on 127.0.0.1 only. PDFs are read in memory, never saved. No accounts, cookies or analytics. Network use: one-time model downloads (Hugging Face) and fonts (Google Fonts).

## Delete everything

Stop with Ctrl+C, delete the folder. Frees ~3.5 GB.

## Credits

[Desklib](https://huggingface.co/desklib) · [GPTTrace by nakas](https://huggingface.co/nakas/gpttrace-ettin150-text-v5e) · [RAID benchmark](https://github.com/liamdugan/raid)

## License

[MIT](LICENSE) © 2026 Sai Jishnu P. Models keep their own licences (Desklib MIT, GPTTrace CC BY-SA 4.0) and download from Hugging Face; not redistributed here.

VerifAI is a second reader, not a judge.
