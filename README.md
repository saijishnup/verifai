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

VerifAI checks PDF reports for passages that AI-text detectors may flag. It scores the document, sections, and passages, then marks them in the margin.

Everything runs locally; PDFs never leave your computer.

## Features

- **Two detectors:** Desklib Academic and GPTTrace.
- **Margin heatmap:** Score each passage and rewrite flagged text for re-checking.
- **Section bars:** Rank sections and jump between them.
- **Document map:** View every passage from start to finish.
- **Word split:** Compare low, borderline, and high-scoring text.
- **Sentence rhythm:** Inspect words per sentence by section.
- **Threshold slider:** Adjust from 0.05 to 0.95.
- **Live results:** Scores appear as passages finish.
- **Paste mode:** Check a paragraph without uploading a PDF.
- **Cleanup:** Removes tables, figure labels, captions, page numbers, title blocks, and references. Skipped content can be viewed.

## How it scores

```
PDF ──► extract text (pypdf)
    ──► remove tables, labels, captions, references, and title block
    ──► split by section into passages of ≤ 700 tokens
    ──► score each passage with both detectors
    ──► calculate token-weighted section and overall averages
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

*AI text caught at 5% false positives across all attacks. Real-world accuracy is lower.

**Memory:** Desklib's embedding table uses bf16; the rest uses fp32. This saves 0.3 GB with a score change of 0.0001.

### Measured on a 17-page report

| | Desklib | GPTTrace |
|---|---|---|
| Whole report | 4.3% | 0.1% |
| AI-style test paragraph | 100% | 100% |
| Run time, both (i5-1245U) | 41–86 s | |

## Run it locally

Requires Python 3.12+ and approximately 4 GB of disk space.

```bash
git clone https://github.com/saijishnup/verifai.git
cd verifai
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the models once:

```bash
hf download desklib/ai-text-detector-academic-v1.01 --local-dir models/academic-v1.01
hf download nakas/gpttrace-ettin150-text-v5e --local-dir models/gpttrace-150m
```

GPTTrace is optional. Without it, VerifAI uses Desklib only. If both models are skipped, Desklib downloads on first run.

```bash
python app.py
```

Open http://127.0.0.1:8000. The model loads in 15–30 seconds.

| Option | Does |
|---|---|
| `--fp32` | Full fp32 Desklib (+0.35 GB RAM) |
| `--port 8001` | Use another port |
| `--no-browser` | Do not open a browser tab |
| `--selftest` | Test the text pipeline without loading a model |

```
verifai/
  app.py             server, cleanup, both detectors
  index.html         interface, plain HTML/CSS/JS
  requirements.txt   torch (CPU), transformers, sentencepiece, pypdf, onnxruntime
  models/            downloaded models, not in git
```

## What the score means

The score is an AI-likelihood estimate, not proof of authorship.

- Low scores do not rule out edited or paraphrased AI text.
- Formal technical writing can score high when written by humans.
- Agreement between both detectors is more useful than one score.
- Commercial detectors have flagged 9–15% of real human abstracts. Never rely on a score alone.

## Privacy

VerifAI runs on `127.0.0.1`. PDFs are read in memory and never saved. There are no accounts, cookies, or analytics.

Network access is limited to model downloads from Hugging Face and fonts from Google Fonts.
  
## Credits

[Desklib](https://huggingface.co/desklib) · [GPTTrace by nakas](https://huggingface.co/nakas/gpttrace-ettin150-text-v5e) · [RAID benchmark](https://github.com/liamdugan/raid)

## License

[MIT](LICENSE) © 2026 Sai Jishnu P. Models retain their own licences: Desklib MIT and GPTTrace CC BY-SA 4.0.

VerifAI is a second reader, not a judge.
