<p align="center">
  <img src="assets/logo.svg" alt="VerifAI logo: a page with a red margin mark" width="96" />
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

VerifAI checks an academic report for passages an AI-text detector would flag, before you hand it in. Drop in a PDF and it scores the whole report, each section and each passage, then marks the text in the margin the way an examiner marks a draft.

It runs on your own laptop. The PDF never leaves it.

## Features

- **Two detectors, one switch.** Desklib's academic detector and GPTTrace score every passage. Flip between them and every chart follows; a line under the score says whether they agree.
- **Margin heatmap.** Each passage gets a thin vertical mark tinted by its score. Flagged passages get a light wash and a Copy button, so you can rewrite and paste back to re-check.
- **Section bars.** Abstract, Introduction, Literature Review and so on, ranked by score. Click one to jump to it.
- **Document map.** One cell per passage, sized by length, from page 1 to the end.
- **Where the words fall.** One bar: how much of the text scores low, borderline or high.
- **Sentence rhythm.** Words per sentence for each section, as dots. Flat rhythm often reads as machine-written. Plain statistics, not used by either model.
- **Threshold slider.** Change the flag line from 0.05 to 0.95; everything re-tints instantly, no re-run.
- **Live results.** Passages fill in as they are scored, with a timer.
- **Paste mode.** Check a single paragraph without a PDF.
- **Honest cleanup.** Figure labels, table cells, captions, page numbers, the title block and the reference list are dropped before scoring. A "Show skipped" list lets you check what was removed.

## How it scores

```
PDF ──► extract text (pypdf)
    ──► drop tables, figure labels, captions, references, title block
    ──► split by section (1 Introduction, 2 …), pack sentences into passages of ≤ 700 tokens
    ──► score each passage with both detectors
    ──► section and overall scores = token-weighted averages
```

| | Desklib academic | GPTTrace (optional) |
|---|---|---|
| Model | [desklib/ai-text-detector-academic-v1.01](https://huggingface.co/desklib/ai-text-detector-academic-v1.01) | [nakas/gpttrace-ettin150-text-v5e](https://huggingface.co/nakas/gpttrace-ettin150-text-v5e) |
| Base | DeBERTa-v3-large, 435M params | Ettin encoder, 150M params, int8 ONNX |
| Tuned for | Academic writing | General text, robust to paraphrase attacks |
| [RAID](https://raid-bench.xyz) accuracy* | not submitted (general sibling: 91.2%) | 99.8% |
| Download | 1.74 GB | 155 MB |
| RAM while running | about 1.9 GB | about 0.3 GB |
| Licence | MIT | CC BY-SA 4.0 |

*RAID: share of AI text caught at a 5% false-positive rate, all attacks included. Most top entries trained on RAID's own training data, so real-world accuracy is lower.

**Memory trick.** Desklib's 128k-word embedding table is a lookup, not maths, so VerifAI stores it in bf16 and hands fp32 to the rest of the model. That saves about 0.3 GB with a measured score change of 0.0001. Full bf16 was 8× slower on a laptop CPU without bf16 units, and int8 quantisation broke the model (every passage scored about 0.99), so neither is used.

### Measured on a 17-page course report

| | Desklib academic | GPTTrace |
|---|---|---|
| Whole report | 4.3% | 0.1% |
| A deliberately AI-style paragraph | 100% | 100% |
| Run time, both models (Intel i5-1245U) | 41–86 s | |

## Run it locally

You need Python 3.12 or newer and about 4 GB of free disk space.

```bash
git clone https://github.com/saijishnup/verifai.git
cd verifai
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the models into `models/` (about 1.9 GB, once):

```bash
hf download desklib/ai-text-detector-academic-v1.01 --local-dir models/academic-v1.01
hf download nakas/gpttrace-ettin150-text-v5e --local-dir models/gpttrace-150m
```

The second line is optional. Without it, VerifAI runs with Desklib only. If you skip the first line too, Desklib downloads itself on the first run.

Start it:

```bash
python app.py
```

Your browser opens at http://127.0.0.1:8000. The model takes 15–30 seconds to load; the upload button unlocks when it is ready.

| Option | What it does |
|---|---|
| `--fp32` | Keeps Desklib fully in fp32 (exact model-card precision, about 0.35 GB more RAM) |
| `--port 8001` | Runs on another port |
| `--no-browser` | Does not open a browser tab |
| `--selftest` | Checks the text pipeline without loading any model |

### Files

```
verifai/
  app.py             server, PDF cleanup, both detectors (Python standard library server)
  index.html         the whole interface, plain HTML/CSS/JS
  requirements.txt   torch (CPU), transformers, sentencepiece, pypdf, onnxruntime
  models/            downloaded models, not in git
```

## What the score means

The score is **AI-likelihood from a detector**, not proof of who wrote the text.

- A low score means the detector found nothing it recognises as AI-style. It does not prove no AI was used: lightly edited or paraphrased AI text often scores low too.
- Formal, uniform technical prose can score high even when a person wrote it.
- Two detectors agreeing is stronger evidence than one. When they disagree, VerifAI says so.
- Independent studies show commercial detectors flagging 9–15% of real human-written abstracts. No detector score should be used on its own to accuse anyone.

## Privacy

Everything runs on your machine. The server listens on 127.0.0.1 only, so other devices on your network cannot reach it. PDFs are read in memory and never saved. There are no accounts, cookies or analytics. The only network requests are the one-time model downloads from Hugging Face and the interface fonts from Google Fonts.

## Delete everything

The models, packages and code all live in the project folder. Close VerifAI with Ctrl+C, then delete the folder to free about 3.5 GB.

## Credits

- [Desklib](https://huggingface.co/desklib) for the academic detector
- [GPTTrace by nakas](https://huggingface.co/nakas/gpttrace-ettin150-text-v5e) for the second detector and its text normaliser
- [RAID benchmark](https://github.com/liamdugan/raid) for public, comparable detector scores

## License

[MIT](LICENSE) © 2026 Sai Jishnu P. The models keep their own licences (Desklib: MIT, GPTTrace: CC BY-SA 4.0); they are downloaded from Hugging Face, not redistributed here.

VerifAI is a second reader, not a judge.
