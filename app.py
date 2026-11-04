"""VerifAI: local AI-text check for PDFs, using desklib/ai-text-detector-academic-v1.01,
plus nakas/gpttrace-ettin150-text-v5e as a second opinion when models/gpttrace-150m/ exists.

Run:  python app.py            (~2.0 GB RAM, scores within 0.0001 of the model card)
      python app.py --fp32     (exact model-card precision, ~2.35 GB RAM)
      python app.py --selftest (checks the text pipeline, no model needed)
"""
import json
import os
import re
import statistics
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).parent
# Keep the 1.7 GB model inside this folder, so deleting the folder removes everything.
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

LOCAL_MODEL = ROOT / "models" / "academic-v1.01"   # manual download goes here
MODEL_ID = str(LOCAL_MODEL) if (LOCAL_MODEL / "model.safetensors").exists() else "desklib/ai-text-detector-academic-v1.01"
MAX_TOKENS = 700   # passage size; model limit is 768 incl. special tokens
MIN_TAIL = 200     # shorter last passage merges into the previous one
PORT = 8000
MAX_PAGES = 50

# ---------------------------------------------------------------- text pipeline

HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z][A-Za-z][^.!?]{2,80})$")
CAPTION = re.compile(r"^(Figure|Fig\.|Table|Algorithm)\s*\d+[.:]", re.I)
END = re.compile(r"[.!?:;)\]]$")
STOPWORDS = set("the a an of and or to in on at by for with from is are was were be as that this it its we our than but not which".split())
RUN_IN = re.compile(r"(?<=[a-z]{3})\.(?=[A-Z0-9])")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\s*[•▪◦]\s*")


def mostly_symbols(line):
    """Table cells, axis labels, page numbers."""
    chars = line.replace(" ", "")
    return not chars or sum(c.isalpha() for c in chars) / len(chars) < 0.6


def no_function_words(line):
    """Table rows ("CoDel 2012 Sojourn time No No") have no function words; prose lines do."""
    return not STOPWORDS & set(re.findall(r"[a-z]+", line.lower()))


def clean(raw):
    """Raw extracted text -> (sections [(name, text)], skipped lines)."""
    lines = [l.strip() for l in raw.splitlines()]
    skipped = []

    low = [l.lower() for l in lines]
    if "abstract" in low:                      # drop title and author block
        start = low.index("abstract")
        skipped += [l for l in lines[:start] if l]
        lines = lines[start:]
    for i, l in enumerate(lines):              # drop references to the end
        if l.lower() in ("references", "bibliography", "works cited"):
            skipped += [x for x in lines[i:] if x]
            lines = lines[:i]
            break

    sections, name, buf = [], "Text", []
    prev_kept, in_caption, top = False, False, 0

    def flush():
        if buf:
            sections.append((name, " ".join(buf)))

    for l in lines:
        if not l:
            continue
        l = RUN_IN.sub(". ", l)                # "Background.5G" -> "Background. 5G"
        if l.lower() == "abstract":
            flush(); name, buf, prev_kept = "Abstract", [], False
            continue
        m = HEADING.match(l)
        if m and len(l.split()) <= 9:
            nums = [int(x) for x in m.group(1).split(".")]
            # Only headings in sequence count, so "10 Mbps" figure labels or table rows don't.
            if len(nums) == 1 and nums[0] == top + 1:
                flush(); top = nums[0]; name, buf, prev_kept = f"{top} {m.group(2)}", [], False
                continue
            if len(nums) > 1 and nums[0] == top:
                prev_kept = False              # subsection heading: not prose
                continue
        if CAPTION.match(l):
            in_caption = not END.search(l)
            skipped.append(l); prev_kept = False
            continue
        if in_caption:
            in_caption = not END.search(l)
            skipped.append(l)
            continue
        words = len(l.split())
        continuation = prev_kept and buf and not END.search(buf[-1]) and END.search(l)
        if mostly_symbols(l) or (not continuation and (words < 6 or no_function_words(l))):
            skipped.append(l); prev_kept = False
            continue
        if buf and buf[-1].endswith("-") and l[:1].islower():
            buf[-1] = buf[-1][:-1] + l         # re-join hyphenated word
        else:
            buf.append(l)
        prev_kept = True
    flush()

    merged = {}                                # same section split by noise -> one entry
    for n, t in sections:
        merged[n] = (merged[n] + " " + t) if n in merged else t
    # ponytail: a 90+ word "sentence" is almost always a table that slipped through;
    # smarter table detection only if real long sentences start getting dropped.
    out = []
    for n, t in merged.items():
        keep = []
        for s in sentences(t):
            (skipped if len(s.split()) > 90 else keep).append(s)
        if keep:
            out.append((n, " ".join(keep)))
    return out, skipped


def sentences(text):
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def pack(sections, count_tokens):
    """Pack sentences into passages of <= MAX_TOKENS, never crossing a section."""
    passages = []
    for name, text in sections:
        cur, cur_tok, sec = [], 0, []
        for s in sentences(text):
            t = count_tokens(s)
            if cur and cur_tok + t > MAX_TOKENS:
                sec.append((cur, cur_tok)); cur, cur_tok = [], 0
            cur.append(s); cur_tok += t
        if cur:
            sec.append((cur, cur_tok))
        if len(sec) > 1 and sec[-1][1] < MIN_TAIL and sec[-2][1] + sec[-1][1] <= 760:
            tail = sec.pop()
            sec[-1] = (sec[-1][0] + tail[0], sec[-1][1] + tail[1])
        passages += [{"section": name, "text": " ".join(c), "tokens": n} for c, n in sec]
    return passages


def rhythm(sections):
    """Words per sentence per section. Plain stats, not used by the model."""
    out = []
    for name, text in sections:
        lens = [len(s.split()) for s in sentences(text)]
        if len(lens) >= 3:
            out.append({"section": name, "lengths": lens, "sigma": round(statistics.pstdev(lens), 1)})
    return out


def pdf_text(data):
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"PDF has {len(reader.pages)} pages. Limit is {MAX_PAGES}.")
    return "\n".join(p.extract_text() or "" for p in reader.pages), len(reader.pages)

# ---------------------------------------------------------------- model

STATE = {"status": "loading", "msg": "Loading model", "dtype": None, "mem_gb": None}
MODEL = TOKENIZER = None
LOCK = threading.Lock()   # one run at a time keeps RAM flat


def load(fp32=False):
    global MODEL, TOKENIZER
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):   # from the model card
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask):
            hidden = self.model(input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.classifier(pooled)

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    MODEL = DesklibAIDetectionModel.from_pretrained(MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    if not fp32:
        # The 128k-word embedding table is a lookup, not math: store it in bf16 (-0.3 GB RAM)
        # and hand fp32 to the rest. Measured drift on a 17-page report: 0.0001.
        # Full bf16 or int8 were rejected: bf16 is ~8x slower on CPUs without bf16 units,
        # int8 dynamic quantization breaks this model (scores jump to ~0.99).
        emb = MODEL.model.embeddings.word_embeddings
        emb.weight.data = emb.weight.data.to(torch.bfloat16)
        emb.register_forward_hook(lambda m, i, out: out.float())
    mem = sum(p.numel() * p.element_size() for p in MODEL.parameters()) / 1e9
    if (GT_DIR / "onnx" / "model_quantized.onnx").exists():
        load_gpttrace()
        mem += 0.15
    STATE.update(status="ready", msg="Ready", dtype="fp32" if fp32 else "fp32, bf16 embeddings",
                 mem_gb=round(mem, 2), second=GT is not None)


# ---------------------------------------------------------------- second opinion: GPTTrace
# nakas/gpttrace-ettin150-text-v5e, int8 ONNX, 512-token limit. Optional: used only if
# models/gpttrace-150m/ exists. Scored with its own tokenizer and text normalizer.

GT_DIR = ROOT / "models" / "gpttrace-150m"
GT = None   # (session, tokenizer, normalize)


def load_gpttrace():
    global GT
    import importlib.util
    import onnxruntime as ort
    from tokenizers import Tokenizer
    spec = importlib.util.spec_from_file_location("gpttrace_normalize", GT_DIR / "normalize.py")
    norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm)
    sess = ort.InferenceSession(str(GT_DIR / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"])
    GT = (sess, Tokenizer.from_file(str(GT_DIR / "tokenizer.json")), norm.normalize_text)


def gt_window(text):
    import numpy as np
    sess, tok, norm = GT
    ids = tok.encode(norm(text)).ids
    if len(ids) > 512:
        ids = ids[:511] + [ids[-1]]            # keep closing [SEP], as in training
    ids = np.array([ids], dtype=np.int64)
    human, ai = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
    return float(1 / (1 + np.exp(human - ai)))


def gt_score(text):
    """Passages can exceed 512 tokens: score sentence windows of <= 480 and token-weight them."""
    tok = GT[1]
    wins, cur, n = [], [], 0
    for s in sentences(text):
        k = len(tok.encode(s, add_special_tokens=False).ids)
        if cur and n + k > 480:
            wins.append((" ".join(cur), n)); cur, n = [], 0
        cur.append(s); n += k
    if cur:
        wins.append((" ".join(cur), n))
    total = sum(n for _, n in wins) or 1
    return round(sum(gt_window(t) * n for t, n in wins) / total, 4)


def count_tokens(text):
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def score(text):
    import torch
    enc = TOKENIZER(text, truncation=True, max_length=768, return_tensors="pt")
    with torch.inference_mode():
        logit = MODEL(enc["input_ids"], enc["attention_mask"])
    return round(torch.sigmoid(logit.float()).item(), 4)

# ---------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/status":
            self.send(200, STATE)
        else:
            self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/analyze":
            return self.send(404, {"error": "Not found"})
        if STATE["status"] != "ready":
            return self.send(503, {"error": "Model is still loading"})
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= 50_000_000:
            return self.send(400, {"error": "Send a PDF up to 50 MB or some text"})
        body = self.rfile.read(size)
        try:
            if self.headers.get("Content-Type", "").startswith("application/pdf"):
                raw, pages = pdf_text(body)
                name = self.headers.get("X-Filename", "document.pdf")
            else:
                raw, pages, name = json.loads(body)["text"], None, "Pasted text"
        except Exception as e:
            return self.send(400, {"error": str(e) if isinstance(e, ValueError) else "Could not read this file"})
        sections, skipped = clean(raw)
        if not sections:
            return self.send(422, {"error": "No text found. If this PDF is a scan, paste the text instead."})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()                     # HTTP/1.0: stream lines, then close

        def line(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()

        with LOCK:
            t0 = time.time()
            passages = pack(sections, count_tokens)
            line({"type": "meta", "name": name, "pages": pages, "raw_words": len(raw.split()),
                  "passages": len(passages), "sections": [n for n, _ in sections],
                  "rhythm": rhythm(sections), "skipped": skipped,
                  "dtype": STATE["dtype"], "mem_gb": STATE["mem_gb"], "second": GT is not None})
            for i, p in enumerate(passages):
                p.update(type="passage", i=i, score=score(p["text"]), words=len(p["text"].split()),
                         score2=gt_score(p["text"]) if GT else None)
                line(p)
            line({"type": "done", "seconds": round(time.time() - t0, 1)})


def selftest():
    raw = "\n".join([
        "Some Title", "Team Members", "Abstract",
        "This study looks at how queues behave when many flows share one slow link today.",
        "We measured delay and throughput over three seeds and report the",
        "mean values.",
        "1", "254.4 153.2 67.2 80", "Figure 2: Published results of the six queue",
        "management configurations.",
        "1 Introduction",
        "Bufferbloat happens when a large buffer stays full and every packet waits behind it.",
        "2.1 Buffer sizing and bufferbloat (2004, 2011)",
        "The classic rule sizes the buffer at the bandwidth-delay product of the path we study.",
        "References", "[1] C. Naik et al., Study of 5G NR Network Against Mixed Traffic, 2025.",
    ])
    secs, skipped = clean(raw)
    d = dict(secs)
    assert list(d) == ["Abstract", "1 Introduction"], d
    assert "mean values." in d["Abstract"], "continuation line kept"
    assert "254.4" not in d["Abstract"] and "Figure" not in d["Abstract"], "noise dropped"
    assert "classic rule" in d["1 Introduction"], "subsection text stays in its section"
    assert "Some Title" in skipped and any("Naik" in s for s in skipped), "header and refs skipped"
    long = [("A", " ".join(f"Sentence number {i} is here." for i in range(400)))]
    ps = pack(long, lambda s: len(s.split()))
    assert all(p["tokens"] <= MAX_TOKENS for p in ps[:-1]) and len(ps) > 1
    assert ps[-1]["tokens"] >= MIN_TAIL or len(ps) == 1
    assert rhythm(long)[0]["sigma"] == 0.0
    assert len(sentences("First point here. •Second point here")) == 2, "bullets split"
    assert clean("Abstract\n10 Mbps link with a buffer of the size we chose here.\n")[0][0][0] == "Abstract", "out-of-sequence number is not a heading"
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()

    def boot():
        try:
            load(fp32="--fp32" in sys.argv)
        except Exception as e:
            STATE.update(status="error", msg=f"Model failed to load: {e}")

    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PORT
    # Windows lets two servers share a port when reuse is on; refuse instead of silently splitting requests.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)   # local only
    except OSError:
        sys.exit(f"Port {port} is busy. Is VerifAI already running? Close it, or use --port {port + 1}.")
    threading.Thread(target=boot, daemon=True).start()
    print(f"VerifAI on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(f"http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
