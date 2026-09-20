#!/usr/bin/env python3
"""
Whippet - a single-file, fully local, voice-controlled browser (accessibility assistant).

Started as a port of the Jev voice browser (Node) to Python, with the remote Jev decision API
replaced by LOCAL inference of the Von decision model from Hugging Face
(https://huggingface.co/wfzyx/von-1.0, ModernBERT-Large NLI head, Apache-2.0).

Runs on Apple Silicon via PyTorch MPS (Metal). ModernBERT's default SDPA attention
kernel does not compile under Metal, so on MPS the model is loaded with eager attention.

Architecture (all in this file):
  VonModel     - loads the pinned safetensors revision, batched premise/hypothesis NLI,
                 Choice (softmax over entailment, temperature-calibrated) and Noul (yes/no).
  spans        - code-generated candidate spans (query text, spoken URLs, "the second one").
  snapshot     - Playwright in-page collector -> compact element list with stable ids e01..
  Decider      - builds the typed question set (intent / target / site / text / url / scroll /
                 complete / is_command / destructive / tab_direction) and asks Von.
  policy       - pure code gates: ACT / WAIT / IGNORE / CONFIRM / CANCEL / DISAMBIGUATE.
  Browser/exec - Playwright Chromium (persistent profile or CDP attach), overlay feedback.
  Gate         - a second Von pass on every interim fragment: reflex (go back / stop / undo,
                 fired in realtime, barge-in over running work), remap ("when I say yeet this
                 tab, close the tab" -> persisted alias), ask (page question -> LLM), command.
  AliasStore   - user-taught phrasings, JSON on disk, matched by Von at gate time.
  LLM          - in-process mlx-lm chat model (Qwen) used ONLY for page summaries / questions.
  TTS          - in-process Pocket TTS (Kyutai, mlx-audio), prewarmed, streamed PCM to the browser;
                 speaks answers, confirmations, errors and "nothing matched" so the UI is optional.
  STT          - in-process Whisper (mlx-whisper) with VAD, interim transcripts and barge-in;
                 Chrome Web Speech remains the fallback when mlx-whisper is not installed.
  Controller   - streaming transcript handling, multi-step queues, reflexes, undo, page tools
                 (read aloud, find, zoom, where am I, what's on this page, help).
  Overlay      - the Whippet side panel (minimal black/white, "whippet" set in Borzoi) docked to the right of
                 every controlled page: conversation, "try saying" suggestions, status, mic orb, typed input,
                 guided-mode step chip. Alt+Space hides it,
                 Alt+V toggles voice mode, Alt+P push-to-talk (hold Space), Alt+K focuses the input. No separate UI page.
  Server       - localhost HTTP + WebSocket (debug/legacy UI, --fake-mic test harness).
  --test / --demo / --script - fixture tests, live demo, and scripted live runs ('~' streams
                 a line word by word as interim speech, '!' does not wait for it to finish).

Install (Apple Silicon, Python 3.11-3.13):
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt          # torch transformers safetensors huggingface_hub playwright aiohttp numpy
                                           # mlx mlx-lm mlx-audio mlx-whisper pymupdf  (no phonemizer / spacy stack)
  playwright install chromium
  python whippet.py --doctor               # checks every import + model and prints what to pip install

Usage:
  python whippet.py                          # open the browser with Whippet docked on the right
  python whippet.py --walkthrough            # guided, narrated demo (scripts/walkthrough.txt), then hands over
  python whippet.py --doctor                 # dependency / model diagnostics (why it is not speaking/hearing)
  python whippet.py --headless               # headless controlled browser
  python whippet.py --cdp ws://...           # attach to a running Chrome (--remote-debugging-port)
  python whippet.py --test                   # decision tests on fixtures (captured live if absent)
  python whippet.py --demo                   # word-by-word browser automation demo (headless)
  python whippet.py --say "go to wikipedia"   # single command, then exit
  python whippet.py --script scripts/realtime.txt      # scripted live run with assertions
  python whippet.py --llm mlx-community/<any-Qwen-MLX-repo>  # bigger LLM on your box
  python whippet.py --llm off                          # no LLM (summaries disabled)
  python whippet.py --tts off / --stt off              # browser speechSynthesis / Web Speech
  python whippet.py --voice-on --fake-mic scripts/voice1.wav  # voice-mode regression run
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import bisect
import difflib
import html as html_mod
import json
import logging
import math
import queue
import platform
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

log = logging.getLogger("vvb")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
MODEL_ID = "wfzyx/von-1.0"
# Latest main commit of wfzyx/von-1.0 (2026-09-19 22:38 UTC), carrying the Von-1.1
# model.safetensors release (fine-tuned on the operational decision corpus).
MODEL_REVISION = "d9258d6a8075d63c172ce152833ae64ae37e38b6"

INTENTS = [
    "navigate_url", "search_web", "click_element", "type_into_field", "select_option",
    "press_enter", "scroll_down", "scroll_up", "go_back", "go_forward", "reload",
    "open_new_tab", "close_tab", "switch_tab", "confirm", "cancel", "none",
]
TARGET_INTENTS = {"click_element", "type_into_field", "select_option"}
PAYLOAD_INTENTS = {"search_web", "type_into_field", "select_option"}

T = dict(
    intentConfidence=0.55, complete=0.6, isCommand=0.5, destructive=0.5,
    targetConfidence=0.45, targetTopProb=0.35, spanConfidence=0.35, candidateCount=3,
)
SILENCE_COMPLETE_MS = 900
PAYLOAD_SILENCE_MS = 600
DEBOUNCE_MS = 200
HIGHLIGHT_MS = 700
MAX_ELEMENTS = 100
MAX_ELEMENT_TEXT = 60
MAX_STATE_CHARS = 24_000
MAX_TRANSCRIPT_CHARS = 400
TARGET_POOL = 14  # elements offered to Von per target question (latency budget)
NAV_TIMEOUT = 15_000
DEFAULT_SEARCH_ENGINE = "duckduckgo"

SITE_HOME = {
    "google": "https://www.google.com/",
    "duckduckgo": "https://duckduckgo.com/",
    "youtube": "https://www.youtube.com/",
    "wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
    "github": "https://github.com/",
    "amazon": "https://www.amazon.com/",
    "reddit": "https://www.reddit.com/",
    "twitter_x": "https://x.com/",
    "hacker_news": "https://news.ycombinator.com/",
    "example_com": "https://example.com/",
}
SITE_SEARCH = {
    "google": "https://www.google.com/search?q=%s",
    "duckduckgo": "https://duckduckgo.com/?q=%s",
    "the_web": "https://duckduckgo.com/?q=%s",
    "youtube": "https://www.youtube.com/results?search_query=%s",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search=%s",
    "github": "https://github.com/search?q=%s&type=repositories",
    "amazon": "https://www.amazon.com/s?k=%s",
    "reddit": "https://www.reddit.com/search/?q=%s",
    "twitter_x": "https://x.com/search?q=%s",
    "hacker_news": "https://hn.algolia.com/?q=%s",
}
FIRST_HIT_URL = "https://duckduckgo.com/?q=!ducky+%s"
NAV_VERB_RE = re.compile(r"^(?:please\s+)?(?:go|navigate|head|take me|bring me|open|visit|pull up|load|show me)\b", re.I)
DEST_RE = re.compile(
    r"^(?:(?:please\s+)?(?:go|navigate|head|take me|bring me|open|visit|pull up|load|show me)\s+(?:up\s+|over\s+)?(?:to\s+)?)?"
    r"(?:the\s+)?(?:website\s+|site\s+|page\s+)?(?:of\s+|for\s+|called\s+)?(.+?)"
    r"(?:'s)?(?:\s+(?:website|site|page|homepage|home page|dot com))?[.!?]*$", re.I)
SITE_WORDS = {
    "google": r"google", "duckduckgo": r"duck\s*duck\s*go|ddg", "youtube": r"you\s*tube",
    "wikipedia": r"wikipedia|wiki", "github": r"git\s*hub", "amazon": r"amazon",
    "reddit": r"reddit", "twitter_x": r"twitter|\bx\.com\b|(?<=on )x\b", "hacker_news": r"hacker\s*news",
    "example_com": r"example\.com|example dot com", "the_web": r"the web|the internet|online",
}

# --------------------------------------------------------------------------------------
# Von model: local NLI inference (MPS / CUDA / CPU)
# --------------------------------------------------------------------------------------
def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class VonModel:
    """Von-1.0 (ModernBERT NLI). Labels: 0=entailment 1=neutral 2=contradiction."""

    def __init__(self, device: str = "auto", dtype: str = "auto", revision: str = MODEL_REVISION,
                 model_id: str = MODEL_ID, batch_size: int = 48, max_length: int = 512):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = pick_device(device)
        if dtype == "auto":
            dtype = "float16" if self.device in ("mps", "cuda") else "float32"
        self.dtype = getattr(torch, dtype)
        # Metal's shader compiler rejects ModernBERT's SDPA path ("unsupported
        # deferred-static-alloca-size function body"); eager attention runs fine and is as
        # fast for our short batches.
        attn = "eager" if self.device == "mps" else "sdpa"
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, dtype=self.dtype, attn_implementation=attn, use_safetensors=True,
        ).to(self.device).eval()
        cfg = self.model.config
        self.entail_idx = next((int(k) for k, v in cfg.id2label.items() if str(v).lower().startswith("entail")), 0)
        self.temperature = 1.0
        try:
            cal = json.loads(Path(hf_hub_download(model_id, "calibration.json", revision=revision)).read_text())
            self.temperature = float(cal.get("temperature", 1.0))
        except Exception:  # calibration file is optional
            pass
        self.batch_size = batch_size
        self.max_length = max_length
        self.revision = revision
        self._lock = threading.Lock()  # the gate may run from another thread while a decision is in flight
        self.load_ms = int((time.time() - t0) * 1000)
        self.calls = 0
        self.pairs = 0
        log.info("von loaded: %s %s attn=%s rev=%s T=%.3f in %dms", self.device, dtype, attn, revision[:8],
                 self.temperature, self.load_ms)

    @property
    def info(self) -> dict:
        return dict(model=MODEL_ID, revision=self.revision, device=self.device, dtype=str(self.dtype).split(".")[-1],
                    temperature=self.temperature, load_ms=self.load_ms)

    def entail_logits(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment logit for each (premise, hypothesis) pair."""
        if not pairs:
            return []
        torch = self.torch
        out: list[float] = []
        with self._lock:
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i:i + self.batch_size]
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], padding=True, truncation="only_first",
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.model(**enc).logits.float()
                out.extend(logits[:, self.entail_idx].tolist())
        self.calls += 1
        self.pairs += len(pairs)
        return out

    @staticmethod
    def softmax(xs: list[float], temp: float) -> list[float]:
        if not xs:
            return []
        m = max(xs)
        es = [math.exp((x - m) / temp) for x in xs]
        s = sum(es)
        return [e / s for e in es]


# Lazily-evaluated question batch: every question adds (premise, hypothesis) pairs; one model
# call scores them all; each question then reads its slice back. Keeps latency to ~1 forward pass.
class Batch:
    def __init__(self, model: VonModel):
        self.model = model
        self.pairs: list[tuple[str, str]] = []
        self.logits: list[float] = []

    def add(self, premise: str, hyps: list[str]) -> Callable[[], list[float]]:
        start = len(self.pairs)
        self.pairs.extend((premise, h) for h in hyps)
        end = len(self.pairs)
        return lambda: self.logits[start:end]

    def run(self) -> None:
        self.logits = self.model.entail_logits(self.pairs)

    def choice(self, premise: str, options: dict[str, str | list[str]], prior: dict[str, float] | None = None,
               agg: str = "lse"):
        """Returns a thunk -> {'choice', 'confidence', 'probabilities'} (Jev Choice shape).
        An option may carry several paraphrased hypotheses; its mass is the log-sum-exp (or max) of them."""
        keys = list(options)
        hyps = [[v] if isinstance(v, str) else list(v) for v in options.values()]
        get = self.add(premise, [h for hs in hyps for h in hs])
        temp = self.model.temperature

        def read() -> dict:
            flat = get()
            logits, i = [], 0
            for hs in hyps:
                seg = flat[i:i + len(hs)]
                i += len(hs)
                m = max(seg)
                logits.append(m if agg == "max" else m + temp * math.log(sum(math.exp((x - m) / temp) for x in seg)))
            probs = self.model.softmax(logits, temp)
            if prior:
                probs = [p * prior.get(k, 1.0) for p, k in zip(probs, keys)]
                s = sum(probs) or 1.0
                probs = [p / s for p in probs]
            d = dict(zip(keys, probs))
            ranked = sorted(probs, reverse=True)
            best = keys[probs.index(ranked[0])]
            conf = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            return {"choice": best, "confidence": round(conf, 4), "probabilities": {k: round(v, 4) for k, v in d.items()}}
        return read

    def noul(self, premise: str, yes: str, no: str):
        """Returns a thunk -> {'noul': p(yes)} (Jev Noul shape)."""
        get = self.add(premise, [yes, no])
        temp = self.model.temperature
        return lambda: {"noul": round(self.model.softmax(get(), temp)[0], 4)}

    def score(self, premise: str, levels: list[str]):
        """Ordinal expectation over levels -> {'score': float in [0, len-1]}."""
        get = self.add(premise, levels)
        temp = self.model.temperature

        def read() -> dict:
            probs = self.model.softmax(get(), temp)
            return {"score": round(sum(i * p for i, p in enumerate(probs)), 3), "probabilities": [round(p, 3) for p in probs]}
        return read


# --------------------------------------------------------------------------------------
# Spans: candidate extraction (code, never the model)
# --------------------------------------------------------------------------------------
TLDS = "com|org|net|io|ai|dev|co|edu|gov|de|uk|us|app|xyz|info|me|tv|ch|at|fr|nl|es|it"
FILLER_RE = re.compile(r"\b(please|thanks|thank you|now|okay|ok|um|uh|and then)\b", re.I)
TEXT_VERBS = [
    re.compile(r"\b(?:search|look)\s+(?:for|up)\s+", re.I),
    re.compile(r"\bsearch\s+(?:on\s+)?(?:google|duckduckgo|wikipedia|youtube|github|amazon|reddit|twitter|x|hacker news|the web)\s+for\s+", re.I),
    re.compile(r"\bsearch\s+", re.I),
    re.compile(r"\bgoogle\s+", re.I),
    re.compile(r"\bfind\s+", re.I),
    re.compile(r"\btype\s+(?:in\s+)?", re.I),
    re.compile(r"\benter\s+", re.I),
    re.compile(r"\bwrite\s+", re.I),
    re.compile(r"\bput\s+", re.I),
    re.compile(r"\bfill\s+(?:in\s+)?", re.I),
]
TRAILING_DEST_RE = re.compile(
    r"\s+(?:in|into|on|inside|to)\s+(?:the\s+)?(?:[\w-]+\s+){0,4}?(?:box|field|input|bar|form|textarea|search|wikipedia|youtube|google|duckduckgo|github|amazon|reddit|twitter|x|web)\b.*$",
    re.I)
LEADING_SITE_RE = re.compile(
    r"^(?:on\s+|in\s+)?(?:google|duckduckgo|wikipedia|youtube|github|amazon|reddit|twitter|x|hacker news|the web)\s+(?:for\s+)?", re.I)


def clean_transcript(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


POLITE_TAIL_RE = re.compile(r"(?:\s+(?:for me|would you|will you|could you|can you|if you can|if you could|real quick|right now|when you can))+$", re.I)


def _strip_filler(s: str) -> str:
    s = FILLER_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,!?]+$", "", s).strip()
    return POLITE_TAIL_RE.sub("", s).strip()


def _push_unique(out: list[str], value: str) -> None:
    v = _strip_filler(value)
    if not v or len(v) > 120:
        return
    if any(x.lower() == v.lower() for x in out):
        return
    out.append(v)


def extract_text_candidates(transcript: str) -> list[str]:
    t = clean_transcript(transcript)
    if not t:
        return []
    out: list[str] = []
    for m in re.finditer(r"[\"“”']([^\"“”']{1,120})[\"“”']", t):
        _push_unique(out, m.group(1))
    matches = [m for m in (rx.search(t) for rx in TEXT_VERBS) if m]
    matches.sort(key=lambda m: (m.start(), -len(m.group(0))))
    for m in matches:
        tail = t[m.end():]
        tail = LEADING_SITE_RE.sub("", tail)
        stripped = TRAILING_DEST_RE.sub("", tail)
        _push_unique(out, stripped)
        if stripped != tail:
            _push_unique(out, tail)
    idx = t.lower().find(" for ")
    if idx >= 0:
        _push_unique(out, TRAILING_DEST_RE.sub("", t[idx + 5:]))
    sp = t.find(" ")
    if sp > 0:
        _push_unique(out, TRAILING_DEST_RE.sub("", t[sp + 1:]))
    _push_unique(out, t)
    return out[:8]


def normalize_spoken_url(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"\s+dot\s+", ".", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    s = re.sub(r"\s+slash\s+", "/", s)
    s = re.sub(r"\bwww\s+", "www.", s)
    s = re.sub(r"\bh\s*t\s*t\s*p\s*s?\s*:\s*/\s*/", lambda m: "https://" if "s" in m.group(0) else "http://", s)
    return s


def normalize_spoken_email(text: str) -> str:
    return re.sub(r"\s+at\s+([a-z0-9-]+(?:\.[a-z0-9-]+)+)", r"@\1", normalize_spoken_url(text), flags=re.I)


def extract_url_candidates(transcript: str) -> list[str]:
    t = normalize_spoken_url(clean_transcript(transcript))
    if not t:
        return []
    rx = re.compile(rf"(?:https?://)?(?:[a-z0-9-]+\.)+(?:{TLDS})(?:/[^\s]*)?", re.I)
    out: list[str] = []
    for m in rx.finditer(t):
        v = re.sub(r"[.,!?]+$", "", m.group(0))
        # spoken emails ("bob at example dot com") are not URLs
        if re.search(r"\bat\s+" + re.escape(v), t):
            continue
        if v not in out:
            out.append(v)
    return out[:6]


def to_http_url(domainish: str) -> str:
    v = str(domainish).strip()
    return v if re.match(r"^https?://", v, re.I) else f"https://{v}"


NUMBER_WORDS = {
    "one": 1, "first": 1, "1": 1, "1st": 1, "two": 2, "second": 2, "2": 2, "2nd": 2,
    "three": 3, "third": 3, "3": 3, "3rd": 3, "four": 4, "fourth": 4, "4": 4, "4th": 4,
    "five": 5, "fifth": 5, "5": 5, "5th": 5, "six": 6, "sixth": 6, "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8, "nine": 9, "ninth": 9, "ten": 10, "tenth": 10, "last": -1,
}
NUMBER_HOMOPHONES = {"won": 1, "to": 2, "too": 2, "for": 4}
PICK_STOPWORDS = {"the", "number", "option", "pick", "choose", "select", "click", "take", "that", "please", "link",
                  "item", "result", "go", "with", "on", "yes", "this", "um", "uh", "one"}


def parse_candidate_pick(transcript: str, maximum: int = 5) -> int | None:
    t = re.sub(r"[.,!?]", "", clean_transcript(transcript).lower())
    if not t:
        return None
    words = t.split(" ")
    meaningful = [w for w in words if w not in PICK_STOPWORDS or w == "one"]
    # "the first one" -> ["first", "one"]; bare "one" -> ["one"]
    meaningful = [w for w in meaningful if not (w == "one" and len(meaningful) > 1 and any(x in NUMBER_WORDS for x in meaningful if x != "one"))]
    if not meaningful or len(meaningful) > 2:
        return None
    for w in meaningful:
        n = NUMBER_WORDS.get(w)
        if n and 0 < n <= maximum:
            return n
    if len(meaningful) == 1:
        n = NUMBER_HOMOPHONES.get(meaningful[0])
        if n and n <= maximum:
            return n
    return None


YES_RE = re.compile(r"^(?:yes|yeah|yep|yup|sure|confirm|confirmed|correct|right|do it|go ahead|ok(?:ay)?|please do|affirmative)(?:\s+(?:please|do it|go ahead|that's right))?[.!]?$", re.I)
NO_RE = re.compile(r"^(?:no|nope|nah|cancel|never ?mind|don't|do not|negative|stop|forget it|wrong)(?:\s+(?:thanks|thank you|don't))?[.!]?$", re.I)


def parse_yes_no(transcript: str) -> bool | None:
    t = _strip_filler(clean_transcript(transcript)).lower()
    if YES_RE.match(t):
        return True
    if NO_RE.match(t):
        return False
    return None


def ordinal_in(transcript: str) -> int | None:
    """'click the second result' -> 2 ; 'the last link' -> -1."""
    for w in re.sub(r"[.,!?]", "", transcript.lower()).split():
        if w in NUMBER_WORDS and w != "one":
            return NUMBER_WORDS[w]
    return None


# --------------------------------------------------------------------------------------
# Snapshot: page perception
# --------------------------------------------------------------------------------------
COLLECT_JS = r"""
() => {
  const SELECTOR = ["a[href]","button","input:not([type=hidden])","textarea","select","summary","[role=button]",
    "[role=link]","[role=tab]","[role=menuitem]","[role=option]","[role=checkbox]","[role=radio]","[role=switch]",
    "[role=searchbox]","[role=combobox]","[role=textbox]","[contenteditable=true]","[onclick]"].join(",");
  const win = window, doc = document;
  if (!win.__vbNextId) win.__vbNextId = 1;
  const vw = win.innerWidth, vh = win.innerHeight;
  const out = [];
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  for (const el of doc.querySelectorAll(SELECTOR)) {
    if (out.length >= 400) break;
    let rect; try { rect = el.getBoundingClientRect(); } catch { continue; }
    if (!rect || rect.width < 2 || rect.height < 2) continue;
    const style = win.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) continue;
    if (el.getAttribute("aria-hidden") === "true") continue;
    if (typeof el.checkVisibility === "function" && !el.checkVisibility()) continue;
    let id = el.getAttribute("data-vb-id");
    if (!id) { id = "e" + String(win.__vbNextId++).padStart(2, "0"); el.setAttribute("data-vb-id", id); }
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    let role = el.getAttribute("role") || "";
    if (!role) {
      if (tag === "a") role = "link";
      else if (tag === "button" || type === "submit" || type === "button" || type === "reset") role = "button";
      else if (tag === "select") role = "select";
      else if (tag === "textarea") role = "textbox";
      else if (tag === "summary") role = "button";
      else if (tag === "input") role = type === "search" ? "searchbox" : type === "checkbox" ? "checkbox" : type === "radio" ? "radio" : "textbox";
      else if (el.isContentEditable) role = "textbox";
      else role = "clickable";
    }
    const img = el.querySelector && el.querySelector("img[alt]");
    const name = clean(el.getAttribute("aria-label")) || clean(el.innerText) || clean(el.value) ||
      clean(el.getAttribute("placeholder")) || clean(el.getAttribute("title")) || (img && clean(img.getAttribute("alt"))) ||
      clean(el.getAttribute("name")) || "";
    const placeholder = clean(el.getAttribute("placeholder"));
    let href = "", fullHref = "";
    if (tag === "a") { try { const u = new URL(el.href, location.href); fullHref = u.href; href = u.hostname.replace(/^www\./, "") + (u.pathname !== "/" ? u.pathname : ""); } catch { href = ""; } }
    const inViewport = rect.bottom > 0 && rect.top < vh && rect.right > 0 && rect.left < vw;
    // structural signature: links that repeat the same ancestor shape form the page's item list
    let sig = "", p = el;
    for (let i = 0; i < 4 && p && p.parentElement; i++) {
      p = p.parentElement;
      const cls = (p.getAttribute("class") || "").split(/\s+/).filter(Boolean).slice(0, 2).join(".");
      sig += "/" + p.tagName.toLowerCase() + (cls ? "." + cls : "");
    }
    const chrome = !!el.closest("nav,header,footer,aside,[role=navigation],[role=banner],[role=contentinfo],[role=complementary]");
    out.push({ id, tag, role, text: name, placeholder, href, fullHref, type, sig, chrome,
      inputName: clean(el.getAttribute("name")) || clean(el.getAttribute("id")), inViewport,
      top: Math.round(rect.top + win.scrollY), left: Math.round(rect.left + win.scrollX) });
  }
  const video = [...doc.querySelectorAll("video")].some(v => v.getClientRects().length && (v.duration > 0 || !v.paused));
  return { url: location.href, title: doc.title, scrollY: win.scrollY, scrollHeight: doc.documentElement.scrollHeight,
    viewportHeight: vh, elements: out, video };
}
"""


def detect_site(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "generic"
    host = re.sub(r"^www\.", "", host)
    if host.endswith(("google.com", "google.co.uk")):
        return "google"
    if host.endswith("duckduckgo.com"):
        return "duckduckgo"
    if host.endswith("youtube.com"):
        return "youtube"
    if host.endswith("wikipedia.org"):
        return "wikipedia"
    if host.endswith("github.com"):
        return "github"
    if host.endswith(("amazon.com", "amazon.de", "amazon.co.uk")):
        return "amazon"
    if host.endswith("reddit.com"):
        return "reddit"
    if host == "x.com" or host.endswith("twitter.com"):
        return "twitter_x"
    if host.endswith("news.ycombinator.com"):
        return "hacker_news"
    if host == "example.com":
        return "example_com"
    if not host or url.startswith("about:"):
        return "blank"
    return "generic"


SEARCHY = re.compile(r"(^|[^a-z])(q|query|search|s|keyword|k|search_query)($|[^a-z])", re.I)


def find_search_box(elements: list[dict]) -> str | None:
    scored = []
    for e in elements:
        if e.get("role") not in ("searchbox", "textbox", "combobox"):
            continue
        s = 0
        if e.get("role") == "searchbox" or e.get("type") == "search":
            s += 5
        if SEARCHY.search(e.get("inputName") or ""):
            s += 3
        if re.search("search", e.get("placeholder") or "", re.I) or re.search("search", e.get("text") or "", re.I):
            s += 3
        if e.get("inViewport"):
            s += 1
        scored.append((s, e.get("top", 0), e["id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2] if scored and scored[0][0] >= 3 else None  # a real search signal, not just "an input is visible"


def _truncate(s: Any, n: int) -> str:
    s = str(s or "")
    return s[: n - 1].rstrip() + "…" if len(s) > n else s


def mark_items(raw: list[dict], viewport_h: int = 800) -> None:
    """Find the page's main list of content links (results, videos, stories, posts...) without knowing
    the site: the group of links sharing one ancestor shape with the most title-like text wins,
    favouring groups that start high on the page.
    Each member gets item=n (1-based, document order)."""
    groups: dict[str, list[dict]] = {}
    for e in raw:
        if e.get("role") == "link" and e.get("href") and e.get("sig") and not e.get("chrome") and len(e.get("text") or "") >= 12:
            groups.setdefault(e["sig"], []).append(e)
    best: list[dict] = []
    best_score = 0.0
    for g in groups.values():
        hrefs = {e.get("fullHref") or e["href"] for e in g}
        if len(hrefs) < 3 or not any(e.get("inViewport") for e in g):
            continue  # a footer menu is not the page's content list
        score = sum(min(len(e["text"]), 80) for e in g)
        score /= 1.0 + min(e.get("top", 0) for e in g) / max(viewport_h, 1)
        if score > best_score:
            best, best_score = g, score
    seen: set[str] = set()
    n = 0
    for e in best:
        h = e.get("fullHref") or e["href"]
        if h in seen:
            continue
        seen.add(h)
        n += 1
        e["item"] = n
    # everything laid out between item k and item k+1 (comment links, bylines, buttons) belongs to item k
    tops = sorted((e["top"], e["item"]) for e in best if e.get("item"))
    for e in raw:
        if e.get("item") or not tops or e.get("chrome"):
            continue
        k = bisect.bisect_right(tops, (e.get("top", 0), 10 ** 9)) - 1
        if k < 0:
            continue
        nxt = tops[k + 1][0] if k + 1 < len(tops) else tops[k][0] + 300
        if tops[k][0] <= e.get("top", 0) < nxt and e.get("top", 0) - tops[k][0] < 600:
            e["near"] = tops[k][1]


def compact_elements(raw: list[dict], max_elements: int = MAX_ELEMENTS, max_text: int = MAX_ELEMENT_TEXT,
                     max_chars: int = MAX_STATE_CHARS, viewport_h: int = 800) -> list[dict]:
    mark_items(raw, viewport_h)
    ordered = sorted(raw, key=lambda e: (0 if e.get("inViewport") else 1, e.get("top", 0), e.get("left", 0)))
    seen: set[str] = set()
    out: list[dict] = []
    for e in ordered:
        is_input = e.get("role") in ("textbox", "searchbox", "combobox", "select", "checkbox", "radio")
        text = _truncate(e.get("text") or e.get("placeholder"), max_text)
        if not text and not is_input:
            continue
        key = f"{e.get('role')}|{text.lower()}|{e.get('href') or ''}"
        if key in seen:
            continue
        seen.add(key)
        rec: dict[str, Any] = {"id": e["id"], "role": e.get("role"), "text": text}
        if e.get("placeholder") and e["placeholder"] != text:
            rec["placeholder"] = _truncate(e["placeholder"], 40)
        if e.get("href"):
            rec["href"] = _truncate(e["href"], 50)
        if not e.get("inViewport"):
            rec["below_fold"] = True
        if e.get("item"):
            rec["item"] = e["item"]
        if e.get("near"):
            rec["near"] = e["near"]
        if e.get("chrome"):
            rec["chrome"] = True
        out.append(rec)
        if len(out) >= max_elements:
            break
    while len(out) > 5 and len(json.dumps(out)) > max_chars:
        del out[max(5, int(len(out) * 0.8)):]
    return out


def build_snapshot(page_data: dict, **extra: Any) -> dict:
    elements = page_data.get("elements", [])
    return {
        "url": page_data.get("url", ""), "title": page_data.get("title", ""), "site": detect_site(page_data.get("url", "")),
        "scrollY": page_data.get("scrollY", 0), "scrollHeight": page_data.get("scrollHeight", 0),
        "viewportHeight": page_data.get("viewportHeight", 0), "searchBoxId": find_search_box(elements),
        "elements": compact_elements(elements, viewport_h=page_data.get("viewportHeight") or 800),
        "rawCount": len(elements), "video": bool(page_data.get("video")), **extra,
    }


# --------------------------------------------------------------------------------------
# Decider: typed questions for Von
# --------------------------------------------------------------------------------------
INTENT_HYP = {
    "navigate_url": "The user wants to go to a website by name or address.",
    "search_web": "The user wants to search for a topic, query or phrase.",
    "click_element": "The user wants to click, tap or press a link, button, result, video or item on the page.",
    "type_into_field": "The user wants to type or enter some text into a text field, search box or form input.",
    "select_option": "The user wants to choose an option from a dropdown or select menu.",
    "press_enter": "The user wants to hit the Enter key (or Return key) on the keyboard.",
    "scroll_down": "The user wants to scroll down the page.",
    "scroll_up": "The user wants to scroll up the page.",
    "go_back": "The user wants to go back to the previous page.",
    "go_forward": "The user wants to go forward to the next page in history.",
    "reload": "The user wants to reload or refresh the page.",
    "open_new_tab": "The user wants to create a brand-new, empty browser tab.",
    "close_tab": "The user wants to close the current tab.",
    "switch_tab": "The user wants to move to another tab that is already open (the next, previous or a named tab).",
    "confirm": "The user says yes, approving the pending action.",
    "cancel": "The user says no or never mind, cancelling the pending action.",
    "none": "The user is not giving the browser any command; it is chit-chat, filler or an incomplete fragment.",
}
# Intent is asked hierarchically: a coarse category question, then one question per category
# (all scored in the same forward pass). P(intent) = P(category) * P(intent | category).
# 17 flat NLI hypotheses spread probability thinly; small typed questions are what Von is good at.
# extra paraphrases for head-to-head runoffs between two close intents
RUNOFF_HYP: dict[str, list[str]] = {
    "search_web": ["The user wants to look something up and see search results for it.",
                   "The user is asking for a web search."],
    "type_into_field": ["The user wants the browser to type the words into a text box on the current page, not run a web search.",
                        "The user is dictating text to be typed into a field."],
    "click_element": ["The user is pointing at something on the page to click."],
    "navigate_url": ["The user wants to open a website's home page."],
}

INTENT_CATEGORIES: dict[str, str | list[str]] = {
    "navigate": "The user wants to go to a different website, naming the site or its web address.",
    "search": ["The user wants to look something up: run a search for a query or topic.",
               "The user wants to search a particular website for something."],
    "interact": "The user wants to click, type into, or choose something that is on the page they are looking at.",
    "scroll": [INTENT_HYP["scroll_down"], INTENT_HYP["scroll_up"]],
    "history": [INTENT_HYP["go_back"], INTENT_HYP["go_forward"], INTENT_HYP["reload"]],
    "tabs": [INTENT_HYP["open_new_tab"], INTENT_HYP["close_tab"], INTENT_HYP["switch_tab"]],
    "none": INTENT_HYP["none"],
}
INTENT_SUBS = {
    "navigate": ["navigate_url"],
    "search": ["search_web"],
    "interact": ["click_element", "type_into_field", "select_option", "press_enter"],
    "scroll": ["scroll_down", "scroll_up"],
    "history": ["go_back", "go_forward", "reload"],
    "tabs": ["open_new_tab", "close_tab", "switch_tab"],
    "none": ["none"],
}
STEP_SEP = re.compile(r"(\s*[,;]\s*(?:and\s+)?(?:th[ea]n\s+)?|\.\s+(?:and\s+)?th[ea]n\s+|\s+and\s+th[ea]n\s+|\s+th[ea]n\s+|\s+and\s+)", re.I)
SCROLL_LITTLE = re.compile(r"\b(a bit|a little|little|slightly|a touch|a tad|a few lines)\b", re.I)
SCROLL_END = re.compile(r"\b(all the way|to the (bottom|top|end)|bottom|top|end)\b", re.I)
SEARCH_VERB_RE = re.compile(r"\b(search|look up|look for|google|find|query)\b", re.I)
ELEMENT_NOUN_RE = re.compile(r"\b(link|button|article|entry|section|heading|option|item|menu|tab|result|story|post|headline)\b", re.I)
ROLE_INTENT = {"link": "click_element", "button": "click_element", "tab": "click_element", "menuitem": "click_element",
               "clickable": "click_element", "checkbox": "click_element", "radio": "click_element", "textbox": "type_into_field",
               "searchbox": "type_into_field", "combobox": "type_into_field", "select": "select_option"}


def best_lexical_element(transcript: str, elements: list[dict]) -> tuple[dict | None, float]:
    """Element whose visible label best matches the spoken words (0 = no overlap)."""
    words = _tokens(transcript)
    best, score = None, 0.0
    tl = transcript.lower()
    for e in elements:
        full = (e.get("text") or e.get("placeholder") or "").lower().strip()
        if not full:
            continue
        et = {w for w in _tokens(full) if not w.isdigit()}  # "12 comments" ~ "comments"
        host = (urlparse(e.get("href") or "").hostname or e.get("href") or "").lower().split("/")[0]
        s = 0.0
        if re.search(rf"\b{re.escape(full)}\b", tl):
            s = 1.0
        elif et and et <= words:
            s = 0.8
        elif host and _tokens(host.replace("www.", "").rsplit(".", 1)[0]) & words:
            s = 0.8  # the spoken words name the link's domain ("the github link")
        elif et and words & et:
            s = 0.5 * len(words & et) / len(et)
        if s > score:
            best, score = e, s
    return best, score


STOP = {"the", "a", "an", "on", "to", "in", "into", "click", "press", "tap", "open", "select", "choose", "go", "hit",
        "link", "button", "tab", "field", "box", "please", "that", "this", "one", "of", "and", "type", "enter", "result"}
# generic names for "an entry of the page's main list": "the second story" points at item 2, nothing more
ITEM_NOUNS = {"result", "results", "item", "items", "entry", "story", "stories", "post", "posts", "video", "videos",
              "article", "articles", "headline", "headlines", "hit", "hits", "match", "matches", "repo", "repository",
              "listing", "option", "thing", "link", "links", "one", "top", "main", "list"}


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP}


def element_label(elements: list[dict], eid: str) -> str:
    for e in elements:
        if e["id"] == eid:
            return f"{e.get('role')} \"{e.get('text') or e.get('placeholder') or ''}\""
    return eid


def _element_desc(e: dict, note: str = "", short: bool = False) -> str:
    parts = [f"{e.get('role')} \"{(e.get('text') or '')[:60 if short else 200]}\""]
    if e.get("placeholder") and e.get("placeholder") != e.get("text"):
        parts.append(f"placeholder \"{e['placeholder']}\"")
    if e.get("href"):
        parts.append(f"href {e['href'][:50] if short else e['href']}")
    if note:
        parts.append(note)
    return ", ".join(parts)


def target_pool(transcript: str, snapshot: dict, intent: str) -> tuple[list[dict], dict[str, str]]:
    """Pick the elements worth asking Von about; returns (elements, notes)."""
    elements: list[dict] = snapshot.get("elements") or []
    page_host = re.sub(r"^www\.", "", (urlparse(snapshot.get("url") or "").hostname or ""))
    words = _tokens(transcript)
    words |= {w for w in re.findall(r"[a-z0-9]+", normalize_spoken_email(transcript).lower())}
    scored: list[tuple[float, dict]] = []
    notes: dict[str, str] = {}
    # "results": the page's main item list (see mark_items); else external links, in order
    results = sorted((e for e in elements if e.get("item")), key=lambda e: e["item"])
    if not results:
        results = [e for e in elements if e.get("role") == "link" and e.get("href") and not e["href"].startswith(page_host)
                   and not e.get("below_fold") and not e.get("chrome")]
        if len(results) < 3:  # a stray "Sign in" is not a result list
            results = []
    for i, e in enumerate(results[:10]):
        notes[e["id"]] = f"item #{i + 1} in the main list"
    n = ordinal_in(transcript)
    want = len(results) if n == -1 else n
    for e in elements:
        if e.get("near") and e["near"] <= 10:
            notes[e["id"]] = f"part of item #{e['near']}"
    for e in elements:
        et = _tokens(f"{e.get('text', '')} {e.get('placeholder', '')} {e.get('href', '')}")
        overlap = len(words & et)
        full = (e.get("text") or "").lower().strip()
        s = float(overlap) + (2.0 if full and full in transcript.lower() else 0.0)
        if intent == "type_into_field" and e.get("role") in ("textbox", "searchbox", "combobox"):
            s += 0.5
        if intent == "select_option" and e.get("role") == "select":
            s += 0.5
        if intent == "click_element" and e.get("role") in ("link", "button", "tab", "menuitem", "clickable"):
            s += 0.1
        if e.get("item") and intent == "click_element":
            s += 0.3 / e["item"]  # content items outrank page chrome; earlier items first
        if want and e.get("near") == want and overlap:
            s += 1.0  # "the comments link on the first story"
        if e.get("chrome"):
            s -= 0.1
        if e.get("below_fold"):
            s -= 0.05
        scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    pool = [e for s, e in scored if s > 0.15][:TARGET_POOL]
    if n is not None and results:
        target = results[n - 1] if 0 < n <= len(results) else (results[-1] if n == -1 else None)
        if target and target not in pool:
            pool.insert(0, target)
    if len(pool) < 6:
        for _, e in scored:
            if e not in pool:
                pool.append(e)
            if len(pool) >= 6:
                break
    return pool[:TARGET_POOL], notes


def lexical_site(transcript: str) -> str | None:
    t = transcript.lower()
    for site, rx in SITE_WORDS.items():
        if re.search(rx, t):
            return site
    return None


class Decider:
    """Builds the question set for a (transcript, snapshot) pair and asks Von."""

    def __init__(self, model: VonModel):
        self.model = model

    def _premise(self, transcript: str, snapshot: dict, pending: dict | None, tabs: int | None) -> str:
        p = f'The user said to the voice-controlled web browser: "{transcript}".'
        if snapshot:
            p += f' The current page is "{snapshot.get("title") or ""}" ({snapshot.get("site")}, {snapshot.get("url") or "about:blank"}).'
        if pending:
            p += f" The browser is waiting for the user to confirm or cancel: {describe(pending)}."
        if tabs and tabs > 1:
            p += f" {tabs} tabs are open."
        return p

    def verify_target(self, transcript: str, label: str) -> float:
        """Second opinion on an externally proposed element: p(this element is what was asked for)."""
        b = Batch(self.model)
        r = b.noul(f'The user said to the web browser: "{transcript}". The proposed element on the page is "{label}".',
                   "This element is the one the user asked for.", "This element is not what the user asked for.")
        b.run()
        return r()["noul"]

    def split_steps(self, transcript: str, known: Callable[[str], bool] | None = None) -> list[str]:
        """Split a finished utterance into sequential commands ("open a new tab, go to X and type Y").
        Code proposes the cut points (commas, 'then', 'and'); Von decides whether each right-hand
        fragment is a command of its own or just the tail of the previous one ("salt and pepper").
        `known(fragment)` marks fragments that are built-in tools ("slide four", "zoom in") as steps outright."""
        parts = STEP_SEP.split(clean_transcript(transcript))
        frags = [p for p in parts[0::2] if p is not None]
        seps = parts[1::2]
        if len(frags) < 2:
            return [transcript]
        b = Batch(self.model)
        reads = []
        for f in frags[1:]:
            plain = f'The user said to the web browser: "{f}".'
            reads.append((b.choice(plain, INTENT_CATEGORIES),
                          b.noul(f'Fragment: "{f}"',
                                 "This fragment is an imperative sentence: it begins with a verb telling the browser what to do.",
                                 "This fragment is not an imperative; it is a noun phrase or the tail end of a longer sentence.")))
        b.run()
        steps = [frags[0]]
        for f, sep, (r_cat, r_imp) in zip(frags[1:], seps, reads):
            if (known and known(f)) or (r_cat()["choice"] != "none" and r_imp()["noul"] >= 0.5):
                steps.append(f)
            else:
                steps[-1] = f"{steps[-1]}{sep}{f}"
        return [s.strip() for s in steps if s.strip()]

    def decide(self, transcript: str, snapshot: dict, pending: dict | None = None, tabs: int | None = None) -> dict:
        t0 = time.time()
        transcript = clean_transcript(transcript)[:MAX_TRANSCRIPT_CHARS]
        snapshot = snapshot or {"elements": [], "site": "blank", "url": "about:blank", "title": ""}
        premise = self._premise(transcript, snapshot, pending, tabs)
        plain = f'The user said to the web browser: "{transcript}".'
        text_c = extract_text_candidates(transcript)
        url_c = extract_url_candidates(transcript)
        answers: dict[str, Any] = {}

        # ---- pass 1: intent, is_command, complete -----------------------------------
        b = Batch(self.model)
        ip = plain if not pending else premise
        cats = dict(INTENT_CATEGORIES)
        if pending:
            cats["confirm"] = INTENT_HYP["confirm"]
            cats["cancel"] = INTENT_HYP["cancel"]
        # Snapshot-grounded prior: naming a visible element favours acting on the page, and the
        # interaction matching that element's role ("press delete account" -> button -> click).
        el, el_score = best_lexical_element(transcript, snapshot.get("elements") or [])
        grounded = el is not None and el_score >= 0.8
        r_cat = b.choice(ip, cats, prior={"interact": 1.0 + 3.0 * el_score} if grounded else None)
        r_subs = {}
        for cat, subs in INTENT_SUBS.items():
            if len(subs) > 1:
                sp = None
                if cat == "interact" and grounded and ROLE_INTENT.get(el.get("role", "")):
                    sp = {ROLE_INTENT[el["role"]]: 1.0 + 3.0 * el_score}
                r_subs[cat] = b.choice(ip, {k: INTENT_HYP[k] for k in subs}, prior=sp)
        r_cmd = b.noul(f'Speech heard by a voice-controlled web browser: "{transcript}"',
                       "The speaker is commanding the browser to do something.",
                       "The speaker is talking to another person or thinking aloud, not commanding the browser.")
        r_complete = b.noul(plain, "The command is complete: it has its verb and the required object.",
                            "The command is cut off before its object; more words are clearly coming.")
        b.run()
        cat_ans = r_cat()
        probs: dict[str, float] = {}
        for cat, pc in cat_ans["probabilities"].items():
            subs = INTENT_SUBS.get(cat, [cat])
            if len(subs) == 1:
                probs[subs[0]] = pc
            else:
                for k, ps in r_subs[cat]()["probabilities"].items():
                    probs[k] = pc * ps
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        # confidence = how far the runner-up trails the winner (0.55 => winner has >2x its mass);
        # a plain difference would shrink just because the mass is split over 16 leaves.
        conf = 1.0 - ranked[1][1] / max(ranked[0][1], 1e-9) if len(ranked) > 1 else 1.0
        answers["intent"] = {"choice": ranked[0][0], "confidence": round(conf, 4),
                             "probabilities": {k: round(v, 4) for k, v in probs.items()}, "category": cat_ans}
        answers["is_command"] = r_cmd()
        answers["complete"] = r_complete()
        els = snapshot.get("elements") or []
        boxes = [e for e in els if e.get("role") in ("textbox", "searchbox", "combobox")]
        top = ranked[0][0]
        # a search with no site named and no search box here, or typing with nothing to type into:
        # the page disagrees with the words, so put the two payload intents head to head
        afford_clash = ((top == "search_web" and boxes and not snapshot.get("searchBoxId") and not lexical_site(transcript))
                        or (top == "type_into_field" and not boxes))
        # "open the link about X" / "go to X": X is the exact visible label of a link or button on this page and no
        # site, address or search verb was spoken - that is a click, whatever the verb suggests to the model
        if (top in ("navigate_url", "search_web") and grounded and el_score >= 1.0 and not url_c and not lexical_site(transcript)
                and ROLE_INTENT.get(el.get("role", "")) == "click_element" and not SEARCH_VERB_RE.search(transcript)
                and (top == "navigate_url" or ELEMENT_NOUN_RE.search(transcript))):
            answers["intent"]["choice"], answers["intent"]["confidence"] = "click_element", 1.0
            answers["intent"]["grounded"] = el.get("text")
            ranked, afford_clash = [("click_element", 1.0), ("none", 0.0)], False
        if afford_clash:
            other = "type_into_field" if top == "search_web" else "search_web"
            ranked = [ranked[0], (other, answers["intent"]["probabilities"].get(other, 0.0))]
        if (afford_clash or (conf < T["intentConfidence"] and ranked[1][1] > 0.15)) and ranked[0][0] != "none" and ranked[1][0] != "none":
            # runoff: the two front-runners head to head, with the page in view ("type in the quick brown
            # fox" on a page with a big text box is typing, not a search)
            a, c = ranked[0][0], ranked[1][0]
            facts = (f" The page has {len(boxes)} text field(s)" + (", one of them a search box." if snapshot.get("searchBoxId")
                     else " and no search box.")) if snapshot else ""
            b = Batch(self.model)
            r = b.choice(premise + facts, {k: [INTENT_HYP[k], *RUNOFF_HYP.get(k, [])] for k in (a, c)})
            b.run()
            ro = r()
            answers["intent"]["choice"] = ro["choice"]
            answers["intent"]["confidence"] = ro["probabilities"][ro["choice"]]  # two-way: the winner's share
            answers["intent"]["runoff"] = ro["probabilities"]
        intent = answers["intent"]["choice"]
        for k in INTENTS:
            answers["intent"]["probabilities"].setdefault(k, 0.0)

        # ---- pass 2: dependent questions -------------------------------------------
        b = Batch(self.model)
        readers: dict[str, Callable[[], dict]] = {}

        if intent in ("navigate_url", "search_web"):
            lex = lexical_site(transcript)
            if lex:
                answers["site"] = {"choice": lex, "confidence": 1.0, "probabilities": {lex: 1.0}}
            else:
                sites = {k: f"The site the user means is {k.replace('_', ' ')}." for k in list(SITE_HOME) + ["the_web"]}
                sites["none"] = "The user did not name any specific website."
                readers["site"] = b.choice(plain, sites)
            if url_c:
                if len(url_c) == 1:
                    answers["url_span"] = {"choice": url_c[0], "confidence": 1.0, "probabilities": {url_c[0]: 1.0}}
                else:
                    readers["url_span"] = b.choice(premise, {u: f'The web address the user wants to open is "{u}".' for u in url_c})

        if intent in PAYLOAD_INTENTS and text_c:
            lex_site = lexical_site(transcript) if intent == "search_web" else None
            if lex_site and lex_site in SITE_WORDS:
                # "search reddit for X": the site name is where to search, never part of the query
                site_rx = re.compile(SITE_WORDS[lex_site], re.I)
                text_c = [c for c in text_c if not site_rx.search(c)] or text_c
            if len(text_c) == 1:
                answers["text_span"] = {"choice": text_c[0], "confidence": 1.0, "probabilities": {text_c[0]: 1.0}}
            else:
                what = "search query" if intent == "search_web" else "text to type"
                readers["text_span"] = b.choice(plain, {c: f'The exact {what}, copied verbatim, is "{c}".' for c in text_c},
                                                prior={c: 1.0 / (1.0 + i) ** 3 for i, c in enumerate(text_c)})

        target_ids: list[str] = []
        if intent in TARGET_INTENTS:
            pool, notes = target_pool(transcript, snapshot, intent)
            if pool:
                listing = "; ".join(f"{e['id']}: {_element_desc(e, notes.get(e['id'], ''), short=True)}" for e in pool)
                tp = premise + f" Candidate elements on the page: {listing}."
                t_opts = {e["id"]: f"The element the user means is {e['id']}, the {_element_desc(e, notes.get(e['id'], ''))}."
                          for e in pool}
                t_opts["none"] = "None of these elements is the one the user means."
                # lexical prior: exact label matches are almost always right
                prior: dict[str, float] = {}
                tl = transcript.lower()
                twords = _tokens(tl)
                any_overlap = False
                exact_ids: list[str] = []
                part_hit = False  # the words name a part of the numbered item ("the comments link on the first story")
                n = ordinal_in(transcript)
                n_items = sum(1 for v in notes.values() if v.startswith("item #"))
                want = n_items if n == -1 else n
                for i, e in enumerate(pool):
                    full = (e.get("text") or e.get("placeholder") or "").lower().strip()
                    words = _tokens(full)
                    p = 1.0 - 0.01 * i  # DOM order breaks exact ties
                    in_item = n is not None and e.get("near") == want
                    role_ok = ROLE_INTENT.get(e.get("role", ""), intent) == intent
                    if full and role_ok and re.search(rf"\b{re.escape(full)}\b", tl):
                        # a whole-label match is nearly always right - unless the user is pointing at
                        # a numbered item ("the comments link on the first story" is not the nav "comments")
                        p *= 20.0 if n is None or in_item else 1.0
                        if words:
                            exact_ids.append(e["id"])
                    elif words and words <= twords:
                        p *= 2.0
                    if in_item:
                        p *= 3.0  # a part of the item the user numbered
                        if words & twords:
                            p *= 3.0
                            part_hit = True
                    if words & twords or (e.get("href") and _tokens(e["href"]) & twords):
                        any_overlap = True
                    if e.get("href") and full and full.replace("www.", "") == e["href"].lower():
                        p *= 0.3  # bare-URL line under a result title duplicates the title link
                    if ROLE_INTENT.get(e.get("role", ""), intent) != intent:
                        p *= 0.2  # can't type into a button, or click-select a textbox
                    prior[e["id"]] = p
                content = twords - set(NUMBER_WORDS)
                if content and not any_overlap and n is None and intent not in PAYLOAD_INTENTS:
                    # the user named something that is not on this page (typed text is not expected to be)
                    prior = {k: v * 0.05 for k, v in prior.items()}
                    prior["none"] = 8.0
                if len(exact_ids) == 1 and n is None:
                    # the user read one label out verbatim and nothing else matches whole: that is the one
                    prior = {k: v * (1.0 if k == exact_ids[0] else 0.05) for k, v in prior.items()}
                pure_ordinal = n is not None and not (content - ITEM_NOUNS)
                if n is not None and not n_items:
                    # "the first video" on a page with no visible list: nothing to count yet (still loading?)
                    prior = {k: 0.0 for k in prior}
                    prior["none"] = 1.0
                elif pure_ordinal and 0 < want <= n_items:
                    # "the second story": counting is arithmetic, not a judgement call
                    hit = next(eid for eid, note in notes.items() if note.startswith(f"item #{want} "))
                    prior = {k: (1.0 if k == hit else 0.0) for k in prior}
                elif n is not None and not part_hit:
                    # "the first video / second result / last story": an ordinal over the item list
                    for eid, note in notes.items():
                        if note.startswith(f"item #{want} "):
                            prior[eid] = prior.get(eid, 1.0) * 6.0
                sure = None  # settled without the model: a label read out verbatim, or plain counting
                if len(exact_ids) == 1 and n is None:
                    sure = exact_ids[0]
                elif pure_ordinal and 0 < want <= n_items:
                    sure = next(eid for eid, note in notes.items() if note.startswith(f"item #{want} "))
                if sure:
                    answers["target"] = {"choice": sure, "confidence": 1.0,
                                         "probabilities": {k: (1.0 if k == sure else 0.0) for k in prior}}
                else:
                    readers["target"] = b.choice(tp, t_opts, prior=prior)
                target_ids = [e["id"] for e in pool]

        if intent in ("scroll_down", "scroll_up"):
            sprior = {"little": 1.0, "page": 1.0, "end": 1.0}
            if SCROLL_LITTLE.search(transcript):
                sprior["little"] = 4.0
            if SCROLL_END.search(transcript):
                sprior["end"] = 4.0
            readers["scroll_amount"] = b.choice(plain, {
                "little": "The user wants to scroll only a little bit.",
                "page": "The user wants to scroll about one screen or page.",
                "end": "The user wants to scroll all the way to the end (top or bottom).",
            }, prior=sprior)
        if intent == "switch_tab":
            readers["tab_direction"] = b.choice(premise, {
                "next": "The user wants the next tab.", "previous": "The user wants the previous tab.",
                "first": "The user wants the first tab.",
            })
        b.run()
        for k, r in readers.items():
            answers[k] = r()
        if "scroll_amount" in readers:
            sa = answers["scroll_amount"]
            sa["score"] = float(["little", "page", "end"].index(sa["choice"]))
        if "site" in readers and answers["site"]["confidence"] < T["spanConfidence"]:
            answers["site"]["choice"] = "none"

        # ---- pass 3: destructiveness of the concrete element action --------------------
        if intent in TARGET_INTENTS or intent == "press_enter":
            tgt = answers.get("target", {}).get("choice")
            label = element_label(snapshot.get("elements") or [], tgt) if tgt and tgt != "none" else "the focused element"
            verb = "type into" if intent == "type_into_field" else "press Enter in" if intent == "press_enter" else "click"
            act_desc = f"{verb} the {label}"
            b = Batch(self.model)
            r_d1 = b.noul(f"Action: {act_desc}.", "This has real-world side effects (a purchase, payment, deletion, sent message or post).",
                          "This only navigates or reveals information.")
            r_d2 = b.noul(f"The browser will {act_desc}.", "Doing this places an order, pays money, deletes data, sends a message or publishes a post.",
                          "Doing this merely opens a page, shows content or focuses a field.")
            b.run()
            answers["destructive"] = {"noul": max(r_d1()["noul"], r_d2()["noul"])}
        else:
            answers["destructive"] = {"noul": 0.0}
        for k in ("site", "text_span", "url_span", "target", "tab_direction"):
            answers.setdefault(k, {"choice": "none", "confidence": 0.0, "probabilities": {}})
        answers.setdefault("scroll_amount", {"score": 1.0, "choice": "page"})
        return {"answers": answers, "candidates": {"text": text_c, "url": url_c, "target": target_ids, "transcript": transcript},
                "latencyMs": int((time.time() - t0) * 1000), "transcript": transcript}


# --------------------------------------------------------------------------------------
# Front gate: what kind of speech is arriving right now? Runs on every interim transcript,
# on the last fragment only, so "…no wait, go back" fires while the user is still talking.
# --------------------------------------------------------------------------------------
GATE_HYP: dict[str, list[str]] = {
    "back": ["The user tells the browser to go back to the previous page.",
             "The user says to go back, or return to where they were before."],
    "forward": ["The user tells the browser to go forward again to the next page in history."],
    "stop": ["The user says stop, wait, hold on, cancel, never mind or scratch that.",
             "The user wants the browser to abort or drop what it is doing right now."],
    "undo": ["The user says undo, or wants the last thing the browser did reversed."],
    "remap": ["The user is renaming a voice command so that a different phrase triggers it from now on.",
              "The user asks the browser to change, switch or map which words mean a certain command.",
              "The user is teaching the browser a new way of saying a command.",
              "The user says: switch or change the command for one thing to some other words."],
    "ask": ["The user asks a question about the page, or asks for a summary, explanation or a read-out of its content.",
            "The user wants to know what the page says or means."],
    "command": ["The user gives the browser an ordinary instruction: open a site, search, click, type or scroll.",
                "The user tells the browser to open, close or switch a tab or window.",
                "The user wants to switch to the previous or next tab (a tab, not a page in history).",
                "The user tells the browser to reload or refresh the page.",
                "The user tells the browser to scroll, or to click or select something on the page."],
    "chatter": ["The user is talking to someone else or thinking aloud, not addressing the browser."],
}
REFLEXES = {"back": "go_back", "forward": "go_forward", "stop": "stop", "undo": "undo"}
GUIDE_CUE_RE = re.compile(r"^(?:(?:please |ok |okay )?(?:guide|walk|take|talk) me through(?: this| it)?|step by step[,:]?|one step at a time[,:]?|"
                          r"guided(?: mode)?[,:]?)\s*[,:]?\s*", re.I)
GUIDE_GO_RE = re.compile(r"^(?:next|go|continue|proceed|do it|go ahead|okay|ok|yes|yep|ready|carry on|next step)(?: please)?[.!]?$", re.I)
GUIDE_SKIP_RE = re.compile(r"^skip(?: (?:that|this|it|this step|that step))?(?: please)?[.!]?$", re.I)
GATE_REFLEX_CONF = 0.6  # mass of the whole reflex family; which reflex is the argmax inside it
GATE_REMAP_P = 0.25
GATE_ASK_P = 0.6
REMAP_NOULS = [
    ('The user said: "{s}".', "The user is defining what a new phrase should mean from now on, for example: when I say X, do Y.",
     "The user is simply telling the browser to do something right now."),
    ('"{s}"', "This defines a custom voice command: a new phrase is being mapped to an action.",
     "This is an ordinary request, not a definition of a new phrase."),
]
REMAP_CUE_RE = re.compile(r"\b(when(?:ever)? i say|if i say|the command for|rename|remap|alias|shortcut|means?\b|from now on|"
                          r"call (?:it|that|this)|instead of saying)", re.I)
GATE_MAX_REFLEX_WORDS = 5
FILLERS = {"hmm", "hm", "um", "uh", "er", "erm", "ok", "okay", "so", "well", "oh", "ah", "yeah", "right", "like", "please",
           "actually", "no", "wait", "now", "then", "and", "just", "can", "you"}
GATE_REFLEX_CONF_INTERIM = 0.9  # words still arriving: only act on a clear reflex
# an interjection starts a new fragment even without a comma: "search for cats actually go back"
INTERJECT_RE = re.compile(r"\s+(?=(?:actually|no wait|wait no|wait,?|never ?mind|scratch that|hold on|hang on|stop|no,? )\b)", re.I)
# "switch the command for close tab to yeet this tab": words that only frame the remap
REMAP_FRAME_RE = re.compile(
    r"\b(i want you to|i'd like you to|can you|could you|please|from now on|going forward|in the future|instead|"
    r"the command|the phrase|command|phrase|trigger|shortcut|alias|voice command|for saying|for|switch|change|rename|remap|"
    r"map|make|set|let|have|so that|so|that|when i say|whenever i say|if i say|when i tell you|i say|say|saying|"
    r"should|will|now|also|too)\b", re.I)
REMAP_CONNECT = {  # connector at the cut -> which side is the NEW phrase (r = right, l = left)
    "to": "r", "into": "r", "as": "r", "with": "r", "becomes": "r", "become": "r", "is now": "r", "should be": "r",
    "means": "l", "mean": "l", "meaning": "l", "is": "l", "be": "l", "equals": "l", "does": "l", "do": "l",
    "then": "l", "and": "?", ",": "?",
}
REMAP_FORGET_RE = re.compile(r"\b(forget|remove|delete|undo|drop|reset|clear)\b", re.I)
# "forget everything" / "reset all my phrases": wipe every taught phrase
REMAP_RESET_RE = re.compile(r"\b(forget|reset|clear)\b.*\b(everything|all of (?:them|it)|"
                            r"(?:all|every) (?:(?:my|the|your|of my|of the) )?(?:phrases?|aliases|commands?|shortcuts|words?))\b", re.I)
STOPWORDS = {"the", "this", "that", "a", "an", "it", "my", "me", "to", "on", "in", "of", "up", "please", "now"}


def _norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def last_fragment(text: str) -> str:
    """The part of the utterance after the last cut point (comma / then / and / interjection)."""
    parts = [p for p in STEP_SEP.split(INTERJECT_RE.sub(", ", text))[0::2] if p and p.strip()]
    return parts[-1].strip() if parts else text.strip()


@dataclass
class Alias:
    phrase: str
    command: str
    created: float = field(default_factory=time.time)


class AliasStore:
    """User-taught phrasings ("yeet this tab" -> "close this tab"), persisted as JSON."""

    def __init__(self, path: Path | None):
        self.path = path
        self.items: list[Alias] = []
        if path and path.exists():
            try:
                self.items = [Alias(**a) for a in json.loads(path.read_text())]
            except Exception as e:
                log.warning("aliases unreadable (%s): starting empty", e)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([a.__dict__ for a in self.items], indent=1))

    def add(self, phrase: str, command: str) -> Alias:
        self.items = [a for a in self.items if _norm_words(a.phrase) != _norm_words(phrase)]
        a = Alias(phrase=phrase.strip(), command=command.strip())
        self.items.append(a)
        self.save()
        return a

    def remove(self, phrase: str) -> Alias | None:
        for a in self.items:
            if _norm_words(a.phrase) == _norm_words(phrase):
                self.items.remove(a)
                self.save()
                return a
        return None

    def clear(self) -> int:
        n = len(self.items)
        self.items = []
        self.save()
        return n

    @staticmethod
    def similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, " ".join(_norm_words(a)), " ".join(_norm_words(b))).ratio()

    def near(self, text: str, floor: float = 0.45) -> list[Alias]:
        """Aliases lexically close enough to be worth asking Von about (contained or ≥floor similar)."""
        out = []
        tw = set(_norm_words(text))
        for a in self.items:
            aw = set(_norm_words(a.phrase))
            if aw and (aw <= tw or self.similarity(text, a.phrase) >= floor):
                out.append(a)
        return out[:8]

    def apply(self, text: str, alias: Alias) -> str:
        """Substitute the taught phrase (or the whole fragment if it is a paraphrase) with the command."""
        rx = re.compile(r"\b" + r"\W+".join(map(re.escape, _norm_words(alias.phrase))) + r"\b", re.I)
        if rx.search(text):
            return rx.sub(alias.command, text, count=1)
        return alias.command

    def as_list(self) -> list[dict]:
        return [a.__dict__ for a in self.items]


class Gate:
    """One cheap Von pass over the live fragment: reflex / remap / ask / command / chatter, plus alias matching."""

    def __init__(self, model: VonModel, aliases: AliasStore):
        self.model = model
        self.aliases = aliases

    def classify(self, fragment: str) -> dict:
        t0 = time.time()
        b = Batch(self.model)
        plain = f'The user said to the voice-controlled web browser: "{fragment}".'
        r_kind = b.choice(plain, GATE_HYP, agg="max")
        r_remap = [b.noul(p.format(s=fragment), y, n) for p, y, n in REMAP_NOULS]
        near = self.aliases.near(fragment)
        r_alias = None
        if near:
            opts: dict[str, str | list[str]] = {a.phrase: [f'The user said "{a.phrase}", or a close paraphrase of it.',
                                                            f'"{fragment}" means the same as "{a.phrase}".'] for a in near}
            opts["none"] = "The user said something else entirely."
            r_alias = b.choice(f'The user said: "{fragment}".', opts)
        b.run()
        kind = r_kind()
        probs = kind["probabilities"]
        reflex_p = sum(probs[k] for k in REFLEXES)
        fam = {"reflex": reflex_p, **{k: probs[k] for k in probs if k not in REFLEXES}}
        remap_p = max([probs["remap"]] + [r()["noul"] for r in r_remap])
        cue = bool(REMAP_CUE_RE.search(fragment))
        # a remap names two things (phrase + command): short fragments are never one
        if len(_norm_words(fragment)) >= 4 and (remap_p >= GATE_REMAP_P or (cue and remap_p >= 0.1)):
            fam["remap"] = max(fam["remap"], remap_p, fam.get("reflex", 0) + 0.01, fam.get("command", 0) + 0.01)
        if REMAP_RESET_RE.search(fragment) and len(_norm_words(fragment)) <= 6:
            fam["remap"] = 0.99
        family = max(fam, key=fam.get)
        choice = max(REFLEXES, key=probs.get) if family == "reflex" else family
        out = {"fragment": fragment, "kind": choice, "confidence": round(fam[family], 4), "family": family,
               "probabilities": probs, "remapP": round(remap_p, 4), "alias": None,
               "latencyMs": int((time.time() - t0) * 1000)}
        if r_alias:
            al = r_alias()
            best = next((a for a in near if a.phrase == al["choice"]), None)
            if best:
                sim = self.aliases.similarity(fragment, best.phrase)
                fw, pw = set(_norm_words(fragment)) - STOPWORDS, set(_norm_words(best.phrase)) - STOPWORDS
                contained = set(_norm_words(best.phrase)) <= set(_norm_words(fragment))
                overlap = len(fw & pw) / max(1, len(fw | pw))
                if sim >= 0.85 or contained or (al["probabilities"][best.phrase] >= 0.6 and overlap >= 0.5):
                    out["alias"] = {"phrase": best.phrase, "command": best.command, "p": al["probabilities"][best.phrase],
                                    "similarity": round(sim, 2)}
        return out

    def parse_remap(self, transcript: str) -> dict:
        """'switch the command for close tab to yeet this tab' -> {'phrase': 'yeet this tab', 'command': 'close tab'}.
        Code proposes every cut of the de-framed words; Von says which side is a browser command it
        already understands; the connector word at the cut (to / means / …) says which side is new."""
        text = clean_transcript(transcript)
        quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,60})[\"“”'‘’]", text)
        if REMAP_RESET_RE.search(text):
            return {"forget": "*"}
        if REMAP_FORGET_RE.search(text):
            target = quoted[0] if quoted else REMAP_FRAME_RE.sub(" ", REMAP_FORGET_RE.sub(" ", text))
            return {"forget": " ".join(w for w in _norm_words(target) if w not in ("the", "a", "my", "about")) or None}
        # frame words become "," (a cut marker that is never part of a phrase); connectors stay as words
        toks = re.findall(r"[a-z0-9']+|,", REMAP_FRAME_RE.sub(" , ", text.lower()))
        toks = [w for i, w in enumerate(toks) if not (w == "," and (i == 0 or toks[i - 1] == ","))]
        cuts: list[tuple[str, str, str]] = []  # (left, right, orientation hint)
        if len(quoted) >= 2:
            cuts.append((quoted[0], quoted[1], "?"))
        for i in range(1, len(toks)):
            left, right = toks[:i], toks[i:]
            hint = "?"
            two = " ".join(right[:2])
            if two in REMAP_CONNECT:
                hint, right = REMAP_CONNECT[two], right[2:]
            elif right[0] in REMAP_CONNECT:
                hint, right = REMAP_CONNECT[right[0]], right[1:]
            elif right[0] == ",":
                right = right[1:]
            left = left[len(left) - left[::-1].index(","):] if "," in left else left  # a phrase never spans a frame word
            right = right[:right.index(",")] if "," in right else right
            if 1 <= len(left) <= 7 and 1 <= len(right) <= 7:
                cuts.append((" ".join(left), " ".join(right), hint))
        if not cuts:
            return {}
        b = Batch(self.model)
        sides = sorted({s for l_, r_, _ in cuts for s in (l_, r_)})
        reads = {s: b.choice(f'The user said to the web browser: "{s}".', INTENT_CATEGORIES) for s in sides}
        formed = {s: b.noul(f'Someone said: "{s}".', "This is a complete, well-formed instruction that starts with what to do.",
                            "This is a fragment: it starts mid-sentence or trails off.") for s in sides}
        b.run()
        cmdness = {s: 1.0 - reads[s]()["probabilities"].get("none", 0.0) for s in sides}
        formed = {s: f()["noul"] for s, f in formed.items()}
        best, best_score = None, 0.0
        for left, right, hint in cuts:
            for phrase, command, side in ((right, left, "r"), (left, right, "l")):
                score = cmdness[command] * (0.3 + 0.7 * formed[command]) * (1.0 - 0.6 * cmdness[phrase])
                if hint == side:
                    score *= 3.0
                elif hint != "?":
                    score *= 0.3
                if score > best_score:
                    best, best_score = {"phrase": phrase, "command": command, "score": round(score, 3),
                                        "commandness": round(cmdness[command], 3)}, score
        return best or {}


# --------------------------------------------------------------------------------------
# LLM helper (in-process MLX): summaries / questions about the page - never on the fast path
# --------------------------------------------------------------------------------------
DEFAULT_LLM = "mlx-community/Qwen3.5-4B-MLX-4bit"  # swap for e.g. mlx-community/Qwen3.6-35B-A3B-4bit on a bigger Mac
LLM_PAGE_CHARS = 9000
LLM_SYSTEM = ("You are the reading assistant of a voice-controlled web browser used by a person who cannot see "
              "the screen well. Answer from the page text given. Be brief and speak plainly: two to four sentences "
              "unless asked for more. Never output markdown, lists or URLs.")


class LLM:
    """Qwen (or any mlx-lm chat model) loaded in this process; generation streams from a worker thread."""

    def __init__(self, model_id: str = DEFAULT_LLM):
        self.model_id = model_id
        self.model = None
        self.tok = None
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.load_ms = 0
        self.stats = {"calls": 0, "tokens": 0, "tps": 0.0}

    def load(self) -> None:
        """Loads lazily on the first question, not at startup: a 4B model pulling in ~2.5 GB while Von, Whisper and
        Chromium are starting is what made the whole Mac crawl. Safe to call repeatedly."""
        with self._load_lock:
            if self.ready or self.error:
                return
            self._load()

    def _load(self) -> None:
        t0 = time.time()
        try:
            from mlx_lm import load
            self.model, self.tok = load(self.model_id)
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("LLM unavailable (%s) - summaries/questions disabled", self.error)
        self.load_ms = int((time.time() - t0) * 1000)
        if self.ready:
            log.info("llm loaded: %s in %dms", self.model_id, self.load_ms)

    @property
    def info(self) -> dict:
        return {"model": self.model_id, "ready": self.ready, "error": self.error, "load_ms": self.load_ms, "lazy": not self.ready and not self.error,
                **self.stats}

    def generate(self, messages: list[dict], max_tokens: int = 220, on_token: Callable[[str], None] | None = None) -> str:
        """Blocking; call from a worker thread. `on_token` receives the growing text."""
        if not self.ready:
            raise RuntimeError(self.error or "llm not loaded")
        from mlx_lm import stream_generate
        with self._lock:
            try:
                prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            except TypeError:  # template without a thinking switch
                prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True)
            text, last = "", None
            for r in stream_generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens):
                text += r.text
                last = r
                if on_token:
                    on_token(text)
            self.stats["calls"] += 1
            if last:
                self.stats["tokens"] += last.generation_tokens
                self.stats["tps"] = round(last.generation_tps, 1)
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


PAGE_TEXT_JS = r"""
() => {
  const sel = window.getSelection && window.getSelection().toString();
  const main = document.querySelector('main, article, [role=main]') || document.body;
  const skip = new Set([...main.querySelectorAll('script,style,noscript,nav,footer,header,aside,[aria-hidden=true]')]);
  const parts = [];
  for (const el of main.querySelectorAll('h1,h2,h3,h4,h5,h6,p,li,td,th,blockquote,pre,figcaption,dd,dt')) {
    if ([...skip].some(s => s.contains(el)) || el.closest('h1,h2,h3,h4,h5,h6,p,li,td,th,blockquote,pre,figcaption,dd,dt') !== el) continue;
    const t = (el.innerText || '').replace(/\s+/g, ' ').trim();
    if (t) parts.push(t);
  }
  const text = (parts.length ? parts.join('\n') : (main.innerText || '')).replace(/[ \t]+/g, ' ').replace(/\n{2,}/g, '\n').trim();
  return {title: document.title, url: location.href, selection: sel || '', text};
}
"""


# --------------------------------------------------------------------------------------
# Speech: Pocket TTS (mlx-audio) and Whisper STT (mlx-whisper), both in-process
# --------------------------------------------------------------------------------------
DEFAULT_TTS = "mlx-community/pocket-tts"  # Kyutai Pocket TTS: ~150 ms to first audio, text in (no phonemizer stack)
DEFAULT_VOICE = "alba"  # alba, marius, javert, jean, fantine, cosette, eponine, azelma
TTS_CHUNK_S = 0.25  # stream audio out in ~250 ms pieces
TTS_TARGET_RMS = 0.08
TTS_MAX_CHARS = 1400  # per utterance; read-aloud is chunked upstream
TTS_RATE = 24_000
DEFAULT_STT = "mlx-community/whisper-large-v3-turbo"
STT_RATE = 16_000
SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")
# Whisper is prompted with the command vocabulary so taught words ("yeet") are not "corrected" away
STT_PROMPT = "Voice commands for a web browser: go to Wikipedia, search for cats, click the second result, scroll down, go back, open a new tab, close this tab, stop."
# what Whisper tends to produce from silence / breath / room noise
STT_HALLUCINATIONS = re.compile(r"^(?:\W*(?:thank you|thanks for watching|thank you for watching|you|bye|the end|so|hmm|oh|uh|um|okay|"
                                r"subtitles? by|amara\.org|please subscribe|i'm sorry)\W*)$", re.I)
VAD_START_FRAMES = 3  # 20 ms frames above the floor before speech is declared
VAD_END_MS = 650  # silence that ends an utterance
DEFAULT_STT_INTERIM = "mlx-community/whisper-base-mlx"
VAD_INTERIM_MS = 450  # how often the growing utterance is re-transcribed while something can be interrupted
VAD_INTERIM_IDLE_MS = 1100  # ... and when nothing is running (interims are then only for display)
VAD_PREROLL_MS = 300
VAD_INTERIM_WINDOW_S = 8  # interim passes only look at the last few seconds (reflexes are short; keeps them cheap)
VAD_MIN_UTTER_MS = 320
VAD_MAX_UTTER_S = 28


# import name -> pip requirement
PIP_FOR_MODULE = {
    "mlx_audio": "mlx-audio", "mlx_whisper": "mlx-whisper", "mlx_lm": "mlx-lm", "mlx": "mlx",
    "torch": "torch", "transformers": "transformers", "safetensors": "safetensors", "huggingface_hub": "huggingface_hub",
    "playwright": "playwright  (then: playwright install chromium)", "aiohttp": "aiohttp", "numpy": "numpy", "soundfile": "soundfile",
    "pymupdf": "pymupdf",
}


def install_hint(e: BaseException) -> str:
    """Turn an import/model error into the exact command that fixes it."""
    if isinstance(e, ModuleNotFoundError) and e.name:
        root = e.name.split(".")[0]
        return f"fix: pip install {PIP_FOR_MODULE.get(root, root)}"
    s = str(e)
    if "401" in s or "403" in s or "offline" in s.lower() or "connection" in s.lower():
        return "fix: the model download failed - check network / HF_TOKEN, then retry"
    return ""


def doctor(tts_id: str, stt_id: str, llm_id: str) -> int:
    """`--doctor`: import every dependency, then actually load Pocket TTS / Whisper, printing the pip fix for each failure."""
    import importlib
    print(f"Whippet doctor  python={sys.version.split()[0]}  platform={platform.platform()}  machine={platform.machine()}")
    print(f"  interpreter {sys.executable}\n  pip         {sys.executable} -m pip install ...  (install with THIS interpreter)")
    problems = 0
    if platform.machine() != "arm64":
        print("  note     not Apple Silicon: MLX (Pocket TTS/Whisper/Qwen) needs an M-series Mac; Von still runs on CPU/CUDA")
    groups = [("core", ["torch", "transformers", "safetensors", "huggingface_hub", "playwright", "aiohttp", "numpy"]),
              ("Pocket TTS (spoken feedback)", ["mlx", "mlx_audio"]),
              ("Whisper (hearing)", ["mlx_whisper"]), ("Qwen (page questions)", ["mlx_lm"]),
              ("Slides (PDF decks)", ["pymupdf"])]
    for title, mods in groups:
        print(f"\n[{title}]")
        for m in mods:
            try:
                mod = importlib.import_module(m)
                ver = str(mod.__dict__.get("__version__", ""))
                print(f"  ok       {m} {ver}")
            except Exception as e:
                problems += 1
                print(f"  MISSING  {m}: {type(e).__name__}: {str(e)[:120]}\n           {install_hint(e) or 'fix: pip install ' + PIP_FOR_MODULE.get(m, m)}")
    try:
        import subprocess
        r = subprocess.run([sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"], capture_output=True, text=True, timeout=30)
        print("  ok       playwright chromium" if r.returncode == 0 else "  note     run: playwright install chromium")
    except Exception:
        pass
    print("\n[models]")
    if tts_id.lower() not in ("off", "none", ""):
        tts = TTS(tts_id)
        tts.load()
        if tts.ready:
            print(f"  ok       Pocket TTS {tts_id} loaded in {tts.load_ms}ms  (voice {tts.voice}, {TTS_RATE} Hz)")
        else:
            problems += 1
            print(f"  FAILED   Pocket TTS {tts_id}: {tts.error}\n           {install_hint(Exception(tts.error or ''))}")
    if stt_id.lower() not in ("off", "none", ""):
        stt = STT(stt_id)
        stt.load()
        if stt.ready:
            print(f"  ok       Whisper {stt_id} loaded in {stt.load_ms}ms")
        else:
            problems += 1
            print(f"  FAILED   Whisper {stt_id}: {stt.error}")
    print(f"  info     Von decision model: {MODEL_ID} @ {MODEL_REVISION[:7]} (downloaded on first run)")
    print(f"  info     LLM: {llm_id} (loads on the first page question, not at startup)")
    print("\nall good - run: python whippet.py" if not problems else f"\n{problems} problem(s) - apply the fixes above, then rerun --doctor")
    return 0 if problems == 0 else 1


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Short first chunk -> first audio fast; long sentences are cut at commas."""
    out: list[str] = []
    for s in SENTENCE_RE.split(text.strip()):
        s = s.strip()
        if not s:
            continue
        while len(s) > max_chars:
            cut = s.rfind(",", 0, max_chars)
            cut = cut if cut > 40 else max_chars
            out.append(s[:cut].strip(" ,"))
            s = s[cut:].strip(" ,")
        if s:
            out.append(s)
    return out


class TTS:
    """Pocket TTS (Kyutai, via mlx-audio) loaded once and kept warm: plain text in, no phonemizer/G2P stack.
    `speak` streams PCM chunks to `sink` as they are generated (~150 ms to first audio); `interrupt` drops the rest."""

    def __init__(self, model_id: str = DEFAULT_TTS, voice: str = DEFAULT_VOICE, speed: float = 1.0):
        self.model_id, self.voice, self.speed = model_id, voice, speed
        self.model = None
        self.ready = False
        self.error: str | None = None
        self.load_ms = 0
        self.gain = 1.0  # per-voice loudness normalisation, measured at warm-up
        self.gen = 0  # generation counter: bumping it cancels whatever is queued
        self._text = ""
        self._until = 0.0  # when playback of the current utterance should have ended
        self.spoke_at = 0.0
        self.stats = {"utterances": 0, "first_audio_ms": [], "synth_ms": []}
        self._q: "queue.Queue[tuple[int, str, Callable[[bytes, int, bool], None], Callable[[], None] | None]]" = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, name="tts", daemon=True).start()

    def load(self) -> None:
        import numpy as np
        t0 = time.time()
        try:
            from mlx_audio.tts.utils import load_model
            self.model = load_model(self.model_id)
            warm = np.concatenate(list(self._chunks("Whippet is ready.")))  # compiles kernels, loads the voice
            rms = float(np.sqrt((warm ** 2).mean())) if warm.size else 0.0
            self.gain = min(3.0, TTS_TARGET_RMS / rms) if rms > 1e-4 else 1.0
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("TTS unavailable (%s) - spoken feedback disabled. %s", self.error, install_hint(e))
        self.load_ms = int((time.time() - t0) * 1000)
        if self.ready:
            log.info("tts loaded: %s (%s) in %dms", self.model_id, self.voice, self.load_ms)

    @property
    def info(self) -> dict:
        fa = self.stats["first_audio_ms"]
        return {"model": self.model_id, "voice": self.voice, "ready": self.ready, "error": self.error, "load_ms": self.load_ms,
                "utterances": self.stats["utterances"], "avg_first_audio_ms": int(sum(fa) / len(fa)) if fa else 0}

    def _chunks(self, text: str):
        """float32 audio chunks for `text`, streamed out of the model as they are decoded."""
        import numpy as np
        for r in self.model.generate(text, voice=self.voice, stream=True, streaming_interval=TTS_CHUNK_S):
            a = np.asarray(r.audio, dtype=np.float32)
            if a.size:
                yield a

    def speak(self, text: str, sink: Callable[[bytes, int, bool], None], on_done: Callable[[], None] | None = None,
              interrupt: bool = True) -> int:
        """Queue `text`. `sink(pcm16_bytes, generation, last)` is called per chunk from the TTS thread."""
        if interrupt:
            self.interrupt()
        self.gen += 1
        self._q.put((self.gen, text, sink, on_done))
        return self.gen

    @property
    def speaking(self) -> str:
        """Text still coming out of the speaker (estimated from audio length; the page confirms the actual end)."""
        return self._text if time.time() < self._until else ""

    def interrupt(self) -> None:
        self.gen += 1  # queued / in-flight utterances see a stale generation and stop
        self.done_speaking()

    def _worker(self) -> None:
        import numpy as np
        while True:
            gen, text, sink, on_done = self._q.get()
            if gen != self.gen or not self.ready:
                continue
            t0 = time.time()
            self._text, self._until = text, time.time() + 2.0  # synthesis is under way
            text = re.sub(r"\s+", " ", text).strip()[:TTS_MAX_CHARS]
            first = True
            try:
                with self._lock:
                    for a in self._chunks(text):
                        if gen != self.gen:
                            break
                        pcm = (np.clip(a * self.gain, -1, 1) * 32767).astype("<i2").tobytes()
                        now = time.time()
                        if first:
                            self.stats["first_audio_ms"].append(int((now - t0) * 1000))
                            first = False
                        self._until = max(self._until, now) + len(pcm) / 2 / TTS_RATE + 0.4
                        sink(pcm, gen, False)
            except Exception as e:
                log.warning("tts failed: %s", e)
            if gen == self.gen:
                sink(b"", gen, True)  # end marker: the page knows the utterance is complete
            self.stats["synth_ms"].append(int((time.time() - t0) * 1000))
            self.stats["utterances"] += 1
            if on_done and gen == self.gen:
                on_done()

    def done_speaking(self) -> None:
        self._until = 0.0
        self.spoke_at = time.time()


class STT:
    """mlx-whisper, loaded once; `transcribe` takes 16 kHz float32 audio and returns text ('' for non-speech)."""

    def __init__(self, model_id: str = DEFAULT_STT, interim_model_id: str = DEFAULT_STT_INTERIM):
        self.model_id = model_id
        # interim (streaming) passes run every ~0.5 s while you talk; a small model keeps them off the GPU
        # budget the browser, Von and the TTS need. Finals still use the big model.
        self.interim_model_id = interim_model_id if interim_model_id.lower() not in ("off", "none", "") else model_id
        self.ready = False
        self.error: str | None = None
        self.load_ms = 0
        self.prompt = STT_PROMPT
        self.stats = {"calls": 0, "ms": []}
        self._lock = threading.Lock()

    def load(self) -> None:
        import numpy as np
        t0 = time.time()
        try:
            import mlx_whisper  # noqa: F401
            self.ready = True
            self.transcribe(np.zeros(STT_RATE, dtype=np.float32), final=True)  # downloads / warms the weights
            if self.interim_model_id != self.model_id:
                self.transcribe(np.zeros(STT_RATE, dtype=np.float32), final=False)
        except Exception as e:
            self.ready = False
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("STT unavailable (%s) - falling back to the browser's speech recognition. %s", self.error, install_hint(e))
        self.load_ms = int((time.time() - t0) * 1000)
        if self.ready:
            log.info("stt loaded: %s in %dms", self.model_id, self.load_ms)

    @property
    def info(self) -> dict:
        ms = self.stats["ms"]
        return {"model": self.model_id, "interim_model": self.interim_model_id, "ready": self.ready, "error": self.error, "load_ms": self.load_ms,
                "calls": self.stats["calls"], "avg_ms": int(sum(ms) / len(ms)) if ms else 0}

    def set_vocabulary(self, phrases: list[str]) -> None:
        extra = ", ".join(p for p in phrases if p)[:300]
        self.prompt = STT_PROMPT + (f" Custom phrases: {extra}." if extra else "")

    def transcribe(self, audio, final: bool = False) -> str:
        import mlx_whisper
        t0 = time.time()
        with self._lock:
            r = mlx_whisper.transcribe(audio, path_or_hf_repo=self.model_id if final else self.interim_model_id, language="en", fp16=True, temperature=0.0,
                                       condition_on_previous_text=False, initial_prompt=self.prompt,
                                       no_speech_threshold=0.5, logprob_threshold=-1.2, verbose=None)
        self.stats["calls"] += 1
        self.stats["ms"].append(int((time.time() - t0) * 1000))
        segs = [s for s in r.get("segments", []) if s.get("no_speech_prob", 0) < 0.7 or s.get("avg_logprob", -9) > -0.6]
        text = " ".join(s["text"].strip() for s in segs).strip()
        text = re.sub(r"\s+", " ", text)
        if STT_HALLUCINATIONS.match(text):
            return ""
        return text


class VoiceSession:
    """Energy VAD over 16 kHz PCM from the browser mic; drives interim + final transcripts while the user
    is still talking, so reflexes ("go back", "stop") fire mid-sentence."""

    def __init__(self, stt: STT, on_text: Callable[[str, bool, str], None], on_speech_start: Callable[[], None] | None = None,
                 loop: asyncio.AbstractEventLoop | None = None, hot: Callable[[], bool] | None = None):
        import numpy as np
        self.np = np
        self.stt = stt
        self.on_text = on_text
        self.on_speech_start = on_speech_start
        self.hot = hot  # "is there anything a reflex could interrupt right now?" -> dense interims
        self.loop = loop or asyncio.get_event_loop()
        self.floor = 0.006  # adaptive noise floor (RMS, full scale 1.0)
        self.frames: list = []  # rolling pre-roll
        self.utter: list = []
        self.in_speech = False
        self.above = 0
        self.silent_ms = 0
        self.since_interim_ms = 0
        self.uid = ""
        self.n = 0
        self.busy = False
        self.last_interim = ""
        self.level = 0.0
        self.empty_interims = 0
        self.noise = False  # set by the STT thread: a long "utterance" that transcribes to nothing is room noise

    def feed(self, pcm16: bytes) -> None:
        np = self.np
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        frame = STT_RATE // 50  # 20 ms
        for i in range(0, len(audio) - frame + 1, frame):
            f = audio[i:i + frame]
            rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
            self.level = rms
            voiced = rms > max(self.floor * 3.0, 0.012)
            if not voiced:
                self.floor = 0.97 * self.floor + 0.03 * rms  # track the room when nobody talks
            if self.noise:  # STT said the last few seconds were not words: end the utterance, adopt its level as floor
                self.noise = False
                self.in_speech, self.frames, self.utter = False, [], []
                self.floor = max(self.floor, rms)
                continue
            if not self.in_speech:
                self.frames.append(f)
                del self.frames[:-(VAD_PREROLL_MS // 20)]
                self.above = self.above + 1 if voiced else 0
                if self.above >= VAD_START_FRAMES:
                    self.in_speech, self.silent_ms, self.since_interim_ms = True, 0, 0
                    self.n += 1
                    self.uid = f"v{int(time.time() * 1000)}-{self.n}"
                    self.utter = list(self.frames)
                    self.last_interim = ""
                    self.empty_interims = 0
                    if self.on_speech_start:
                        self.on_speech_start()
                continue
            self.utter.append(f)
            self.silent_ms = 0 if voiced else self.silent_ms + 20
            self.since_interim_ms += 20
            dur_s = len(self.utter) * 0.02
            if dur_s > 4.0:  # nobody talks for this long without a pause: creep the floor up so steady noise ends it
                self.floor = 0.995 * self.floor + 0.005 * rms
            if self.silent_ms >= VAD_END_MS or dur_s >= VAD_MAX_UTTER_S:
                self.in_speech = False
                speech_s = dur_s - self.silent_ms / 1000
                if speech_s * 1000 >= VAD_MIN_UTTER_MS:
                    self._kick(final=True)
                self.frames = []
            elif self.since_interim_ms >= (VAD_INTERIM_MS if self.hot is None or self.hot() else VAD_INTERIM_IDLE_MS) \
                    and not self.busy and dur_s >= 0.6:
                self.since_interim_ms = 0
                self._kick(final=False)

    def _kick(self, final: bool) -> None:
        audio = self.np.concatenate(self.utter if final else self.utter[-(VAD_INTERIM_WINDOW_S * 50):])
        uid = self.uid
        self.busy = True

        def work() -> None:
            try:
                text = self.stt.transcribe(audio, final=final)
            except Exception as e:
                log.warning("stt failed: %s", e)
                text = ""
            finally:
                self.busy = False
            if not text or (not final and text == self.last_interim):
                if final and self.last_interim:  # the final pass rejected it but we already showed words: settle them
                    self.loop.call_soon_threadsafe(self.on_text, self.last_interim, True, uid)
                if not final and not text and not self.last_interim:
                    self.empty_interims += 1
                    if self.empty_interims >= 2 and len(audio) >= STT_RATE * 2.5:
                        self.noise = True
                return
            if not final:
                self.last_interim = text
                self.empty_interims = 0
            self.loop.call_soon_threadsafe(self.on_text, text, final, uid)
        threading.Thread(target=work, name="stt", daemon=True).start()


# --------------------------------------------------------------------------------------
# Built-in page tools: phrasing patterns (site-agnostic) and spoken feedback
# --------------------------------------------------------------------------------------
TOOL_READ_RE = re.compile(r"^(?:read|read out|read aloud|read me|start reading|narrate)(?:\s+(?:me|out|aloud|out loud))?"
                          r"(?:\s+(?:the|this|that))?(?:\s+(?:page|article|text|content|story|post|whole thing|it|this|selection|out loud|aloud|to me))*$", re.I)
TOOL_FIND_RE = re.compile(r"^(?:find|look for|locate|highlight|search (?:this|the) page for|where (?:does it say|is the (?:word|phrase|text)))\s+"
                          r"(?:the\s+)?(?:(?:word|phrase|text|words)\s+)?(.+?)(?:\s+(?:on|in) (?:this|the) page)?$", re.I)
TOOL_ZOOM_RE = re.compile(r"^(?:.*\bzoom\b.*|(?:make|set)?\s*(?:the\s+)?(?:text|font|page|everything|it)\s+(?:a bit\s+|much\s+)?(?:bigger|larger|smaller|normal)|"
                          r"(?:bigger|larger|smaller)\s+(?:text|font|page)|(?:increase|decrease|enlarge|reduce)\s+(?:the\s+)?(?:text|font)(?:\s+size)?)$", re.I)
TOOL_WHERE_RE = re.compile(r"^(?:where am i|what (?:page|site|website|tab) (?:is this|am i on|is open)|what's this (?:page|site)|which (?:page|site|tab) is this|"
                           r"what(?:'s| is) the (?:title|address|url)(?: of this page)?)$", re.I)
TOOL_HEADINGS_RE = re.compile(r"^(?:what(?:'s| is) on (?:this|the) page|what are the (?:headings|sections|links|options)|list (?:the )?(?:headings|sections|links|options)|"
                              r"what can i click(?: on)?|what(?:'s| is) here|what (?:sections|headings) are there|describe (?:this|the) page)$", re.I)
SLIDE_NOUN = r"(?:slides?|slide ?show|slide deck|deck|presentation|pdf|talk)"
SLIDES_OPEN_RE = re.compile(
    rf"^(?:(?:open|show|start|begin|launch|load|bring up|pull up|go to|switch to|back to|show me)(?: up)?(?: my| the| our)?(?: [a-z]+){{0,2}}? {SLIDE_NOUN}"
    rf"|(?:open|load) (?:the |my )?{SLIDE_NOUN}(?: from| at)? (?P<path>[~/].+\.pdf))$", re.I)
SLIDES_EXIT_RE = re.compile(rf"^(?:close|exit|leave|quit|stop|end)(?: the| my)? {SLIDE_NOUN}$", re.I)
SLIDES_NEXT_RE = re.compile(
    r"^(?:(?:go |move |skip )?(?:to |on to |onto )?(?:the )?next(?: slide| page| one)?|advance|forward|go forward|continue|go on|move on|keep going|"
    r"next slide please|scroll down|page down|down|onwards?)$", re.I)
SLIDES_PREV_RE = re.compile(
    r"^(?:(?:go |move )?(?:back )?(?:to )?(?:the )?(?:previous|prior|preceding)(?: slide| page| one)?|last slide|back one|one back|"
    r"(?:go )?back(?: a| one)?(?: slide| page)?|scroll up|page up|up|rewind)$", re.I)
SLIDES_FIRST_RE = re.compile(
    r"^(?:(?:go |jump |back )?(?:to )?(?:the )?(?:first slide|first one|beginning|start|top)(?: of the (?:deck|slides|talk))?|start over|from the top|restart)$", re.I)
SLIDES_END_RE = re.compile(
    r"^(?:(?:go |jump |skip )?(?:to )?(?:the )?(?:final slide|last one|end|final one|closing slide)(?: of the (?:deck|slides|talk))?|"
    r"go to the last slide|jump to the last slide|skip to the end)$", re.I)
SLIDES_GOTO_RE = re.compile(r"^(?:(?:go|jump|skip|move) (?:to |on to )?|show (?:me )?|open )?(?:the )?slide (?:number )?(?P<n>[a-z0-9]+)$", re.I)
SLIDES_WHICH_RE = re.compile(r"^(?:what|which) slide (?:am i on|is this|are we on)|what slide|how many slides(?: are there| in the deck)?$", re.I)
VIDEO_NOUN = r"(?:the |this |that )?(?:video|movie|clip|playback|player|music|song|it)"
VIDEO_NOUN_STRICT = r"(?:the |this |that )?(?:video|movie|clip|playback|player|music|song)"
TOOL_VIDEO_PAUSE_RE = re.compile(rf"^(?:pause|pause {VIDEO_NOUN}|(?:stop|halt|freeze) {VIDEO_NOUN_STRICT}|stop playing|hold {VIDEO_NOUN_STRICT})$", re.I)
TOOL_VIDEO_PLAY_RE = re.compile(rf"^(?:play|resume|unpause|(?:play|resume|unpause|start|continue) {VIDEO_NOUN}|keep playing|continue playing|"
                                rf"(?:start|play) (?:it )?again|press play)$", re.I)
CAPTION_NOUN = r"(?:the )?(?:captions?|subtitles?|cc|closed captions?|captioning)"
TOOL_VIDEO_CC_RE = re.compile(
    rf"^(?:(?:turn|switch|put) (?:on|off) {CAPTION_NOUN}|(?:turn|switch) {CAPTION_NOUN} (?:on|off)|(?:enable|disable|show|hide|toggle|add|remove) {CAPTION_NOUN}|"
    rf"{CAPTION_NOUN}(?: (?:on|off|please))?|(?:can i|could you|i want|i need) (?:have |get |see )?{CAPTION_NOUN})$", re.I)
SPEED_WORD = r"(?P<n>\d+(?:\.\d+)?|(?:\d+|one|two|zero) ?point ?(?:\d+|five|seven five|two five|twenty five|seventy five)|one and a half|two|one|half|double|normal|regular|default)"
SPEED_SUFFIX = r"(?: ?x| times|x speed| times speed| speed| times the speed)"
TOOL_VIDEO_SPEED_RES = [  # "speed" named, number optional suffix / number with a speed suffix / relative
    re.compile(rf"^(?:set |change |make |put )?(?:the )?(?:playback |video |play )?speed (?:to |at |back to |up to |down to )?{SPEED_WORD}{SPEED_SUFFIX}?$", re.I),
    re.compile(rf"^(?:(?:play|watch|go|run|set|put) (?:it |this |the video )?)?(?:at |on |in |to )?{SPEED_WORD}{SPEED_SUFFIX}$", re.I),
    re.compile(r"^(?:(?P<up>speed (?:it |the video |this )?up|faster|(?:play |go |make it |a bit |a little )?(?:it )?faster|quicker|speed up)|"
               r"(?P<down>slow (?:it |the video |this )?down|slower|(?:play |go |make it |a bit |a little )?(?:it )?slower|slow down)|"
               r"(?P<reset>(?:normal|regular|default) speed|reset (?:the )?speed|(?:back to |at )?(?:normal|regular) speed))$", re.I),
]
SPEED_WORDS = {"one and a half": 1.5, "two": 2.0, "one": 1.0, "half": 0.5, "double": 2.0, "normal": 1.0, "regular": 1.0, "default": 1.0}


def parse_speed(t: str) -> float | str | None:
    """'1.5x', 'speed one point five', 'double speed', 'faster' -> rate, 'up'/'down', or None."""
    for rx in TOOL_VIDEO_SPEED_RES:
        m = rx.match(t)
        if not m:
            continue
        gd = m.groupdict()
        if gd.get("up") is not None:
            return "up"
        if gd.get("down") is not None:
            return "down"
        if gd.get("reset") is not None:
            return 1.0
        n = (gd.get("n") or "").lower()
        if n in SPEED_WORDS:
            return SPEED_WORDS[n]
        for a, b in (("one", "1"), ("two", "2"), ("zero", "0"), ("seven five", "75"), ("seventy five", "75"), ("two five", "25"),
                     ("twenty five", "25"), ("five", "5"), (" point ", "."), ("point", ".")):
            n = n.replace(a, b)
        try:
            return float(n)
        except ValueError:
            return None
    return None


TOOL_VIDEO_MUTE_RE = re.compile(rf"^(?:(?P<un>un)?mute {VIDEO_NOUN}|(?:turn|switch) (?:the )?(?:sound|audio|volume) (?P<off>off|on)|(?:sound|audio) (?P<off2>off|on)|unmute|mute)$", re.I)
TOOL_VIDEO_SEEK_RE = re.compile(
    r"^(?:(?P<fwd>skip|jump|fast forward|forward|go forward|skip ahead|skip forward|jump ahead|jump forward|ahead)|(?P<back>rewind|go back|back|jump back|skip back|back up))"
    r"(?: (?:the |this )?(?:video|it))?(?: by| about)? (?P<n>\d+|ten|five|thirty|fifteen|twenty|sixty|a minute|one minute|two minutes)(?: ?(?:seconds?|secs?|s|minutes?|mins?))?"
    r"(?: (?:in|of|on) (?:the |this )?(?:video|clip))?$", re.I)
VIDEO_JS = r"""(op) => {
  const arg = op.arg;
  const vis = v => v.getClientRects().length > 0;
  let vids = [...document.querySelectorAll("video")].filter(vis);
  if (!vids.length) vids = [...document.querySelectorAll("video")];
  vids.sort((a, b) => (+!a.paused - +!b.paused) * 1e9 || (b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight));
  const v = vids.find(x => !x.paused) || vids[0];
  const btn = rx => [...document.querySelectorAll("button,[role=button]")].find(b => vis(b) && rx.test((b.getAttribute("aria-label") || b.title || b.textContent || "").trim()));
  const words = {"one and a half": 1.5, two: 2, one: 1, half: 0.5, double: 2, normal: 1, regular: 1, default: 1};
  if (op.kind === "cc") {
    const b = btn(/subtitles|captions|closed caption|\bcc\b/i);
    if (b) {
      const label = (b.getAttribute("aria-label") || b.title || "");
      if (b.disabled || b.getAttribute("aria-disabled") === "true" || /unavailable|not available/i.test(label)) return {ok: false, video: !!v, reason: "no captions"};
      const on = () => b.getAttribute("aria-pressed") === "true" || /\bon\b|hide|disable/i.test(b.getAttribute("aria-label") || "");
      const want = arg === "toggle" ? !on() : arg === "on";
      if (on() !== want) b.click();
      return {ok: true, video: !!v, cc: want, via: "button"};
    }
    if (!v) return {ok: false, video: false};
    const tracks = [...v.textTracks];
    if (!tracks.length) return {ok: false, video: true, reason: "no captions"};
    const showing = tracks.some(t => t.mode === "showing");
    const want = arg === "toggle" ? !showing : arg === "on";
    tracks.forEach((t, i) => t.mode = want && i === 0 ? "showing" : "disabled");
    return {ok: true, video: true, cc: want, via: "tracks"};
  }
  if (!v) return {ok: false, video: false};
  if (op.kind === "pause") { const was = !v.paused; v.pause(); return {ok: true, video: true, paused: v.paused, wasPlaying: was}; }
  if (op.kind === "play") { const p = v.play(); if (p && p.catch) p.catch(() => {}); return {ok: true, video: true, paused: false}; }
  if (op.kind === "mute") { v.muted = !!arg; return {ok: true, video: true, muted: v.muted}; }
  if (op.kind === "seek") { v.currentTime = Math.max(0, Math.min((v.duration || 1e9), v.currentTime + arg)); return {ok: true, video: true, t: v.currentTime}; }
  if (op.kind === "speed") {
    let r = typeof arg === "number" ? arg : arg === "up" ? v.playbackRate + 0.25 : arg === "down" ? v.playbackRate - 0.25 : (words[arg] ?? parseFloat(arg));
    if (!isFinite(r)) return {ok: false, video: true, reason: "bad speed"};
    r = Math.min(4, Math.max(0.25, Math.round(r * 100) / 100));
    v.playbackRate = r;
    return {ok: true, video: true, rate: v.playbackRate};
  }
  return {ok: false, video: true};
}"""
TOOL_HELP_RE = re.compile(r"^(?:help|help me|what can you do|what can i say|what do you do|how does this work|what are the commands)$", re.I)
TOOL_QUIET_RE = re.compile(r"^(?:be quiet|quiet|shush|hush|shut up|stop talking|stop reading|stop speaking|mute|silence|enough)$", re.I)
TOOL_WAIT_RE = re.compile(r"^(?:wait|hold on|hang on|hold|sleep|wait for|pause for)(?:\s+for)?(?:\s+(?:a|an|about))?"
                          r"\s*(\d+(?:\.\d+)?|[a-z]+)?\s*(seconds?|secs?|s|minutes?|mins?|moment|bit)?$", re.I)
TAB_STEP_RE = re.compile(r"^(?:(?:switch|change|swap|flip|move|jump|go)(?: over)?(?: (?:to|back to))?(?: the)?(?: [a-z]+)? tabs?"
                         r"|(?:next|previous|other) tab|(?:close|open)(?: up)?(?: a| an| the| this| that| another)? (?:new )?tabs?)$", re.I)
HELP_TEXT = ("I can open sites, search, click links and buttons, type, scroll, go back, and manage tabs. "
             "Ask me to read the page, summarize it, or answer a question about it. Say find, then a word, to highlight it. "
             "Say zoom in or zoom out. Teach me phrases: when I say yeet this tab, close the tab. Say stop at any time. "
             "On a video, say pause, play, turn on captions, speed one point five, or skip ahead ten seconds. "
             "With a PDF loaded, say open my slides, then next, last slide, or slide five.")
READ_CHARS = 6000
HEADINGS_JS = r"""() => [...document.querySelectorAll('h1, h2, h3, [role=heading]')]
  .filter(h => h.offsetParent !== null && h.closest('von-assistant') === null)
  .map(h => h.innerText.replace(/\s+/g, ' ').trim()).filter(t => t && t.length < 90)
  .filter((t, i, a) => a.indexOf(t) === i).slice(0, 12)"""
# what the assistant says when a finished request cannot be carried out
NOT_FOUND = {
    "no matching element on this page": "I don't see anything like that on this page. Name a link or button, or ask what's on this page.",
    "where to? (no site or domain recognised)": "Where should I go? Say a site name or a web address.",
    "search for what?": "What should I search for?",
    "type what?": "What should I type?",
    "no recognizable command yet": "I didn't catch a command in that.",
    "intent not confident yet": "I'm not sure what to do with that. Try go to, search for, click, or scroll.",
    "waiting for the rest of the command": "That sounded unfinished. What should I do?",
}


# --------------------------------------------------------------------------------------
# Policy: pure code gates
# --------------------------------------------------------------------------------------
def _r2(x: float) -> float:
    return round(x * 100) / 100


def _check(reasons: list, name: str, value: Any, threshold: Any, ok: bool, note: str) -> bool:
    reasons.append({"name": name, "value": _r2(value) if isinstance(value, float) else value, "threshold": threshold,
                    "pass": ok, "note": note})
    return ok


def top_choices(ans: dict | None, n: int = 3) -> list[dict]:
    if not ans or not ans.get("probabilities"):
        return []
    items = [(k, v) for k, v in ans["probabilities"].items() if k != "none"]
    items.sort(key=lambda kv: -kv[1])
    return [{"id": k, "p": _r2(v)} for k, v in items[:n]]


def _pick_span(ans: dict | None, min_conf: float, fallback: str | None) -> str | None:
    if not ans:
        return fallback
    if ans.get("choice") == "none":
        return None
    if ans.get("confidence", 0) < min_conf:
        return fallback if fallback is not None else ans.get("choice")
    return ans.get("choice")


def _fill(tpl: str, q: str) -> str:
    return tpl.replace("%s", quote(q, safe=""))


def describe(action: dict | None) -> str:
    if not action:
        return ""
    t = action.get("type")
    if t == "navigate_url":
        return f"open {(action.get('label') or '').replace('_', ' ') or action.get('url')}"
    if t == "type_into_field":
        return f"type \"{action.get('text')}\" into {action.get('label') or action.get('targetId')}" + (" + enter" if action.get("submit") else "")
    if t == "click_element":
        return f"click {action.get('label') or action.get('targetId')}"
    if t == "select_option":
        return f"select \"{action.get('text')}\" in {action.get('label') or action.get('targetId')}"
    return action.get("label") or str(t).replace("_", " ")


def evaluate_policy(answers: dict, candidates: dict, snapshot: dict, silent_ms: int = 0, is_final: bool = False,
                    pending: dict | None = None) -> dict:
    reasons: list[dict] = []
    intent = answers.get("intent") or {}
    name = intent.get("choice", "none")
    conf = intent.get("confidence", 0.0)

    if pending:
        if name == "confirm" and conf >= T["intentConfidence"]:
            _check(reasons, "intent", f"confirm ({_r2(conf)})", T["intentConfidence"], True, "pending action confirmed")
            return {"decision": "act", "action": {**pending, "confirmed": True}, "reasons": reasons, "summary": f"confirmed: {describe(pending)}"}
        if name == "cancel" and conf >= T["intentConfidence"]:
            _check(reasons, "intent", f"cancel ({_r2(conf)})", T["intentConfidence"], True, "pending action cancelled")
            return {"decision": "cancel", "reasons": reasons, "summary": "cancelled pending action"}

    is_cmd = (answers.get("is_command") or {}).get("noul", 0.0)
    if not _check(reasons, "is_command", is_cmd, T["isCommand"], is_cmd >= T["isCommand"], "user is addressing the browser"):
        return {"decision": "ignore", "reasons": reasons, "summary": "not a browser command"}

    intent_ok = name != "none" and conf >= T["intentConfidence"]
    _check(reasons, "intent", f"{name} ({_r2(conf)})", T["intentConfidence"], intent_ok, "confident, non-none intent")
    if not intent_ok:
        if name == "none" and is_final:
            return {"decision": "ignore", "reasons": reasons, "summary": "finished utterance with no browser command"}
        return {"decision": "wait", "reasons": reasons,
                "summary": "no recognizable command yet" if name == "none" else "intent not confident yet"}

    complete = (answers.get("complete") or {}).get("noul", 0.0)
    silent = silent_ms >= SILENCE_COMPLETE_MS or is_final
    complete_ok = complete >= T["complete"] or silent
    _check(reasons, "complete", complete, T["complete"], complete_ok,
           ("recognizer marked utterance final" if is_final else f"silent for {silent_ms}ms") if silent else "command has verb + object")
    if not complete_ok:
        return {"decision": "wait", "reasons": reasons, "summary": "waiting for the rest of the command"}

    if name in PAYLOAD_INTENTS:
        payload_ok = is_final or silent_ms >= PAYLOAD_SILENCE_MS
        _check(reasons, "payload_final", "final" if is_final else f"{silent_ms}ms silence", f"final or {PAYLOAD_SILENCE_MS}ms",
               payload_ok, "free text must be finished before it is copied")
        if not payload_ok:
            return {"decision": "wait", "reasons": reasons, "summary": "waiting for the end of the phrase (free text)",
                    "retryInMs": max(50, PAYLOAD_SILENCE_MS - silent_ms)}

    built = _build_action(name, answers, candidates, snapshot, reasons,
                          payload_final=is_final or silent_ms >= PAYLOAD_SILENCE_MS)
    if built["decision"] != "act":
        return {**built, "reasons": reasons}
    action = built["action"]

    destructive = (answers.get("destructive") or {}).get("noul", 0.0)
    if action["type"] in ("click_element", "press_enter", "select_option"):
        safe = destructive < T["destructive"]
        _check(reasons, "destructive", destructive, T["destructive"], safe, "reversible action" if safe else "needs spoken confirmation")
        if not safe:
            return {"decision": "confirm", "action": action, "reasons": reasons, "summary": f'say "confirm" to {describe(action)}'}
    return {"decision": "act", "action": action, "reasons": reasons, "summary": describe(action)}


def _build_action(name: str, answers: dict, candidates: dict, snapshot: dict, reasons: list,
                  payload_final: bool = True) -> dict:
    site = (answers.get("site") or {}).get("choice", "none")
    elements = (snapshot or {}).get("elements") or []
    text_c = candidates.get("text") or []
    url_c = candidates.get("url") or []

    if name == "navigate_url":
        url_pick = _pick_span(answers.get("url_span"), T["spanConfidence"], url_c[0] if url_c else None)
        if url_pick:
            _check(reasons, "url_span", url_pick, T["spanConfidence"], True, "domain spoken verbatim")
            return {"decision": "act", "action": {"type": "navigate_url", "url": to_http_url(url_pick), "label": url_pick}}
        if site in SITE_HOME:
            _check(reasons, "site", f"{site} ({_r2(answers['site'].get('confidence', 0))})", "-", True, "known site")
            return {"decision": "act", "action": {"type": "navigate_url", "url": SITE_HOME[site], "label": site}}
        spoken = _strip_filler(clean_transcript(candidates.get("transcript") or ""))
        m = DEST_RE.match(spoken)
        dest = m.group(1).strip() if m else ""
        # an unknown name only becomes a destination when it is framed as one ("go to word counter",
        # "word counter website"); a stray word on its own ("zap") is not a place to go
        framed = bool(NAV_VERB_RE.match(spoken)) or bool(re.search(r"\b(website|site|homepage|home page|dot com|\.com)\b", spoken, re.I))
        if payload_final and dest and framed and len(dest.split()) <= 6 and _tokens(dest) - {"go", "open", "website", "site", "page"}:
            _check(reasons, "site", dest, "-", True, "unknown site name: open the search engine's first hit")
            return {"decision": "act", "action": {"type": "navigate_url", "url": _fill(FIRST_HIT_URL, dest), "label": dest}}
        _check(reasons, "site", site, "known site or spoken domain", False, "no destination yet")
        return {"decision": "wait", "summary": "where to? (no site or domain recognised)"}

    if name == "search_web":
        query = _pick_span(answers.get("text_span"), T["spanConfidence"], text_c[0] if text_c else None)
        if not query:
            _check(reasons, "text_span", "none", T["spanConfidence"], False, "no query text yet")
            return {"decision": "wait", "summary": "search for what?"}
        _check(reasons, "text_span", query, T["spanConfidence"], True, "query copied verbatim")
        if site in SITE_SEARCH:
            return {"decision": "act", "action": {"type": "navigate_url", "url": _fill(SITE_SEARCH[site], query),
                                                  "label": f"search {site}: {query}", "query": query}}
        if snapshot.get("searchBoxId") and snapshot.get("site") != "blank":
            return {"decision": "act", "action": {"type": "type_into_field", "targetId": snapshot["searchBoxId"], "text": query,
                                                  "submit": True, "label": f"search this site: {query}"}}
        if snapshot.get("site") in SITE_SEARCH:  # on a known site whose search UI is not a plain input
            return {"decision": "act", "action": {"type": "navigate_url", "url": _fill(SITE_SEARCH[snapshot["site"]], query),
                                                  "label": f"search {snapshot['site']}: {query}", "query": query}}
        return {"decision": "act", "action": {"type": "navigate_url", "url": _fill(SITE_SEARCH[DEFAULT_SEARCH_ENGINE], query),
                                              "label": f"search: {query}", "query": query}}

    if name in TARGET_INTENTS:
        target = answers.get("target") or {}
        top = top_choices(target, T["candidateCount"])
        chosen = target.get("choice")
        target_ok = bool(chosen) and chosen != "none" and target.get("confidence", 0) >= T["targetConfidence"] \
            and (target.get("probabilities", {}).get(chosen, 0) >= T["targetTopProb"])
        text = None
        if name != "click_element":
            text = _pick_span(answers.get("text_span"), T["spanConfidence"], text_c[0] if text_c else None)
            if text:
                text = normalize_spoken_email(text) if "@" in normalize_spoken_email(text) else text
            if not text:
                _check(reasons, "text_span", "none", T["spanConfidence"], False, "no text to type yet")
                return {"decision": "wait", "summary": "type what?"}
        if target_ok:
            _check(reasons, "target", f"{chosen} ({_r2(target.get('confidence', 0))})", T["targetConfidence"], True,
                   element_label(elements, chosen))
            return {"decision": "act", "action": {"type": name, "targetId": chosen, "text": text, "label": element_label(elements, chosen)}}
        if name == "type_into_field" and snapshot.get("searchBoxId"):
            _check(reasons, "target", f"{chosen} ({_r2(target.get('confidence', 0))})", T["targetConfidence"], False, "falling back to search box")
            return {"decision": "act", "action": {"type": name, "targetId": snapshot["searchBoxId"], "text": text, "label": "search box"}}
        _check(reasons, "target", f"{chosen or 'none'} ({_r2(target.get('confidence', 0))})", T["targetConfidence"], False, "ambiguous target")
        viable = [c for c in top if c["p"] >= 0.08]
        if len(viable) == 1:
            viable = [c for c in top if c["p"] >= 0.02][:2]
        if not viable:
            return {"decision": "wait", "summary": "no matching element on this page"}
        cands = [{**c, "label": element_label(elements, c["id"])} for c in viable]
        return {"decision": "disambiguate", "candidates": cands, "pendingIntent": {"type": name, "text": text},
                "summary": "which one? " + " | ".join(f"{i + 1}: {c['label']}" for i, c in enumerate(cands))}

    if name in ("scroll_down", "scroll_up"):
        score = (answers.get("scroll_amount") or {}).get("score", 1.0)
        lvl = min(2, max(0, int(round(score))))
        amount = ["little", "page", "end"][lvl]
        _check(reasons, "scroll_amount", score, "round", True, amount)
        return {"decision": "act", "action": {"type": name, "amount": amount, "label": f"{name.replace('_', ' ')} ({amount})"}}

    if name == "switch_tab":
        d = (answers.get("tab_direction") or {}).get("choice") or "next"
        d = d if d != "none" else "next"
        return {"decision": "act", "action": {"type": "switch_tab", "direction": d, "label": f"switch tab ({d})"}}

    if name in ("confirm", "cancel"):
        return {"decision": "wait", "summary": f"nothing pending to {name}"}

    return {"decision": "act", "action": {"type": name, "label": name.replace("_", " ")}}


# --------------------------------------------------------------------------------------
# Browser: Playwright Chromium + overlay
# --------------------------------------------------------------------------------------
OVERLAY_JS = r"""
(() => {
  if (window.top !== window || window.__vb) return;
  const Z = 2147483000;
  const send = (m) => { try { window.__vbSend(JSON.stringify(m)); } catch (e) {} };
  // ---- page-level marks (light DOM: they sit over real elements) --------------------------------
  const pageCss = `
    @font-face{font-family:Borzoi;font-weight:400;src:url(data:font/otf;base64,__BORZOI__) format("opentype")}
    .__vb-hl{position:absolute;border:2px solid #000;border-radius:6px;box-shadow:0 0 0 3px rgba(255,255,255,.9),0 0 0 4px rgba(0,0,0,.35);
      z-index:${Z};pointer-events:none;transition:opacity .3s}
    .__vb-badge{position:absolute;background:#000;color:#fff;font:600 13px/1 -apple-system,Segoe UI,Inter,sans-serif;
      padding:4px 8px;border-radius:999px;z-index:${Z};pointer-events:none;border:2px solid #fff}
    .__vb-cand{position:absolute;border:2px dashed #000;border-radius:6px;z-index:${Z};pointer-events:none;background:rgba(0,0,0,.04)}
    .__vb-find{background:#000;color:#fff;border-radius:2px;box-shadow:0 0 0 2px #000}`;
  const ensureStyle = () => { if (document.getElementById("__vb-style")) return; const s = document.createElement("style");
    s.id = "__vb-style"; s.textContent = pageCss; (document.head || document.documentElement).appendChild(s); };
  const byId = (id) => document.querySelector(`[data-vb-id="${id}"]`);
  const box = (el) => { const r = el.getBoundingClientRect(); return { top: r.top + window.scrollY, left: r.left + window.scrollX, width: r.width, height: r.height }; };
  // ---- the assistant (shadow DOM: immune to page CSS) ------------------------------------------
  const W = 300;  // docked panel width (px)
  const css = `
    @font-face{font-family:Borzoi;font-weight:400;src:url(data:font/otf;base64,__BORZOI__) format("opentype")}
    :host{all:initial}
    *{box-sizing:border-box}
    .wrap{position:fixed;right:0;top:0;height:100vh;width:${W}px;z-index:${Z};display:flex;contain:strict;
      font:13px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text",Inter,Segoe UI,sans-serif;color:#000;transition:transform .2s ease;
      --bg:#fff;--fg:#000;--line:#e6e6e6;--dim:#888;--soft:#f4f4f4}
    .dark .wrap{--bg:#000;--fg:#fff;--line:#2a2a2a;--dim:#8a8a8a;--soft:#141414;color:#fff}
    .glass{flex:1;display:flex;flex-direction:column;min-height:0;background:var(--bg);border-left:1px solid var(--line);padding:14px 14px 14px;position:relative;overflow:hidden}
    .head{display:flex;align-items:center;gap:4px;height:28px;flex:none;position:relative}
    .brand{font:400 26px/1 Borzoi,-apple-system,sans-serif;color:var(--fg);user-select:none;padding-top:2px;margin-right:auto;letter-spacing:0}
    .ic{width:26px;height:26px;border-radius:6px;border:0;background:transparent;cursor:pointer;color:var(--fg);opacity:.45;display:grid;place-items:center;flex:none}
    .ic:hover{opacity:1;background:var(--soft)}
    .ic svg{width:15px;height:15px;fill:currentColor}
    .pill{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--dim);min-height:18px;margin:8px 0 4px;flex:none;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
    .pill:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--line);flex:none;transition:background .2s}
    .pill.listening:before{background:var(--fg);animation:blink 1.6s ease-in-out infinite}
    .pill.thinking:before,.pill.working:before{background:var(--fg);animation:blink .7s ease-in-out infinite}
    .pill.speaking:before{background:var(--fg)}
    .pill.error:before{background:var(--fg)} .pill.error{color:var(--fg)}
    @keyframes blink{50%{opacity:.2}}
    .log{flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;gap:6px;padding:4px 0}
    .u{font-size:12.5px;color:var(--dim);padding:0 2px;align-self:flex-end;max-width:92%;text-align:right;word-wrap:break-word}
    .u.live{opacity:.6}
    .a{font-size:13px;line-height:1.45;padding:7px 10px;border-radius:10px;background:var(--soft);white-space:pre-wrap;word-wrap:break-word;max-width:96%;align-self:flex-start}
    .a.error{background:transparent;border:1px solid var(--fg)} .a.ok{background:transparent;color:var(--dim);font-size:12.5px;padding:2px 2px}
    .sug{flex:none;display:flex;flex-direction:column;gap:5px;padding:6px 0 2px}
    .sug:empty{display:none}
    .sug .t{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);padding:0 2px}
    .sug button{text-align:left;font:inherit;font-size:12.5px;color:var(--fg);background:transparent;border:1px solid var(--line);border-radius:8px;padding:5px 9px;cursor:pointer;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .sug button:hover{border-color:var(--fg)}
    .chips{display:flex;flex-wrap:wrap;gap:5px;flex:none;padding-top:4px}
    .chips:empty{display:none}
    .chip{border:1px solid var(--fg);border-radius:999px;padding:2px 9px;font-size:12px;cursor:pointer;display:flex;gap:6px;align-items:center;max-width:100%;background:var(--bg);color:var(--fg)}
    .chip b{font-weight:600}
    .chip.warn{background:var(--fg);color:var(--bg)}
    .chip span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .chip.guide{cursor:default;border-style:dashed}
    .chip.guide em{font-style:normal;text-decoration:underline;cursor:pointer;margin-left:2px;flex:none}
    .foot{flex:none;display:flex;flex-direction:row;align-items:center;gap:8px;padding-top:10px}
    .orb{width:40px;height:40px;border-radius:50%;flex:none;border:1.5px solid var(--fg);cursor:pointer;position:relative;padding:0;background:var(--fg);
      transition:transform .15s;will-change:transform}
    .orb:hover{transform:scale(1.05)} .orb:active{transform:scale(.96)}
    .orb:focus-visible{outline:2px solid var(--fg);outline-offset:3px}
    .orb.off{background:var(--bg)} .orb.off svg{fill:var(--fg)}
    .orb .ring{position:absolute;inset:-6px;border-radius:50%;border:1.5px solid var(--fg);opacity:0;transform:scale(.8);pointer-events:none;will-change:transform,opacity}
    .orb.listening .ring{animation:pulse 1.6s ease-out infinite}
    .orb.speaking .ring{opacity:.6;transform:scale(1)}
    .orb.thinking .ring,.orb.working .ring{opacity:.6;transform:scale(1);border-style:dashed;animation:spin 1.4s linear infinite}
    .orb .lvl{position:absolute;inset:0;border-radius:50%;background:rgba(255,255,255,.35);transform:scale(0);transition:transform .08s;pointer-events:none}
    .dark .orb .lvl{background:rgba(0,0,0,.35)}
    .orb svg{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:18px;height:18px;fill:var(--bg);pointer-events:none}
    @keyframes pulse{0%{opacity:.8;transform:scale(.85)}100%{opacity:0;transform:scale(1.45)}}
    @keyframes spin{to{transform:scale(1) rotate(360deg)}}
    input{flex:1;min-width:0;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:9px 12px;font:inherit;font-size:13.5px;color:var(--fg);outline:none}
    input:focus{border-color:var(--fg)}
    input::placeholder{color:var(--dim)}
    .hint{font-size:11px;color:var(--dim);text-align:center;flex:none;padding-top:8px}
    .panel{display:none;position:absolute;inset:44px 14px auto 14px;max-height:60%;overflow:auto;padding:10px 12px;border-radius:10px;font-size:12px;z-index:2;
      background:var(--bg);border:1px solid var(--fg)}
    .panel.on{display:block}
    .panel h4{margin:6px 0 3px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);font-weight:500}
    .panel .l{display:flex;gap:8px;align-items:center;padding:2px 0;cursor:default}
    .panel .l[data-i]{cursor:pointer}
    .panel .l .x{margin-left:auto;cursor:pointer;opacity:.5} .panel .l .x:hover{opacity:1}
    .panel .l.act{font-weight:600} .panel .dim{color:var(--dim)}
    .mini.wrap{transform:translateX(${W}px)}
    .tab{position:fixed;right:14px;bottom:14px;z-index:${Z};width:44px;height:44px;border-radius:50%;border:1.5px solid #fff;cursor:pointer;display:none;place-items:center;background:#000}
    .tab svg{width:20px;height:20px;fill:#fff}
    .tab.on{display:grid}`;
  const MIC = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a4 4 0 0 0 4-4V6a4 4 0 1 0-8 0v5a4 4 0 0 0 4 4zm6-4h-2a4 4 0 0 1-8 0H6a6 6 0 0 0 5 5.91V20H8v2h8v-2h-3v-3.09A6 6 0 0 0 18 11z"/></svg>';
  const DOTS = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>';
  const CLOSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6-1.4-1.4L12.2 12 7.6 7.4z"/></svg>';
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const st = { got: false, voice: false, active: true, stt: null, tts: null, mini: false, panel: false, tabs: [], aliases: [], cands: [], pending: null, sug: [], mode: "idle" };
  let host = null, R = null, E = {};
  const $ = (c) => R.querySelector(c);

  function dockPage(on) {  // the page keeps its full layout in the space left of the panel
    const de = document.documentElement;
    if (on) { de.style.setProperty("width", `calc(100% - ${W / (st.zoom || 1)}px)`, "important"); de.style.setProperty("min-height", "100vh"); }
    else { de.style.removeProperty("width"); de.style.removeProperty("min-height"); } }
  function mount() {
    if (host && host.isConnected) return;
    if (!document.body) { document.addEventListener("DOMContentLoaded", mount, { once: true }); return; }
    ensureStyle();  // @font-face only registers in the document scope, never inside a shadow root
    host = document.createElement("whippet-assistant");
    host.setAttribute("role", "complementary"); host.setAttribute("aria-label", "Whippet assistant");
    R = host.attachShadow({ mode: "open" });
    R.innerHTML = `<style>${css}</style><div class="wrap" id="wrap"><div class="glass">
      <div class="head"><span class="brand" aria-label="Whippet">whippet</span>
        <button class="ic" id="more" title="tabs, phrases, engine" aria-label="details">${DOTS}</button><button class="ic" id="mini" title="hide panel (Alt+Space)" aria-label="hide panel">${CLOSE}</button></div>
      <div class="pill" id="pill"><span id="pillt">ready</span></div>
      <div class="log" id="log" aria-live="polite"></div>
      <div class="sug" id="sug"></div>
      <div class="chips" id="chips"></div>
      <div class="foot">
        <button class="orb off" id="orb" title="voice mode (Alt+V)" aria-label="voice mode" aria-pressed="false"><span class="ring"></span><span class="lvl"></span>${MIC}</button>
        <input id="cmd" type="text" autocomplete="off" spellcheck="false" aria-label="command" placeholder="Type a command…">
      </div>
      <div class="hint">Alt+V voice · Alt+P hold-Space-to-talk · Alt+K type · Alt+Space hide</div><div class="panel" id="panel"></div></div></div>
      <button class="tab" id="tab" title="show Whippet (Alt+Space)" aria-label="show Whippet">${MIC}</button>`;
    document.documentElement.appendChild(host);
    E = { wrap: $("#wrap"), orb: $("#orb"), pill: $("#pill"), pillt: $("#pillt"), log: $("#log"), cmd: $("#cmd"), sug: $("#sug"), chips: $("#chips"), panel: $("#panel"), lvl: $(".lvl"), tab: $("#tab") };
    E.orb.onclick = () => send({ type: "voice", on: !st.voice });
    E.tab.onclick = () => setMini(false);
    E.cmd.onkeydown = (e) => { if (e.key === "Enter" && E.cmd.value.trim()) { send({ type: "command", text: E.cmd.value.trim() }); heard(E.cmd.value.trim(), true); E.cmd.value = ""; }
      if (e.key === "Escape") { E.cmd.blur(); } e.stopPropagation(); };
    E.cmd.onkeyup = E.cmd.onkeypress = (e) => e.stopPropagation();
    E.cmd.onfocus = () => resumeAudio();
    $("#more").onclick = () => { st.panel = !st.panel; E.panel.classList.toggle("on", st.panel); renderPanel(); };
    $("#mini").onclick = () => setMini(true);
    host.addEventListener("pointerdown", resumeAudio, { once: true });
    theme(); dockPage(true);
    send({ type: "hello" });
  }
  function setMini(on) { st.mini = on; E.wrap.classList.toggle("mini", on); E.tab.classList.toggle("on", on); dockPage(!on); }
  function theme() { try { let el = document.body, bg = "rgba(0, 0, 0, 0)"; while (el && /rgba\(0, 0, 0, 0\)|transparent/.test(bg)) { bg = getComputedStyle(el).backgroundColor; el = el.parentElement; }
    const m = bg.match(/\d+/g) || [255, 255, 255]; const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
    host.classList.toggle("dark", lum < 0.45 && !/rgba\(0, 0, 0, 0\)/.test(bg)); } catch (e) {} }
  const idleMode = () => (st.voice && st.active && !(st.ptt && !st.held) ? "listening" : "idle");
  const idleText = () => (st.voice && st.ptt ? (st.held ? "listening" : "hold Space to talk") : undefined);
  const LABEL = { idle: "ready", listening: "listening", thinking: "thinking", working: "working", speaking: "speaking", error: "" };
  function setMode(m, text) { st.mode = m; if (!E.orb) return;
    E.orb.className = "orb" + (st.voice ? "" : " off") + (m !== "idle" ? " " + m : ""); E.orb.setAttribute("aria-pressed", String(st.voice));
    E.pill.className = "pill " + m; E.pillt.textContent = text || LABEL[m] || m; }
  function status(text, mode) { setMode(mode || st.mode, text); }
  let liveU = null, liveA = null;
  function trim() { while (E.log.children.length > 40) E.log.firstChild.remove(); }
  function heard(text, final) { if (!E.log) return;
    if (final && !liveU) { const p = E.log.lastElementChild; if (p && p.className === "u" && p.textContent === text) return; }
    if (!liveU) { liveU = document.createElement("div"); E.log.appendChild(liveU); }
    liveU.className = "u" + (final ? "" : " live"); liveU.textContent = text + (final ? "" : " …");
    if (final) liveU = null; liveA = null; trim(); E.log.scrollTop = E.log.scrollHeight; }
  function say(text, kind, stream) { if (!E.log) return; if (!text) return;
    const prev = E.log.lastElementChild;  // the same line can arrive twice around a navigation (event + history replay)
    if (!stream && prev && prev.className.startsWith("a") && prev.textContent === text) return;
    if (!(stream && liveA)) { liveA = document.createElement("div"); E.log.appendChild(liveA); }
    liveA.className = "a" + (kind === "error" ? " error" : kind === "ok" ? " ok" : ""); liveA.textContent = text;
    if (!stream) liveA = null; trim(); E.log.scrollTop = E.log.scrollHeight; }
  function renderSug() { if (!E.sug) return; const list = st.sug || [];
    E.sug.innerHTML = list.length ? `<div class="t">Try saying</div>` + list.map((t) => `<button type="button">“${esc(t)}”</button>`).join("") : "";
    E.sug.querySelectorAll("button").forEach((b, i) => (b.onclick = () => { const t = list[i]; send({ type: "command", text: t }); heard(t, true); })); }
  function renderChips() { if (!E.chips) return; let h = "";
    if (st.guide && st.guide.step) h += `<div class="chip guide"><b>${st.guide.step}/${st.guide.total}</b><span>${esc(st.guide.text)}</span><em data-g="next">next</em><em data-g="skip">skip</em><em data-g="stop">stop</em></div>`;
    if (st.pending) h += `<div class="chip warn"><span>say “confirm” to ${esc(st.pending.label || st.pending.type)}</span></div>`;
    for (const c of st.cands) h += `<div class="chip" data-n="${c.n}"><b>${c.n}</b><span>${esc(c.label)}</span></div>`;
    E.chips.innerHTML = h; E.chips.querySelectorAll(".chip[data-n]").forEach((el) => (el.onclick = () => send({ type: "command", text: `number ${el.dataset.n}` })));
    E.chips.querySelectorAll("em[data-g]").forEach((el) => (el.onclick = () => { send({ type: "command", text: el.dataset.g }); heard(el.dataset.g, true); })); }
  function renderPanel() { if (!st.panel || !E.panel) return; const s = st.stt, t = st.tts, l = st.llm;
    const mod = (x, name) => x ? (x.ready ? `${name}: ${x.model.split("/").pop()}${x.avg_ms ? " · " + x.avg_ms + " ms" : ""}${x.avg_first_audio_ms ? " · first audio " + x.avg_first_audio_ms + " ms" : ""}` : `${name}: ${x.error ? "unavailable — " + esc(x.error) : "loading…"}`) : `${name}: off`;
    E.panel.innerHTML = `<h4>Tabs</h4>${st.tabs.map((t) => `<div class="l${t.active ? " act" : ""}" data-i="${t.index}">${t.index + 1}. ${esc((t.title || t.url || "blank").slice(0, 60))}</div>`).join("")}
      <h4>Your phrases</h4>${st.aliases.length ? st.aliases.map((a) => `<div class="l">“${esc(a.phrase)}” → ${esc(a.command)}<span class="x" data-p="${esc(a.phrase)}">✕</span></div>`).join("") : `<div class="l dim">say “when I say yeet this tab, close the tab”</div>`}
      <h4>Engine</h4><div class="l">${mod(s, "hearing")}</div><div class="l">${mod(t, "voice")}</div><div class="l">${mod(l, "reading")}</div>`;
    E.panel.querySelectorAll(".x").forEach((x) => (x.onclick = () => send({ type: "forget", phrase: x.dataset.p })));
    E.panel.querySelectorAll(".l[data-i]").forEach((x) => (x.onclick = () => { send({ type: "switch_tab", index: +x.dataset.i }); st.panel = false; E.panel.classList.remove("on"); })); }
  // ---- audio out (Pocket TTS PCM streamed from Python) ----------------------------------------
  let actx = null, nextT = 0, playing = 0, curGen = 0;
  function resumeAudio() { try { if (!actx) actx = new AudioContext({ sampleRate: 24000 }); if (actx.state === "suspended") actx.resume(); } catch (e) {} }
  function play(b64, rate, gen, last) { resumeAudio(); if (!actx) return;
    if (gen !== curGen) { stopAudio(); curGen = gen; }
    const bin = atob(b64), n = bin.length >> 1; if (!n) return;
    const buf = actx.createBuffer(1, n, rate), ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) { const v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); ch[i] = ((v << 16) >> 16) / 32768; }
    const src = actx.createBufferSource(); src.buffer = buf; src.connect(actx.destination);
    const t = Math.max(actx.currentTime + 0.01, nextT); src.start(t); nextT = t + buf.duration; playing++; setMode("speaking");
    src.onended = () => { playing--; if (playing <= 0) { playing = 0; setMode(idleMode()); send({ type: "spoken", gen }); } };
    sources.push(src); }
  let sources = [];
  function stopAudio() { for (const s of sources) { try { s.onended = null; s.stop(); } catch (e) {} } sources = []; playing = 0; nextT = 0; setMode(idleMode()); }
  // ---- audio in (mic PCM to Python's Whisper) or Web Speech fallback ------------------------
  let mic = null, micCtx = null, proc = null, rec = null, recOn = false, recBase = 0, recDead = false, recFails = 0, micDead = false, micOpening = false, pttTail = 0;
  function setHeld(on) { if (st.held === on) return; st.held = on; if (!on) pttTail = 5; setMode(idleMode(), idleText()); }
  async function startMic() { if (mic || micOpening || micDead) return; micOpening = true;
    try { mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } }); }
    catch (e) { micOpening = false; micDead = true; send({ type: "mic_error", text: e.message }); setMode("idle", "microphone unavailable: " + e.message); return; }
    micOpening = false; if (!st.voice) { mic.getTracks().forEach((t) => t.stop()); mic = null; return; }
    try { micCtx = new AudioContext({ sampleRate: 16000 }); } catch (e) { micCtx = new AudioContext(); }
    const src = micCtx.createMediaStreamSource(mic); proc = micCtx.createScriptProcessor(4096, 1, 1); const mute = micCtx.createGain(); mute.gain.value = 0;
    proc.onaudioprocess = (e) => { const d = e.inputBuffer.getChannelData(0); let s = 0; const u8 = new Uint8Array(d.length * 2);
      // push-to-talk: the mic stays open (no start-up latency) but only frames while Space is held reach Whisper,
      // followed by a short silent tail so the utterance is finalized as soon as the key is released
      if (st.ptt && !st.held) { if (pttTail <= 0) { if (E.lvl) E.lvl.style.transform = "scale(0)"; return; } pttTail--; send({ type: "audio", pcm: btoa(String.fromCharCode.apply(null, u8)), rate: micCtx.sampleRate }); return; }
      for (let i = 0; i < d.length; i++) { const v = Math.max(-1, Math.min(1, d[i])); s += v * v; const q = v < 0 ? v * 32768 : v * 32767; u8[2 * i] = q & 255; u8[2 * i + 1] = (q >> 8) & 255; }
      const lvl = Math.min(1, Math.sqrt(s / d.length) * 6); if (E.lvl) E.lvl.style.transform = `scale(${lvl.toFixed(2)})`;
      let bin = ""; for (let i = 0; i < u8.length; i += 8192) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
      send({ type: "audio", pcm: btoa(bin), rate: micCtx.sampleRate }); };
    src.connect(proc); proc.connect(mute); mute.connect(micCtx.destination); setMode("listening"); }
  function stopMic() { try { proc && proc.disconnect(); mic && mic.getTracks().forEach((t) => t.stop()); micCtx && micCtx.close(); } catch (e) {} mic = micCtx = proc = null; if (E.lvl) E.lvl.style.transform = "scale(0)"; }
  // Web Speech is only a fallback (no mlx-whisper). Playwright's Chromium has no Google speech keys, so it can fail
  // instantly and forever: give up after a fatal error instead of restarting in a tight loop.
  function startRec() { const SR = window.SpeechRecognition || window.webkitSpeechRecognition; if (!SR || recOn || recDead) { if (!SR || recDead) setMode("idle", "no speech recognition (install mlx-whisper)"); return; }
    rec = new SR(); rec.continuous = true; rec.interimResults = true; rec.lang = "en-US"; recBase++;
    rec.onresult = (ev) => { recFails = 0; for (let i = ev.resultIndex; i < ev.results.length; i++) { const r = ev.results[i]; send({ type: "transcript", text: r[0].transcript, final: r.isFinal, utteranceId: `w${recBase}-${i}` }); } };
    rec.onend = () => { if (!recOn || recDead) return; recBase++; setTimeout(() => { if (recOn && !recDead) { try { rec.start(); } catch (e) {} } }, Math.min(4000, 250 * 2 ** recFails)); };
    rec.onerror = (e) => { recFails++; const fatal = /not-allowed|service-not-allowed|audio-capture|language-not-supported/.test(e.error) || recFails >= 4;
      if (fatal) { recDead = true; recOn = false; setMode("idle", `speech recognition unavailable (${e.error}) — install mlx-whisper`); send({ type: "mic_error", text: "webspeech: " + e.error }); } };
    try { rec.start(); recOn = true; setMode("listening"); } catch (e) {} }
  function stopRec() { recOn = false; try { rec && rec.stop(); } catch (e) {} rec = null; }
  function applyVoice() { const want = st.voice && st.active && !document.hidden;
    if (want) { if (!st.got) { send({ type: "hello" }); return; }  // state not received yet
      if (st.stt && st.stt.ready) startMic(); else if (st.stt && !st.stt.error) setMode("idle", "warming up hearing…"); else startRec(); }
    else { stopMic(); stopRec(); setMode("idle"); } }
  document.addEventListener("visibilitychange", applyVoice);
  // ---- events from Python -------------------------------------------------------------------
  const H = {
    state(m) { st.got = true; Object.assign(st, { voice: !!m.voice, ptt: !!m.ptt, active: !!m.active, stt: m.stt, tts: m.tts, llm: m.llm, tabs: m.tabs || [], aliases: m.aliases || [], cands: m.candidates || [], pending: m.pending, sug: m.suggestions || [], guide: m.guide || null });
      applyVoice(); renderChips(); renderSug(); renderPanel();
      if (!E.log.children.length && m.history) { for (const h of m.history) { if (h.role === "u") heard(h.text, true); else say(h.text, h.kind); } E.log.scrollTop = E.log.scrollHeight; }
      setMode(idleMode(), idleText()); },
    voice(m) { st.voice = !!m.on; applyVoice(); setMode(idleMode(), st.voice ? idleText() || "listening" : "voice off"); },
    ptt(m) { st.ptt = !!m.on; st.held = false; setMode(idleMode(), st.ptt ? "hold Space to talk" : (st.voice ? "listening" : "push-to-talk off")); },
    transcript(m) { heard(m.text, !!m.final); },
    gate(m) {},
    decision(m) { if (m.decision === "act") setMode("working", m.summary || "working"); else if (m.decision === "confirm") status(`confirm? ${m.summary}`, "thinking"); },
    action(m) { if (m.result && m.result.ok) { say(m.action.label || m.action.type, "ok"); setMode(st.mode === "speaking" ? "speaking" : idleMode()); }
      else status(`couldn't ${m.action.label || m.action.type}`, "error"); },
    say(m) { say(m.text, m.kind); if (m.kind === "error") status(m.text.length > 60 ? "couldn't do that" : m.text, "error"); },
    answer(m) { say(m.text || "…", "answer", true); if (!m.done) setMode("thinking", "reading the page…"); else { liveA = null; setMode(idleMode()); } },
    candidates(m) { st.cands = m.candidates || []; renderChips(); },
    suggestions(m) { st.sug = m.items || []; renderSug(); },
    pending(m) { st.pending = m.pending; renderChips(); },
    guide(m) { st.guide = m.step ? m : null; renderChips(); },
    aliases(m) { st.aliases = m.aliases || []; renderPanel(); },
    tabs(m) { st.tabs = m.tabs || []; renderPanel(); },
    audio(m) { play(m.pcm, m.rate, m.gen, m.last); },
    audio_stop() { stopAudio(); },
    toast(m) { status(m.text); },
    log() {}, snapshot() {}, steps() {},
    highlight(m) { api.highlight(m.id, m.ms); },
    marks(m) { api.candidates(m.items, m.ms); },
    clear_marks() { api.clearCandidates(); },
    find(m) { api.find(m.text); },
    zoom(m) { st.zoom = m.level; document.documentElement.style.zoom = m.level; if (host) host.style.zoom = 1 / m.level; dockPage(!st.mini); },
    focus() { setMini(false); E.cmd && E.cmd.focus(); },
  };
  const api = {
    event(m) { if (!host || !host.isConnected) mount(); const h = H[m.type]; if (h) { try { h(m); } catch (e) { send({ type: "log", text: "overlay " + m.type + ": " + e.message }); } } },
    toast(msg) { status(msg); },
    highlight(id, ms = 600) { ensureStyle(); const el = byId(id); if (!el) return false;
      el.scrollIntoView({ block: "center", inline: "nearest" }); const b = box(el); const h = document.createElement("div"); h.className = "__vb-hl";
      Object.assign(h.style, { top: b.top - 4 + "px", left: b.left - 4 + "px", width: b.width + 8 + "px", height: b.height + 8 + "px" });
      document.documentElement.appendChild(h); setTimeout(() => (h.style.opacity = "0"), ms); setTimeout(() => h.remove(), ms + 350); return true; },
    candidates(list, ms = 8000) { ensureStyle(); api.clearCandidates(); let first = true;
      for (const c of list) { const el = byId(c.id); if (!el) continue; if (first) { el.scrollIntoView({ block: "center" }); first = false; }
        const b = box(el); const f = document.createElement("div"); f.className = "__vb-cand __vb-c";
        Object.assign(f.style, { top: b.top - 3 + "px", left: b.left - 3 + "px", width: b.width + 6 + "px", height: b.height + 6 + "px" });
        const badge = document.createElement("div"); badge.className = "__vb-badge __vb-c"; badge.textContent = String(c.n);
        Object.assign(badge.style, { top: Math.max(0, b.top - 14) + "px", left: Math.max(0, b.left - 14) + "px" });
        document.documentElement.appendChild(f); document.documentElement.appendChild(badge); }
      api._candTimer = setTimeout(api.clearCandidates, ms); },
    clearCandidates() { clearTimeout(api._candTimer); document.querySelectorAll(".__vb-c").forEach((n) => n.remove()); },
    find(text) { ensureStyle(); document.querySelectorAll(".__vb-find").forEach((n) => { const p = n.parentNode; p.replaceChild(document.createTextNode(n.textContent), n); p.normalize(); });
      if (!text) return 0; const rx = new RegExp(text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"); let n = 0, first = null;
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, { acceptNode: (t) => (t.parentNode && !/^(SCRIPT|STYLE|NOSCRIPT|WHIPPET-ASSISTANT)$/.test(t.parentNode.nodeName) && rx.test(t.nodeValue) ? 1 : 2) });
      const hits = []; while (walker.nextNode()) hits.push(walker.currentNode);
      for (const t of hits.slice(0, 200)) { const m = rx.exec(t.nodeValue); if (!m) continue; const r = document.createRange(); r.setStart(t, m.index); r.setEnd(t, m.index + m[0].length);
        const mark = document.createElement("mark"); mark.className = "__vb-find"; try { r.surroundContents(mark); n++; if (!first) first = mark; } catch (e) {} }
      if (first) first.scrollIntoView({ block: "center" }); return n; },
    mount,
  };
  window.__vb = api;
  const editing = (e) => { const p = (e.composedPath && e.composedPath()[0]) || e.target; return !!(p && (p.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(p.tagName || ""))); };
  window.addEventListener("keydown", (e) => {
    if (!e.altKey) { if (st.ptt && st.voice && e.code === "Space" && !e.ctrlKey && !e.metaKey && !editing(e)) { e.preventDefault(); if (!e.repeat) setHeld(true); } return; }
    if (e.code === "Space") { e.preventDefault(); setMini(!st.mini); } else if (e.code === "KeyV") { e.preventDefault(); send({ type: "voice", on: !st.voice }); }
    else if (e.code === "KeyP") { e.preventDefault(); send({ type: "ptt", on: !st.ptt }); }
    else if (e.code === "KeyK") { e.preventDefault(); H.focus(); } }, true);
  window.addEventListener("keyup", (e) => { if (e.code === "Space" && st.held) { e.preventDefault(); setHeld(false); } }, true);
  window.addEventListener("blur", () => setHeld(false));
  mount();
})();
"""
# Borzoi-Regular.otf (the uploaded font, byte for byte) so the wordmark needs no installed font.
BORZOI_OTF_B64 = (
    "T1RUTwALAIAAAwAwQ0ZGIJ/3anAAAAbAAAHpE0dQT1M25C+PAAHv1AAAAV5HU1VCkA2VHwAB8TQAAABmT1MvMkpu6mwAAAEgAAAAYGNtYXDyau0oAAAF9AAA"
    "AKxoZWFkQBW3vwAAALwAAAA2aGhlYRekE8cAAAD0AAAAJGhtdHjbRWfVAAHxnAAABWRtYXhwAVlQAAAAARgAAAAGbmFtZVloPywAAAGAAAAEdHBvc3T/iwAw"
    "AAAGoAAAACAAAQAAAAEAAPYQ1nFfDzz1AAMEsAAAAADmxTmFAAAAAObFOYYANv7DEz4ESQAAAAMAAgAAAAAAAAABAAAD1P6sAFoTiAA2ADkTPgABAAAAAAAA"
    "AAAAAAAAAAABWQAAUAABWQAAAAMC3QGQAAUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAABwAAAEoAAAAAAAAAAD8/Pz8AQAAgImUD1P6s"
    "AFoEnAF8AAAAAAAAAAACWANcAAAAIAAGAAAAEgDeAAEAAAAAAAEABgAAAAEAAAAAAAIABwAGAAEAAAAAAAMAFAANAAEAAAAAAAQADgAhAAEAAAAAAAUADQAv"
    "AAEAAAAAAAYADgA8AAEAAAAAAAgAFQBKAAEAAAAAAAkAHwBfAAEAAAAAAAoAtAB+AAMAAQQJAAEADAEyAAMAAQQJAAIADgE+AAMAAQQJAAMAKAFMAAMAAQQJ"
    "AAQAHAF0AAMAAQQJAAUAGgGQAAMAAQQJAAYAHAGqAAMAAQQJAAgAKgHGAAMAAQQJAAkAPgHwAAMAAQQJAAoBaAIuQm9yem9pUmVndWxhckJvcnpvaS1SZWd1"
    "bGFyLTguMDAwQm9yem9pIFJlZ3VsYXJWZXJzaW9uIDguMDAwQm9yem9pLVJlZ3VsYXJDdXN0b20gcmVjb25zdHJ1Y3Rpb25DdXN0b20gZ2VvbWV0cmljIHJl"
    "Y29uc3RydWN0aW9uQm9yem9pIGdlb21ldHJpYyBkaXNwbGF5L3RleHQgZmFtaWx5LiBQcmVzZXJ2ZXMgdGhlIHN1cHBsaWVkIGJvcnpvaSB3b3JkbWFyayBp"
    "biBSZWd1bGFyIGFuZCBhZGRzIGEgYnJvYWQgTGF0aW4sIG51bWVyaWMsIHB1bmN0dWF0aW9uLCBjdXJyZW5jeSwgbWF0aCwgYW5kIHR5cG9ncmFwaGljIGNo"
    "YXJhY3RlciBzZXQuAEIAbwByAHoAbwBpAFIAZQBnAHUAbABhAHIAQgBvAHIAegBvAGkALQBSAGUAZwB1AGwAYQByAC0AOAAuADAAMAAwAEIAbwByAHoAbwBp"
    "ACAAUgBlAGcAdQBsAGEAcgBWAGUAcgBzAGkAbwBuACAAOAAuADAAMAAwAEIAbwByAHoAbwBpAC0AUgBlAGcAdQBsAGEAcgBDAHUAcwB0AG8AbQAgAHIAZQBj"
    "AG8AbgBzAHQAcgB1AGMAdABpAG8AbgBDAHUAcwB0AG8AbQAgAGcAZQBvAG0AZQB0AHIAaQBjACAAcgBlAGMAbwBuAHMAdAByAHUAYwB0AGkAbwBuAEIAbwBy"
    "AHoAbwBpACAAZwBlAG8AbQBlAHQAcgBpAGMAIABkAGkAcwBwAGwAYQB5AC8AdABlAHgAdAAgAGYAYQBtAGkAbAB5AC4AIABQAHIAZQBzAGUAcgB2AGUAcwAg"
    "AHQAaABlACAAcwB1AHAAcABsAGkAZQBkACAAYgBvAHIAegBvAGkAIAB3AG8AcgBkAG0AYQByAGsAIABpAG4AIABSAGUAZwB1AGwAYQByACAAYQBuAGQAIABh"
    "AGQAZABzACAAYQAgAGIAcgBvAGEAZAAgAEwAYQB0AGkAbgAsACAAbgB1AG0AZQByAGkAYwAsACAAcAB1AG4AYwB0AHUAYQB0AGkAbwBuACwAIABjAHUAcgBy"
    "AGUAbgBjAHkALAAgAG0AYQB0AGgALAAgAGEAbgBkACAAdAB5AHAAbwBnAHIAYQBwAGgAaQBjACAAYwBoAGEAcgBhAGMAdABlAHIAIABzAGUAdAAuAAAAAgAA"
    "AAMAAAAUAAMAAQAAABQABACYAAAAIgAgAAQAAgB+AX8gECAVIBkgHSAiICYgMCAzIDogrCEiIhIiYCJl//8AAAAgAKAgECASIBggHCAgICYgMCAyIDkgrCEi"
    "IhIiYCJk////4f/A4TDhL+Et4SvhKeEm4R3hHOEX4KbgMd9C3vXe8gABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAA/4gAMAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAQAEAwABAQEPQm9yem9pLVJlZ3VsYXIAAQEBQ/ifAPigAfihAviiA/gYBPsMDAO7DAQegzMzM8Efi4segzMzM8Efi4sMB8H7"
    "0RwTPvrdBRwFjw+MHQAB6RISHAbsEQCIAgABAAgADwAWAB0AJAArADIAOAA+AEUATABSAFgAYwBuAHgAggCIAI4AlACaAKAApgCtALQAugDAAMoA1ADbAOIA"
    "6ADuAPkBBAEKARABGgEkASsBMgE9AUgBTAFQAVYBXAFjAWoBcAF2AX0BhAGOAZABkgGdAagBrwG2AcIByAHOAdUB3AHiAegB7AHwAfYB/AIDAgoCEAIWAiEC"
    "JAInAi4CNQI7AkECTgJbAmECZwJuAnUCewKBAocCjQKYAqMCqwKzAroCwQLHAs0C0QLVAtsC4QLoAu8C9QL7AwADBQMSAx8DJgMtAzgDQwNOA1kDXwNlA28D"
    "eQN+A4UDjAOSA5gDnAOkA60DuQPEA8kEBwQVBBtuYnNwYWNldW5pMDBBRHVuaTAwQjJ1bmkwMEIzdW5pMDBCOUFtYWNyb25hbWFjcm9uQWJyZXZlYWJyZXZl"
    "QW9nb25la2FvZ29uZWtDYWN1dGVjYWN1dGVDY2lyY3VtZmxleGNjaXJjdW1mbGV4Q2RvdGFjY2VudGNkb3RhY2NlbnRDY2Fyb25jY2Fyb25EY2Fyb25kY2Fy"
    "b25EY3JvYXRkY3JvYXRFbWFjcm9uZW1hY3JvbkVicmV2ZWVicmV2ZUVkb3RhY2NlbnRlZG90YWNjZW50RW9nb25la2VvZ29uZWtFY2Fyb25lY2Fyb25HY2ly"
    "Y3VtZmxleGdjaXJjdW1mbGV4R2JyZXZlZ2JyZXZlR2RvdGFjY2VudGdkb3RhY2NlbnR1bmkwMTIydW5pMDEyM0hjaXJjdW1mbGV4aGNpcmN1bWZsZXhIYmFy"
    "aGJhckl0aWxkZWl0aWxkZUltYWNyb25pbWFjcm9uSWJyZXZlaWJyZXZlSW9nb25la2lvZ29uZWtJZG90YWNjZW50SUppakpjaXJjdW1mbGV4amNpcmN1bWZs"
    "ZXh1bmkwMTM2dW5pMDEzN2tncmVlbmxhbmRpY0xhY3V0ZWxhY3V0ZXVuaTAxM0J1bmkwMTNDTGNhcm9ubGNhcm9uTGRvdGxkb3ROYWN1dGVuYWN1dGV1bmkw"
    "MTQ1dW5pMDE0Nk5jYXJvbm5jYXJvbm5hcG9zdHJvcGhlRW5nZW5nT21hY3Jvbm9tYWNyb25PYnJldmVvYnJldmVPaHVuZ2FydW1sYXV0b2h1bmdhcnVtbGF1"
    "dFJhY3V0ZXJhY3V0ZXVuaTAxNTZ1bmkwMTU3UmNhcm9ucmNhcm9uU2FjdXRlc2FjdXRlU2NpcmN1bWZsZXhzY2lyY3VtZmxleFNjZWRpbGxhc2NlZGlsbGF1"
    "bmkwMTYydW5pMDE2M1RjYXJvbnRjYXJvblRiYXJ0YmFyVXRpbGRldXRpbGRlVW1hY3JvbnVtYWNyb25VYnJldmV1YnJldmVVcmluZ3VyaW5nVWh1bmdhcnVt"
    "bGF1dHVodW5nYXJ1bWxhdXRVb2dvbmVrdW9nb25la1djaXJjdW1mbGV4d2NpcmN1bWZsZXhZY2lyY3VtZmxleHljaXJjdW1mbGV4WmFjdXRlemFjdXRlWmRv"
    "dGFjY2VudHpkb3RhY2NlbnRsb25nc3VuaTIwMTB1bmkyMDE1bWludXRlc2Vjb25kRXVyb25vdGVxdWFsbGVzc2VxdWFsZ3JlYXRlcmVxdWFsYm9yem9pX2xv"
    "Z284LjAwMEN1c3RvbSByZWNvbnN0cnVjdGlvbiBmcm9tIGEgdXNlci1zdXBwbGllZCB3b3JkbWFyayByZWZlcmVuY2UuQm9yem9pIFJlZ3VsYXJCb3J6b2kA"
    "AAEAAQYAaAAACTcAfAAAQh0BhwAAYAIAZwAAZAAAoAAAZgAAgwAAqgAAiwAAagAAlwABiAAApQAAgAAAoQAAnAABiQEAfQAAmAAAcwAAcgAAhQABiwAAjwAA"
    "eAAAngAAmwAAowAAewAArgAAqwEAsAAArQAArwAAigAAsQAAtQAAsgIAuQAAtgIAmgAAugAAvgAAuwEAvwAAvQAAqAAAjQAAxAAAwQIAxQAAnQAAlQAAywAA"
    "yAEAzQAAygAAzAAAkAAAzgAA0gAAzwIA1gAA0wIApwAA1wAA2wAA2AEA3AAA2gAAnwAAkwAA4QAA3gIA4gAAogAA4wABjDAAkQABvQ4AjAAAkgABzA4AjgAA"
    "lAAB2wsAwAAA3QAB5xUAxgAB/QMAxwAA5AACAQEBOgAAbwAAiQACAwAAQQAACAAAaQAAdwAAcAEAdAAAeQECBAEAawECBgAAmQAApgACBwMBWQMAAAEAAAQA"
    "AAcAALoAANMAAWEABHAACG8ACy0ACzkADA0ADN0ADS0ADVEADeoADfgADqQADrgAEJEAEN8AEo4AFaAAFicAGGgAG0cAG30AHykAIhAAI2AAJJ8AJNwAJPQA"
    "JTgAJ+MALI8ALPwALwYAMQIAMhkAMn8AMrwANOkANQsANRUANicANl0ANowAN0wAN8gAObIAOtIAPN8APgsAQRQAQS0AQjsAQoUAQ1cAQ6IAQ9AARDUARI4A"
    "RKIARP4ARUkARVYARW4AR0cASaEAS6kATegAUBQAULMAU4cAVH8AVTMAVmEAVpkAVqMAWIgAWZAAW5sAXesAX8QAYFkAY2MAZBMAZRwAZWUAZiEAZmkAZxYA"
    "Z4UAaaQAabEAa80AbNoAbN0AbOgAbv4AcMgAcwEAc18Ac3cAd+sAeT0AfSUAfuAAfzwAf3AAf34AgqgAgrYAhLEAhN0AhmAAiTYAiUoAimQAi6IAjE0AjU4A"
    "jY8Aj5IAj+8AkLEAkoMAlcoAl8AAmEIAmMMAmWAAmroAnEwAnnQAnv4AocwAokIAorgAo0kApNsApPgApRcApVEApnwAp8EAqRUAqxEArQ0AryUAse0AtPQA"
    "tTsAt00AuG0AuY0AuskAvPcAvTUAvlUAwXEAw1wAxUcAx04AygUAzP4A0IcA030A1lkA2JgA2tcA3TIA4HsA4UEA4gkA4ugA5LYA5ukA6M0A6uoA7QcA70AA"
    "8ikA9VEA9q4A+OAA+fsA+xYA/E0A/nQA/zEBAWkBAzgBA7QBBZoBBowBCOYBCjQBDNkBDuUBEP0BEyUBFVkBF+sBGowBHLYBHuwBIDABIp8BI+QBJjsBJqsB"
    "KOUBKcsBLHkBLXUBMDoBMXYBNGUBNPkBN1YBOa4BPLABP10BQrIBRXIBSN4BS+EBTzIBT4ABUKQBUO4BUfwBUuYBVHUBVI8BVU4BVdwBVw8BV/0BWZEBWjAB"
    "WjoBWtQBW2gBXKYBXgUBXxwBYDcBYHEBYLIBYNEBYeABYs4BYy0BY2kBZDwBZOkBZUUBZXkBZgUBZx0BaHwBamcBaxEBbEcBbU8BbkwBb9EBccgBc+ABdkwB"
    "eNkBet0BfQIBfw4BgvIBhDABhNUBhuMBiFsBibYBinkBjZIBkKwBk+EBlxcBmvMBnsYBof0BpTUBpjEBp8UBqAsBqOkBqRgBqd4Bq8YBrakBrsQBr9oBsWoB"
    "svUBtbMBuGwBuZUBurkBvJABvmQBv2EBwEkBwKIBwXsBwskBwz4Bw70BxLYBxbkBxksBxugBx3ABx34Bx4wBx5oBx6gBx7YByD8ByMcBydQByuEBywUByz8B"
    "y/YBzewB0esB0fcB0hAB0kQB0nQB1LEB1YEB1Y8B1eMB1i4B1nYB3hb47A733g73Zs6rFYiMiAeMiIyIjImNiIyJjYiNiY2JjomNio6JjYqOio6KBY6KjgaO"
    "Bo4GjoyOBo6MjoyNjI6NjYyOjY2NjY2NjoyNjY6MjYyOjI4FjoyOB44HjgeOio4Hio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmMiIyIjAWIjIgGiAaIBoiKiAaI"
    "ioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIBYiKiAeIB5T3axXF+PZR/PYGDvgX95r58BX7esX3elEH+1cW+3rF93pRBw75ct33sxVp5YqMigd3+47FiZ/3"
    "kIuMjIwF94SKjIoGd/uOxYmf95CLjIyMBeCtOQaKjIuMn/eaBYyMjOytLYyKjAee94RRjXj7hgWKior7hIyKjAee94RRjXj7hgWKioo6admKjIoHd/uaBYqK"
    "ii4H90D3nBWMjIz3hIqMigd3+5oFioqK+4QHiowFjAcO+dTd9BWWgYyLl4GMipeCjIqYgouKmYOMipiDjIoFmoSMipmEjYqahYyKm4WMipuGjYqbho2KnIeM"
    "ip2HjYudiIyKnoiNi56JjIuaigWKjHPFoQeMjJ6LpY2Ni6ONjYujj42LBaGPjYygkI6MnpGOjJ2SjYydk42Mm5SNjJmVjYyYlo2Ml5eMjJWXjI2UmIyMkpkF"
    "i4yMjJGai4yPm4yMjZyLjIyci4yKn4uMiZ2KjIeci4yFnIuMg5qLjYKZioyAmQWNigeAmImMBX6Xiox8l4qMe5aJjHqViYx5lImMd5SKi3aTiYx1komMdJGK"
    "jHKRiotxkYqLco8FjIr4LQeMjAeZBpmJmYmZiZiImIiYh4yLl4aYhgWXhpiFl4SWhIyLloOWg5aCloKWgZWAvZ2AloCWiouAlYqMf5SLjH6UfpSKi36TBYqM"
    "Bn6Siot+koqMfZGKi32RiYt9kImMfY+JjHyOiox7joqLe46Ji3yNiYx7jImLe4wFiQaJBoqMBaFRcweKioKLiIpziImLdIYFiIp2hoiKd4SJineDiYp5gYmK"
    "eoGKiXuAiop9foqKiot/fYqKf3yKioF7i4qCegWLioR5ioqGeId3ioqJd4uKinWMdouKjXmMio95i4qRe4uJkn2MiZR9jIqVfoyKBZZ/jYqYgIyKmoGNipqC"
    "jYqcg42KnYSNip6EjYughIyLoYWMi6KFjIukhoyLlYkFjIoG/BQHiooHhQZ6jXqOeo57j3uPfJB8kH2RfZF9kn6SfpN+k3+TgJR/lYCVW3cF9/X3tRWMjAeM"
    "BpKKo4aihaGFn4Sfg52DnIObgZqBmIGXgJZ/lH6Mi5N9kn2QfIuKBY97jXqMeYp7i4qJfId8hn2EfoN/goCAgX+BfoN9g3yEe4V5hXiGd4d2iHSJdIoFfAaK"
    "jAZQ+ckVjIoG/B8HiooHe451kHaQi4x4kHmSe5J7kn6TfpOAlYCVgpaEmISYi4wFhpmHnImdip6Mn4uMjZ6PnpCdkpuTm5SalpiLjJeXl5eZlZqUm5Sckp2R"
    "npCfjwUO+cD4RNsVgQeKB42Bi4qNgouKjoKLio6CjIuOgoyKkIOLigWQhIyKkYOMipGFjIqShIyKkoaMipOFjYqTh42Kk4eOipSHjYqViY6KlImPi5SJBY8G"
    "lQaPBpUGjwaUjY+LlI2OjJWNjYyUj46Mk4+NjJOPjYwFk5GMjJKQjIySkoyMkZGMjJGTjIyQkouMkJOMjI6UjIuOlIuMjpSLjI2Ui4yNlQWMB5UHlQeMB4mV"
    "i4yJlIuMiJSLjIiUiouIlIqMhpOLjAWGkoqMhZOKjIWRioyEkoqMhJCKjIORiYyDj4mMg4+IjIKPiYyBjYiMgo2Hi4KNBYcGgQaHBoEGhwaCiYeLgomIioGJ"
    "iYqCh4iKg4eJioOHiYoFg4WKioSGioqEhIqKhYWKioWDioqGhIuKhoOKioiCiouIgouKiIKLiomCi4qJgQWKB4EHy60VjAeOko6Tj5KQkpCRkJGRkJGQkY+R"
    "jpGOkY6RjAWRjZGLkYyRipGLkYmRipGIkYiRiJGHkYaRhpCFkIWQhI+EjoOOhIuKjoONg4yCBYIHggeKgomDiIOLioiEiIOHhIaEhoWGhYWGhYaFh4WIhYiF"
    "iIWKBYWJhYuFioWMhYuFjYWMhY6FjoWOhY+FkIWQhpGGkYaSh5KIk4iSi4yIk4mTipQFlAeUB4yUjZMF/AAnFcF/+Ij58FWX/Ij98AVc+ZoVgQeKB42Bi4qN"
    "gouKjoKLio6CjIuOgoyKkIOLigWQhIyKkYOMipGFjIqShIyKkoaMipOFjYqTh42Kk4eOipSHjYqViY6KlImPi5SJBY8GlQaPBpUGjwaUjY+LlI2OjJWNjYyU"
    "j46Mk4+NjJOPjYwFk5GMjJKQjIySkoyMkZGMjJGTjIyQkouMkJOMjI6UjIuOlIuMjpSLjI2Ui4yNlQWMB5UHlQeMB4mVi4yJlIuMiJSLjIiUiouIlIqMhpOL"
    "jAWGkoqMhZOKjIWRioyEkoqMhJCKjIORiYyDj4mMg4+IjIKPiYyBjYiMgo2Hi4KNBYcGgQaHBoEGhwaCiYeLgomIioGJiYqCh4iKg4eJioOHiYoFg4WKioSG"
    "ioqEhIqKhYWKioWDioqGhIuKhoOKioiCiouIgouKiIKLiomCi4qJgQWKB4EHxp0VjZOOk4uMjpKOk4+SkJKQkZCRkZCRkJGPkY6RjpGOkYwFkY2Ri5GMkYqR"
    "i5GJkYqRiJGIkYiRh5GGkYaQhZCFkISPhI6DjoSLio6DjYOMggWCB4IHioKJg4iDi4qIhIiDh4SGhIaFhoWFhoWGhYeFiIWIhYiFigWFiYWLhYqFjIWLhY2F"
    "jIWOhY6FjoWPhZCFkIaRhpGGkoeSiJOIkouMiJOJk4qUBZQHlAcO+bH3kPgEFYoHigdFWYqLfH+Kin1/iop+fgWKigeAfouKgX2KioN9ioqEfIqKhnyKiod9"
    "i4qIfIuKiXwFigd8B4oHjHyMio18jIqPfQWLipF9jIqSfYyKlH6MipV/jImXgI2Km32Nip2Ajomego+KnoSQip+HkIqgiZCKBaEGjwahjY+Loo6PjKKRjoui"
    "k46LopWNi6OWjYyimIyMo5mMjJ2YBYyKBr1Sv5tM1AWKjIwGlZWMjKCgjIueooyL6vcHV5ss+wd4dX5+BYoGigb7Z/eHBYwHjAf3ANedmIyLnJiMjJuYjIua"
    "mYyMmZiMjJiZBYyMBpeZlpqMi5SajIyTmpKajIyQmoyMj5qLjI6bi4yNmwWMB5oHioyKm4uMh5uLjIaci4yEm4uMBYGdi4yAm4qMf5qJjH6ZiYx8l4mMe5aJ"
    "jHqUiYx5k4iMeZGIjHiQh4x4joeLd40FhwZ3BocGeImHi3eIh4p4hoiKeISIiniDiYp5gYmJBXp/iop8gIqKfn6Kin99ioqAfYqKgnyKioN8ioqEe4uKhXuL"
    "ioZ6i4qIeouKinoFigd5B4x5i4qOeouKkHmLipF6i4qTeouKlXqLipZ7jIqYewWKB7+bFX+agZqCm4SbhZyHm4icipyLm4ycjpuPm5CbkpqTmZSZlZiWmJeW"
    "mZaalgWalJuSm5Cbj5uOm4ybi5uKm4ibh5qGjIuahZmDjIuZgZiBl3+WfYyLlH2LipR7BZF8kHyOfI18i3yJfIl9h32LioV9hX2DfYJ+gX1/fn99fX58fnx/"
    "iot7fnl+LkoFigZ/TBWMigb3a/uLBYoHigeBg3V8dX12gIqKdoF1g3eFdoZ4iHiKeot6jXuOiot8kXuSfJQFfZeAlYKWg5eEl4aYh5iImIqZi5mNmY2Zj5mR"
    "mZGYk5mVl4uMlZeWl4yLl5aZlgUO91TO+QUVxfd/UQYO+Bn3jfoEFYoGeG+KiwV5b4qKem97bnxui4p9bn9tf22BbYuKgm2LioNthGyLioVshmyLiodsiGuJ"
    "a4lrBYoHawdrB4oHjWsFjWuOa49si4qQbJFsi4qSbJNti4qUbYuKlW2XbZdtmW6Lippum26cb4yKnW+MiwWeb4yLoHC/mXameKZ5p3qne6d9p32of6iAqIGp"
    "gqmDqYSqhqmGqoeqiKuJqomrBasHqweNq42qjquPqpCqkKmSqpOplKmVqZaol6iZqJmnm6ecp52nnqagpleZBQ74Gc77uRXAfaCmn6eepwWLjJynm6iaqIuM"
    "maiXqYyLlqmVqYuMlKmLjJOpkqqLjJGqkKqLjI+qjquOq4yrBYwHqwerB4wHiquIq4irh6qLjIaqhaqLjISqg6mLjIKpi4yBqYCpiot/qX2oi4x8qHuoeqeL"
    "jAV4p3endqZWfaBwnnCdb5xvm2+ab5hul26WbpVtlG2TbZJskW2QbI5sjmuObIxrBWsHaweKa4hsiGuIbIZshW2EbINtgm2BbYBuf25+bnxve296b3lveHB2"
    "cAUO+PHd+A0VtHP3KeQFjIoG+1OMB8T3U4yMjAb3KTK0o/tG9gWMB4wH90b2YqP7KTIFioyK91NSBor7UwaKigf7KeRic/dGIAWKB4oH+0YgBQ75QN339BVp"
    "93qKjPuGxfeGjIz3eq37eoyK94ZR+4aKivt6Bw73g877IBW9BsL3OQVwBoqMBo4HjI4FjgeOB4qOi46KjoqOio2JjoqNiY6JjYmNiI2JjIiNiYyIjIiMBYiM"
    "iAaIBogGiIqIBoiKiIqJioiJiYqIiYmJiYmJiIqJiYiKiYqIioiKiAWIB4gHiAeIB4yIjIiMiIyJjYiMiY2IjYmNiY6JjYqOiY2KjoqOigWMBoyKBooHavsc"
    "BQ74quz3xxVp9+it++gHDvdmzqEViIyIB4yIjIiMiY2IjImNiQWKB4aRB4yKjYqOiY2KjoqOigWOio4GjgaOBo6MjgaOjI6MjYyOjY2MjIwFkZAGjAeNjYyN"
    "jY6MjYyOjI4FjoyOB44HjgeOio4Hio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmMiIyIjAWIjIgGiAaIBoiKiAaIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqI"
    "BYiKiAeIBw75d937BxXCgfic+pBUlfyc/pAFDvnO3/hCFYxnjmiQaZFqBYyKBpNsi4qVbYuKl26Yb5pxi4qbc4uKnHSMip13BYyKnniNip97jImhfY2KoX+O"
    "iqGBj4qjhI+Ko4eRiqOKkYujjJGMo4+PjKOSj4wFoZWOjKGXjYyhmYyNn5uNjJ6ejIydn4yMnKKLjJuji4yapZinl6iLjJWpi4yTqgWLjJKskK2Oroyviq+I"
    "roathKyLjIOqi4yBqYuMf6h+p3yli4x7o4uMeqKKjHmfBYqMeJ6JjHebio11mYmMdZeIjHWVh4xzkoeMc4+FjHOMhYtzioWKc4eHinOEh4oFdYGIinV/iYp1"
    "fYqJd3uJinh4iop5d4qKenSLintzi4p8cX5vf26LioFti4qDbAWKigeFaoZpiGiKZwXFFoyujq2QrJGqi4yTqZWplqeYppikjIuaopuhnJ6dnZ6anpielQWf"
    "k56Rno6ejZ6JnoiehZ+DnoGefp58nXmceJt1mnSMi5hymHCWb5Vtk22LipFsBZBqjmmMaIpoiGmGaoVsi4qDbYFtgG9+cH5yiot8dHt1enh5eXh8eH54gXeD"
    "eIUFeIh4iXiNeI54kXeTeJV4mHiaeZ16nnuhfKKKi36kfqaAp4Gpg6mLjIWqhqyIrQUO+GP3DvlUFb159wH3AQWMBoyKBv2uxfnwB40Hio2LjIqNiY2JjYiM"
    "iY2IjIqLiIyIjAWHBocGhwaIBoeKiIqIioiKi4qIiomJ+zn7OQUO+b7f+TsVxIORnAWSm5OajIuUmpWYlZeMi5aXl5aYlJmUmZOakouMmpGbkJuQnI+cjp2N"
    "nYydjKSKBaKJoIiLiqCHnoWehJyDm4KagIyLmICMi5d+jIuWfZZ9i4qUfJN6kXmReY54jncFdgd5B4l6iHqHe4d7iouGe4R8g3yCfIJ9i4qAfYB9fn0Ffn19"
    "fX19e317fXt+eXz8EPu/iYmJiYqIi4mKiYyJi4iNiYyJjYmOio6JjoqOio+KBZIG+OKt/KSMiowGjIz36fehjIycmZ2Zm5qMi5qZjIuamgWZmYyMmJmMjJeZ"
    "jIyWmYyMlpqVmoyMlJqLjJObjIuSm4uMkZuLjJGcj52OnY2dBYwHngehB4qMiZ+LjIefi4yFn4Sei4yDnYqMgZyKjICaBYqNf5mKjXyYio18l4mNepaJjHmW"
    "iYx3lImMdpOJjHSSiYtzkIiMc4+Ii3GNiIwFcAaIBnYGiQZ3iYiLd4mJiniIiIt4homLeYWJi3mFiYp6hAWJinuDiYp7goqKfIGKi3yAiop+f4qKfn+KioB+"
    "ioqAfIF8i4qCe4uKg3qLioR6BQ75zt/vFZaBjIqXgoyKl4KMipiCjIuYgoyLmYOMipmEjIoFmoSMi5qEjIubhYyLm4WMi5yGjIucho2LnIeMi52IjYqdiYyL"
    "nomNip6KjIufigWMBp8GjQalBo0GpI2NiwWijY6MoY6Oi6GQjYufkY6LnpKOjJ2SjYyck42Nm5SNjJmWjIuMjJiWjY2Wl42MBYuMlZiMjJSai4yTm4uMkZuL"
    "jY+cjIyNnouMjJ+KoIuMiZ6LjIeei4yGnIuMhJwFi4yDmoqMgpqKjIGZiox/mIqMfpaKjIuMfZWJjHyViYx6k4mMepOIjHmRiIyIjAWKjAaMjAeYkI2Mm5KN"
    "jJqTjYyYlI2MmJWMjJeWjIyWl4yMlJiMjJOYjIySmgWLjJGai4yQnIuMjpyLjI2di4yMnoqgi4yJn4uMh56LjIadi4yEnYuMgpuLjIKbBYqMBoCaBYqMf5mK"
    "jH2YioyLjHyWiY17lYmMeZWJjHiUiYx3koiMdpGIi3SQiIxzjomLcY4FiQZwBokGegaJBnuKiYt6ioqKeomKi3uIiYsFfIiJinyIiYp8h4qKfIaKi3yFiot9"
    "hYqKfYSKi36Eiop+g4qLf4KKi3+Ciop/ggWKioCBiouAgICAvXmWlpWVlpSWlJaTl5OXkpeSl5GYkJeQjIuYkJiPmI6Mi5iOBZmNmo2ajZqLmoyjiqGJoIif"
    "h56GnYSbhJuCmYKZgJd/ln+VfZR8k3yLipF7kHkFjnmNeIx3inmKeYd7h3yGfIR+hH6CgIKAgIKAgn+DfoR+hXyGfId7h3qIeIl3igWIi4eKiIqHiomKiImJ"
    "iYmKi4qKiYqJiomLiYyJjImLioyJjYmNio6JjoqOio6KBY+KjouiiZ+JnoieiIuKnIebhZqFmoOYg5iBloGWgJR+lH6SfJF8kHqOeY15jHcFiniJeouKh3uG"
    "fIuKhH2DfoF/gYCLin+BfoJ9g3yEi4p7hXqFeYd3h3eHdYl0igVzinmMiot5jHqMeo16jXuOe458j3yQfJB9kH2RfZF+kn+TfpJ/lICTf5WAlVt3BQ75x/jd"
    "FsX3coyM6q0sjIr47gaNB4qNio2LjImMi4yJjImNiI2IjIiMh4wFiAaHBocGiAaHioiKiIqKi4mKiImJiomJ/I39AImJi4qKiQWJB4gHjImMiYyJjYmOiY6J"
    "joqPio6KBY8Gjwb4b4qMigb8P68VjIqMB/g++J8FjIuMigb8n4qKBw75yt/mFZSCjIqVg4uKloOMipWDjIuXg4yLl4SMipeEjIsFmISNi5iFjYqZhoyKmoaM"
    "i5uGjIubh4yKnIiMi5yIjYqciY2LnYmNip6KjIueigWNBp8GjQamBo4GpY6Oi6SOjoyjkI6LBaKSjoyhko2MoJSNjJ+WjYydlo2NnJiMjJuZjIyZm4yMmJyM"
    "jJadi42VnouMk6AFjIyRoYuMkKKLjI6ji4yMpYqji4yIoouMhqCLjIWgi4yDoIuMgZ6LjICdiox+nAWKjH2biox8momMepmKjHiYiYx4loiMdpWJjHSTiYxz"
    "koiMcpCIjHGPiItvjYmMBW4GiQZ7BooGe4qKi3yKiYt8ioqKBXyJiot8iYqKfYiJi32Iiop9iIqKfYeKin6Hiop+hoqLfoWKin6Fiot/hIqLiooFigaKjAaf"
    "9+cFjIyM+JSt/LAHhwaHBoiKh4qIioiKiYmJiYmJiomKiYuJcvwuBYgHiQeMiYyLjImNiY6JjomOio6Kj4oFjwaPBpIGj4yOjI+MjY2Mi42NlZOWk5WSlpKW"
    "kpaRlpCXkJaQl4+Xj5iPl46YjZiOmIyZjQWYjJmLmYyliqSJo4ehhqCFn4Segp2BnICafpp9mH2We5Z6i4qUeZJ4kXePdo51BYx0inKIdIuKh3WFdoN3gnmB"
    "eoqLgHt+fXx+i4p8gHqAeoJ4g3iEdoaLinaHdIgFc4lyinmMeYt6jXqNe418jnyOfY99j32Pf5B+kX+QgJKAkYGSgZKBk4KTgpNaeQUO+bD3JfhDFYoGiowG"
    "lweOtI+ykbCTrpWrlqqXp5mkmqOMiwWboJ2dnZuLjJ+Yn5eglIyLoJKikKKPpYyVipaLlYqVipWJlYmViJWJi4qUiJWIBZSHlIaVhpSGlIaUhZSEk4SUhJSD"
    "u5+Ck4qLgpOBk4GSioyCkYqMgZGKi4GRiosFgZGKi4CQioyBj4mMgY+Ji4CPioyAjomLgI6Ji4COiYt/jYmLf42Ki36Miot+jAV9BogGboqHi3CHhosFcYWH"
    "inKCiIpzgIiKdH6JinV7iYp3eYqKeHeKinp1iop7coqLfXCKin9uioqAbAWBaYNnhGWHY4uKiGKLiopfi/sqjHSLio11jIqPdouKkXiLipJ4jIqUeYuKlXqM"
    "igWXe4yKmHyMipl+jYmaf42KnICNip2BjYqego2KoIONi6GEjYuiho2Ko4iOi6OIBY4GpAanBo6Mo42Oi6KPjoyikI2MBaGRjYygk42MnpWNjJ2VjY2clouM"
    "jIybmIyMmZmMjJibjIyXm4yMlZ2LjJSdjIwFkp+LjJGfi4yPoIyMjaGLjIyiiqKLjImhioyHoIuMhZ6LjISeioyCnYuMgZyKjAV/m4qMfpqKjH2YiY18l4mM"
    "epaJjHmViYx4lImMdpOJi3WSiYt0kImMc46Ii3OOBYgGcgaJBngGiAZ5iYiLeYkFiYt5iIiKeoeJi3qGiYp7homKe4WJinyEiop8g4mKfYKKin6CiYp+gYuK"
    "foCKigWELRWSmJSZlJiVmJaXlpaWlZeVmJOYk5iTmJGMi5mRmZCajwWajpqOm42Mi5uMnIyhiqGJn4ifh52FnYWcg5uCmoGZgJl/l32VfZV7k3qSeZF4BY93"
    "jXaMdYp1iXaHdoV4hHiDeoF6gHx/fX59fYCLinyBiot7gXqDeoSKi3mFd4YFd4h2iXWKdYx1jXeOd495kXmRepN7lHyVfZZ9l3+ZgZmBm4OchJ2FnoefiaCK"
    "oQUO+czb+d8V+OaKjIoG/ED92MOB+Ez58IyNi5CKjYqNiY2JjYmNiI2IjIeMiIwFhwaHBv0PaQYO+c74F/oBFYkGc4mJinSIiYt1h4mKdoWJi3eEiYp5g4mK"
    "eoKJinuBiYp8gIqKBX5+iYp/foqKgH2LioF7ioqDe4uKhHqLioV5i4qHeIl3i4qKdox4i4qNeI95i4oFkHuLipF7jIqTfIuKlH2MipZ+jIqWf42KmICMiZqB"
    "jYqago2KnIOOip2EjYqSiQWMigaKigd6hoiKd4MFiYp4g4mKeYGJinuAioqKi31/iYp+foqKf32KioB8ioqCe4uKg3qLioV5i4qHeQWKigeJd4uKinaMdouK"
    "jXiMio94i4qReouKk3uMiZN8jIqWfYyJBZd+jIqZf4yKmoCNiZuBjYqdgo2KnoONi5+EjoqghY6LoYaOi6KHjoukiI2LpYkFjQamBo0GpgaNBqWNjYsFpI6O"
    "i6KPjouhkI6LoJGOjJ+SjYuek42MnZSNjJuVjY2aloyMmZeMjJeYjI2WmQWMjAeTmoyNk5uLjJGci4yPnouMjp6LjIygiqCLjImfBYqMBoedi4wFhZ2LjIOc"
    "i4yCm4qMgJqKjH+Ziox+mImMfJeKjHuWiYx5lYmMeJOJjHeTiIx6kAWKjAaMjAeSjY2MnZKOjJyTjYyalI2MmpWMjZiWjYyWl4yMlpiMjJSZi4wFk5qMjJGb"
    "i4yQm4uMj52NnouMjJ6KoIuMiZ+HnouMhZ2LjISci4yDm4qMgZuLjAWAmYqMf5iJjH6Yiox8lomMe5WJjHqUiYx5k4mMd5KJi3aRiYx1j4mLdI6JjHONBYkG"
    "cgaJBqX8ZRWjiaKJoIefhp6GnYSchIuKm4OZgZiBl3+Mi5V/lX2TfJJ8kXoFj3mNeYx3ineJeYd6hXqEfIN9gn+LioCAf4B+gXyCfIN6hYuKeYV4hneHdYh0"
    "iAVzinKKcoxzjHSOdY53j3iQeZGLjHqRfJN8lH6Vf5aAlouMgpeDmYSahZyHnImdBYqfjJ+NnY+dkZySmpOalZmVl4yLl5eYlZmVm5OLjJySnZKekJ+QoI+i"
    "jaONpIwFofhBFZ+Jn4meh52Gm4WbhJqEmYKYgZeAjIuWf5V+k32TfJF7kHoFj3mNeIx3inmJeYh7hnuFfYR9g3+LioKAgICAgouKfoN+g3yEe4V6hnmGd4h3"
    "iAV0inSKdIyKi3WMd453jnmQepB7kXySfpN+k4uMgJSAloKWi4yDl4SZhZmGm4ibBYmdip2Mn42ej52QnJGbk5qTmZWYlpeMi5eWmJWZlJqSm5KbkZ2Qno+f"
    "jZ+NoYwFDvmw9xDWFZSDjIuUg5WDlYSMipSFjIqVhYyLlYWMiwWVhYyLloaMipWHjYqVh42LloeMipaIjYuWiI2LloiNi5eJjYuXiYyLmIqMi5iKBZkGjgao"
    "jI+Lpo+QiwWlkY+MpJSOjKOWjoyimI2MoZuNjJ+djIyen4yMnKGMjJukjIuZpoyMl6iMjJaqBZWtk6+SsY+zi4yOtIuMjLeL9yqKoouMiaGKjIegi4yFnouM"
    "hJ6KjIKdi4yBnIqMBX+biox+moqMfZiJjXyXiYx6lomMeZWJjHiUiYx2k4mLdZKJi3SQiYxzjoiLc44FiAZyBogGcgaIinOJiIt0h4iKdIaJinWFiYp2g4mK"
    "eIGJinmBiYl6gIuKiop7foqKfX2Kin57BYqKBn97ioqBeYuKgnmKigWEd4uKhXeLiod2ioqJdYuKinSMdIuKjXWMio92i4qReIuKkniMipR5i4qVeoyKBZd7"
    "jIqYfIyKmX6NiZp/jYqcgI2KnYGNip6CjYqgg42LoYSNi6KGjYqjiI6Lo4gFjgakBo0GngaOBp2NjoudjY2LBZ2Ojoycj42LnJCNjJuQjYybkY2MmpKMjJqT"
    "jYyZlIyMmJSNjJiVi4yYloyMj5AFjAaMigZ/B4hih2SFZoNogWuAbH9vfXJ8c4qLe3Z5eXl7i4p3fnd/doKKi3aEdIZ0hwVxioGMgIuBjIGMgY2BjYGOgY2L"
    "jIKOgY6Cj4KQgZCCkIKQgpGCkoOSgpKCk1t3Bfim+E8VhH6CfYJ+gX6Af4CAgIF/gX6DfoN+g36Fiot9hX2GfId8iAV8iHuJiot7inqKdYx1jXeOd495kXmR"
    "epN7lHyVfZZ9l3+ZgZmBm4OchJ2FnoefBYmgiqGMoY2gj6CRnpKek5yVnJaal5mYmZmWi4yalYyLm5Wck5ySjIudkZ6QjIsFn46gjaGMoYqhiZ+In4edhZ2F"
    "nIObgpqBmYCZf5d9lX2Ve5N6knmReI93jXaMdQUO92bO+FUVjIgGjIiMiIyJjYiMiY2IjYmNiY6JjYqOiY2KjoqOigWOio4GjgaOBo6MjgaOjI6MjYyOjY2M"
    "jo2NjY2NjY6MjY2OjI2MjoyOBY6MjgeOB44HjoqOB4qOio6KjYmOio2JjomNiY2IjYmMiI2JjIiMiIwFiIyIBogGiAaIiogGiIqIiomKiImJioiJiYmJiYmI"
    "iomJiIqJioiKiAWIiogHiAeIB/w7BIyIBoyIjIiMiY2IjImNiI2JjYmOiY2KjomNio6KjooFjoqOBo4GjgaOjI4GjoyOjI2Mjo2NjI6NjY2NjY2OjI2NjoyN"
    "jI6MjgWOjI4HjgeOB46KjgeKjoqOio2JjoqNiY6JjYmNiI2JjIiNiYyIjIiMBYiMiAaIBogGiIqIBoiKiIqJioiJiYqIiYmJiYmJiIqJiYiKiYqIiogFiIqI"
    "B4gHiAcO94PO+yAVvQbC9zkFcAaKjAaOB4yOBY4HjgeKjouOio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmMiIyIjAWIjIgGiAaIBoiKiAaIioiKiYqIiYmKiImJ"
    "iYmJiYiKiYmIiomKiIqIiogFiAeIB4gHiAeMiIyIjIiMiY2IjImNiI2JjYmOiY2KjomNio6KjooFjAaMigaKB2r7HAWN+OEVjIiMiIyIjImNiIyJjYiNiY2J"
    "jomNio6JjYqOio6KBY6KjgaOBo4GjoyOBo6MjoyNjI6NjYyOjY2NjY2NjoyNjY6MjYyOjI6LjoyOBY4HjgeKjouOio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmM"
    "iIyIjAWIjIgGiAaIBoiKiAaIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIiogFiAeIB4gHDvlK5ffMFYmJiYqLiomKi4qKiYqJBYkHiQeMiYyJi4qNiouK"
    "jYqNifh0+8C1o/xf97OKjIyM+F/3s2GjBQ75Nt33dxVp+JKt/JIH93UEafiSrfySBw75St344BX4X/uzBYqMB4qKBvxf+7O1c/h098CNjY2Mi4yNjIuMjI2M"
    "jQWNB40Hio2KjYuMiYyLjImMiY38dPfAYXMFDvmv3flgFcJ/BZOZi4yUmJSXlZeVloyLlZWXlJaTi4yYkpeSmJKYkIyLmJCZkJqOmo6ajpuMm4wFm4ykiqKJ"
    "oYiLiqCHnoWMi52EnYObgpuBi4qZgJl/l32WfZR7k3uSepB5j3iOeAV3B3oHiXuJfId9h32FfoV/hICDgIOAgoEFgYGAgoCCf4J+g36CfYN8g3yDfYOKi36D"
    "iot/g4qKf4OAgoqLgYKKi4KCiouCggWKigeDgoSBiouEgYWBi4qFgouKh4GKioiBiouIgImAi4qJgYuKioCLigWKgMWLjJWLjIyVjJWNlIuMjpSPlI+Ui4yQ"
    "k5CUkZSSk5KTk5OUk5STlpOVkpeTBZeSmZKak4yMmpOalJmUjIuYlIyMmJSMjJeUjIyWlYyMlZWMjJWWjIyUloyMk5YFjIwHkpiLjJKYi4yRmIuMkJmLjI+a"
    "i4yOm4uMjZsFjAedB6AHioyJn4uMh5+FnouMg52LjIKdi4yBm4qMf5uKjAV+moqMfZiJjIuMe5eKjHqXiYx4lYmMd5SJjHaTiIx1koiLc5CJjHKPiItxjYiM"
    "BXAGiAZ5BokGeYqJi3mJiYp5iYmLeoeJi3uHiYp7homLfIWJigV8hYmKfYSJin2Diot9goqKfoKKin+Biop/gIuKf4CLioB/i4qBfoqKgn2LioJ9BfeI/UAV"
    "jIiLiIyIjIiNiYyIjYmMiIyLjImOiY2JjYqOiY6KjYqOigWOio4GjgaOBo4GjIyOi42MjoyOjI2NjoyNjY2NjY2Njo2NjI6MjYyOjI6MjgWOB44HjgeOB4qO"
    "io6KjoqNio6JjYmOiY2JjYmNiIyJjYiMiIyJjIiLiowFiAaIBogGiIqIBoiKiYqIioiJiYqJiYiJiomKi4qIiYmKiImJioiKiIuIiogFiAcO+lv5aO0VjwaW"
    "Bo8GlQaPBpWMjoyVjY6MlI2OjJSOjo2Tj42Mk4+NjQWSkI2MkZGNjJGSjIyRkouMkZOLjJCUi4yPlIyMjpSLjI6VjIyNlouMjZaLjIyXBZgH93AHiqyIq4ap"
    "ioyEp4uMgqaLjICliox/o4qMfaKKjHugiox5n4qMd52KjHabiY0FdJmJjHOYiIxyloiMcJSIjG6SiIttkYiLbI6Ii2uMh4toioiLaYeIimuEh4ptggWIim5/"
    "iIpvfYmKcXuLiomKc3mKiXR3iop2dIqKeHKKinpwiop8b4qKfm2KioBrBYuKgWqLioNoi4qGZ4qKiGWLiopkjGiOaY9ri4qRbYuKk2+LipVwi4qWcYyKl3MF"
    "jIoGmXWMipt3jImceYyJnnqMip98jYqhfo2JooCOiqOBjoqlhI6KpoaPiqeIj4uoigWOBqEGjQagjY6Ln46Oi5+PjYyfj42MnpGNi52SjYydk4yLnJSNjJuU"
    "jYyalQWMjQealgWMjAeZl4yMmZiLjJSUBYwG+wb4axUqioqKjAeIjoGWioyBlIqMgZSJjIGTiouKjICSiYx/kYmMf5CJjH+PiIx+j4iLfo6Ii36NBYcGfgaH"
    "BnoGh4p6iYeLe4iIinuGiIp8homKfYSJin2Diop+goqKf4GKioGABYqKBoF/ioqCfoN9ioqFfYqKhXyGe4uKh3sFi4qIe4uKinqLiop5jHmLiox6i4qOe4uK"
    "j3uLipB7kXyMipF9jIqTfZR+jIqVfwWKjAeVgIyKl4GMipiCjIqZg42KmYSNipqGjoqbho6Km4iPi5yJj4oFnAaPBpgGjwaYjY6LBZiOjouYj46Ml4+NjJeQ"
    "jYyXkY2MlpKMjIyLlZONjJWUjIyVlIyMlZaVl4yMjIwFiowHjIeLio+Ci4qPgouKj4SLioyKkISMipCEjIuMipGFjImSho2Jk4aNigWKB4oHhIR+f36AfYF8"
    "gXyCfISLinuFeoWLinuGeYd6iHmIeIp4igVxjHOOc5B1kXaTdpWLjHiWeJl6m3udfJ59oIuMf6GLjIGjgaaEp4Woh6qKi4msBYqtjLGLjI6wka+SrZWslqqY"
    "qJqnnKWdoouMn6Cgn4yLoZyjmqSYpZank6eRqI4Fqo2oiqeIpoakhaSDooGhgKB9nnyee4uKm3mMi5p3mXWXdJVylHGSb5Bujm2MawX7cAd/B4qAiYCJgYiB"
    "iIKIg4qLh4SHhIaFhoWGh4qKhoeFiIaIhYmFioWJBYUGiAaKjAaMB5aai4yVm4uMlJxTlYN7i4qCfIJ/BYqKB4ePh5GKi4eRiJKKi4iSi4yIk4mUiZSJlYqW"
    "ipYFlgf31FEH+6UEhnyFe4V8hH2DfoR/iouDgIOAgoGCg4GDgoSBhYGGgoeKi4KHgYmCiYGKBYGKfYx+jX6Nf49/j3+RgJGBk4GUiouClIKWg5aLjISXhZiK"
    "i4aZhpqImoibipwFipyMnIycjpuOmpCakJmMi5GYkpeLjJOWlJaUlIyLlZSVk5aRl5GXj5ePmI2YjQWZjJWKlYqUiZWJlIeMi5SHlYaVhZSElYOUg5SBk4CT"
    "gIyLkn+TfpJ9kXyRe5B8BQ7549uPFcSD9wD3vAWMjPf3iowG9wD7vMST+8/58IqNiY2JjYmNiIyIjYiMiIuKjAWIBocGhwaIBoeKiAaHioiJiYqIiYqJiouK"
    "iYqJ+8/98AX4kPfZFYoHiooF+9yMioyMBvc3+FMFjIyMigYO+ejdFqcGjIoGe/fhB4wGqo2Ni6iNjYumj42LpJCOi6OQjYyhkY2MoJONjJ6TjoyclI2Nm5WN"
    "jZmWjIwFl5iMi4yMlZmMjJWai4yTm4uMkZuLjY+cjIyNnouMjJ+Kn4uMiJ6LjIedi4yFnAWKjIScioyCmoqMgJmKjX6YioyKi32Xio17lomMeZWJjHiViYx2"
    "k4mLdZOIi3SRBYqMBoyMB46Njouek46MnZONjZyUjYyalo2MmZeMjAWYmIyMl5mLjJWajIyTm4yMkpyLjJGdi4yPnouMjZ6LjIygiqGJoIuMh56LjIadBYyK"
    "jAeFnIqMg5yKjIKaiowFgJqKjH6Ziox9l4mMe5eKjHqViIx5lImMd5OIjHaSiIx0kImMc4+Ii3GPiYtwjQWJBm4Gigb7u3sGiooHb/3wBvfd+B0VqYqniaWJ"
    "pIeih6GFn4WdhJyDmoKZgYyLl4CXf5V+k32SfYuKkXyPeo15jHgFiniJeouKh3uGfIuKhH2DfoF/gIB/gX2CfIJ6hHmEd4Z1hnSHcohwiIqLb4psigX7ooyK"
    "+AqMjAb3t/hVFaSJooihiJ+GnoWchJyDmoKYgZiAln+Mi5V9lH2SfJJ6kHqOeI14BYuKjHeKd4l5i4qHeoZ6hXuDfYJ9gX5/gH+AfYJ8gnuEeoR4hniGdod1"
    "iXOJcYoF+6GMivgyjIz3nAYO+eb5ZPcjFYB+gH9/gH+Bf4F+gn6CfoN9hAV+hX2FfYaKi32GfYd8iHyIe4l8inuKfIqKi3CMcY9zkHOSdZR1lneZd5p5nXqf"
    "BXuhfaN+pYuMgKeBqYOrha2HroixirKMso6xj66RrZOrlamWp4uMmKWZo5uhnJ8FnZ2fmp+ZoZahlKOSo5Clj6aMjIuaipuKmoqbiZqImoiZh5mGjIuZhpmF"
    "mIWZhAWYg5iCmIKXgZeBl4CWf5Z+v5t/mIuMf5eKjH+Xiot+louMfpaKi32Viox9lIqLBX2Uiot8k4qMfJKJjHyRiox7kImMe5CKi3qQiot6j4mLeo6KjHmN"
    "iot5jYmLeowFiQZ5BogGbIqHi26Hh4tvhYeKcIMFiIpygIiKc36JinR8iYp2eoqKd3iLioqKeXaKint0iop8couKfXCLin9ui4qBbAWDaouKhGmLiodni4qI"
    "ZYpjjGOOZYuKj2eLipJpi4qTapVsi4qXbouKmXCLippyBYyKm3SMip12jIqLip94jIqgeo2KonyNiqN+joqkgI6KpoOPiqeFj4uoh4+LqooFjgadBo0GnIyN"
    "i52NjIudjYyMnI6Ni5yPjIuckIyLm5CNjJuQBYyMmpGNjJqSjIyak4yLmZSMi5mUjIyZlYyLmJaLjJiWjIuXl4yMl5eLjJeYV5sFDvn33RanBoyKBnv3jgeN"
    "BrCMjouvjo6MrZEFjourk46MqZWOjKiYjYymmo2MpJyNjKKejIygoIyMnqKMjJykjIyapoyMmKiLjAWWqoyMlKyLjJKtjIyQsI6yjLSKtIiyhrCKjISti4yC"
    "rIqMgKqLjH6oiox8poqMBXqkiox4ooqMdqCKjHSeiYxynImMcJqJjG6YiIxtlYiMa5OIi2mRiIxnjoiLZowFiQb7jnsGiooHb/3wBsX53RWMjIz3bweuiqyH"
    "qYaohKeCpICkfaJ8oHmfeJ11m3OacZhvlm2Ua5JpBZBnjmWMY4pjiGWGZ4RpgmuAbX5vfHF7c3l1d3h2eXR8cn1ygG+CboRthmqHaIoF+2+MiowGDvmN+On4"
    "UxX8YIyK+B6MjPi6rfzYBocGhwaIioiKiouIiomJiouJiomJiYmKiYuKiokFiQf98AeJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+Nit/LqMivge"
    "jIz4YK0GDvmF2RbF+DCMjPhRrfxRjIr4HoyM+Lqt/NgGhwaHBoiKiIqKi4iKiYmKi4mKiYmJiYqJi4qKiQWJB/3wBw759/h6+BwVafeEioz7aQeDgoB/f4B/"
    "gH+BfoF+gn6DfoQFfYR9hXyFfIeLinyHe4h7iHuIeop5inqKiotwjHGPc5BzknWUdZZ3mXeaeZ16nwV7oX2jfqWLjICngamDq4Wth66IsYqyjLKOsY+uka2T"
    "q5WplqeLjJilmaOboZyfBZ2dn5qfmaGWoZSjkqOQpY+mjIyLmoqbipqKm4maiJqImYeZhoyLmYaZhZiFmYQFmIOYgpiCl4GXgZeAln+Wfr+bf5iLjH+Xiox/"
    "l4qLfpaLjH6Wiot9lYqMfZSKiwV9lIqLfJOKjHySiYx8kYqMe5CJjHuQiot6kIqLeo+Ji3qOiox5jYqLeY2Ji3qMBYkGeQaIBmyKh4tuh4eLb4WHinCDBYiK"
    "coCIinN+iYp0fImKdnqKind4i4qKinl2iop7dIqKfHKLin1wi4p/bouKgWwFg2qLioRpi4qHZ4uKiGWKY4xjjmWLio9ni4qSaYuKk2qVbIuKl26Liplwi4qa"
    "cgWMipt0jIqddoyKi4qfeIyKoHqNiqJ8jYqjfo6KpICOiqaDj4qnhY+LqIePi6qKBY4GnwaNBp6MjYuejY2LnY6Ni52PjYudj4yLnJCNjJyQjIybkY2LBZuS"
    "jIyakoyMmpOMjJmTjIyZlYyLmZWLjJiWjIuYl5eXjIyXl4uMl5iMjYyNjI0FjQf3fweNB4qNi4yKjYmNiY2JjIqLiY2IjIqLiIyIjAWHBocG+6IGDvne3RbF"
    "+DCMjPjEioz8MMX58FH8MIqK/MSMivgwUf3wBg73RscWxfnwUQYO+YXR9z4Vk3qUe4uKlHyMipV9i4qWfouKl36LipeAjIqYgAWMipiBjIuMipmCjIqago2K"
    "moSNipuEjoubhY6KnIaOi52HjYqeiI6Ln4mNip+KBY4GoAaNBqMGjgaijY6LoY6OjKCPjYyfkI6MnpKNjJ2TjYyclI2NmpUFjIyMi5iXjYyYmIyNlpmMjJWb"
    "jIyUnIuMk52LjJGei4yQoIuMj6GLjI2ii4yMpAX49lH89geKcwWJdIh2hneFeIR6g3uBfYuKgX5/gH+AfoJ9g3yEe4Z6hnmHiot4iXeJdYp4jHiMBXqNiot7"
    "jnuOiot8kHyQfZB9kn6TfpN/lICViouAloGXgZiCmIqLg5qDmoOcU4EFDvnu2RbF9/YGjAf3LPcOBYwGjAb4Tvx5vZv8WviIi4z4OvfpXZ/8yvxcBYoGiowG"
    "+FFR/fAHDvmt2RaJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+QCt/OKMivneUQYO+gbdFsX5fQaMjAeMBvej/ImNiYuKjYqLio2KjomOio6Jj4uO"
    "igWPBo8GjwaOjI+Ljo2OjI6NjYyLjI2Mi4yNjfej+IkFjAaMigb9fcX58AeNB4qNio2LjImNiYyJjYiNiIyIjIeMBYgGhwaHBocGiIqHioiKiIqJiYmJiYmJ"
    "ifvB/MEFigaKBvvB+MGJjYmNiY2JjYiMiIyHjIiMBYcGhwaHBogGh4qIioiKiImJiYmKiYmLioqJiokFiQf98AcO+d7dFsX5mAaMjAeMBvjG/aCNiY2JjoqO"
    "iY6KjoqPigWSBo8GjwaOjI+MjoyOjY2MjY2NjYyNi4yMjQWNB/nwUf2YB4qKB4oG/Mb5oImNiY2IjIiNiIyIjIeMBYQGhwaHBoiKh4qIioiJiYqJiYmJiomL"
    "ioqJBYkH/fAHDvoa3fhCFYxnjmiMi5Bpi4qTa4uKlGyLipdti4qYbouKmnCLipxxi4qdcwWMip51jIqfdoyKoXiNiqF7jYqjfI6Ko3+OiqWCj4qlhJCKpoeQ"
    "iqeKj4unjJCMBaaPkIylko+MpZSNjIyLo5eOjKOajIyMi6GbjYyhnoyMn6CMjJ6hjIydo4uMnKUFi4yapouMmKiLjJepi4yUqouMk6uLjJCtj66Mr4qvh66G"
    "rYuMg6uLjIKqi4x/qQWLjH6oi4x8pouMeqWLjHmjiox4oYqMd6CKjHWeiYx1m4qLioxzmoiMc5eKi4mMBXGUh4xxkoaMcI+GjG+Mh4tvioaKcIeGinGEh4px"
    "goiKc3+IinN8iYp1e4mKdXgFiop3doqKeHWKinlzi4p6cYuKfHCLin5ui4p/bYuKgmyLioNri4qGaYqLiGiKZwXFFoyujqyRrJKrlKqWqJiomaWbpJyjBZ2g"
    "n5+fnKCaoZihlqKToZGijqKNoomiiKGFooOhgKF+oHyfep93nXacc5tymXEFmG6WbpRskmuRao5qjGiKaIhqhWqEa4JsgG5+bn1xe3J6c3l2d3d3enZ8dX51"
    "gAV0g3WFdIh0iXSNdI51kXSTdZZ1mHaad5x3n3mgeqN7pH2lfqiAqIKqhKuFrIisBQ75x90WxffHjIz3mAaMBqkGjganjo6Lpo8FjoulkI2MpJGNjKKTjYyh"
    "lI2Mn5aNjJ2XjYybmI2MmpqMjJmajI2Xm4yMlZ2MjAWUnouMkp+MjJCgi4yQoo2jjKSKpImji4yGoYuMhqCKjISfi4yCn4qMgZ2KjH+cBYqMfZuKjHyaiox6"
    "mYmMeZeJjHeWiYx2lYiMdZOIjHOSiItxkYiLcI+Ji26NiIwFbQaKBvu2ewaKigdv/fAG9+753hWliaSHooeghZ+DnoOdgZuAmn+Zfph8lnuVepN5kneRd491"
    "jXWMc4p0i4oFiXWHdoV2hHiDeYF7i4qAfH59fn58f4qLe4B6goqLeYN3hIqLdoV0hnOIcYluigX7l4yK+IeMjPeXBg76Pt34QhWMZ45ojIuQaYuKk2uLipRs"
    "i4qXbYuKmG4Fi4qacIuKnHGLip1zjIqedYyKn3aMiqF4jYqhe42Ko3yOiqN/joqlgo+KpYSQigWmh5CKp4qPi6eMkIymj5CMpZKPjKWUjYyMi6OXjoyjmoyM"
    "jIuhm42MoZ6MjI2NBYyMB4yKBvcR+xG9nfsj9yOLjI+QjIydo4uMnKUFi4yapouMmKiLjJepi4yUqouMk6uLjJCtj66Mr4qvh66GrYuMg6uLjIKqi4x/qQWL"
    "jH6oi4x8pouMeqWLjHmjiox4oYqMd6CKjHWeiYx1m4qLioxzmoiMc5eKi4mMBXGUh4xxkoaMcI+GjG+Mh4tvioaKcIeGinGEh4pxgoiKc3+IinN8iYp1e4mK"
    "dXgFiop3doqKeHWKinlzi4p6cYuKfHCLin5ui4p/bYuKgmyLioNri4qGaYqLiGiKZwXGrhWOrJGskqsFlKqWqJiomaWbpJyjnaCfn5+coJqhmKGWopOhkaKO"
    "oo2iiaKIoYWig6GAoX6gfAWfep93nXacc5tymXGYbpZulGySa5FqjmqMaIpoiGqFaoRrgmyAbn5ufXF7coaEBYqKB4qMBjzaWXnuKIyKBYqKBnl5d3p2fHV+"
    "dYB0gwV1hXSIdIl0jXSOdZF0k3WWdZh2mnecd595oHqje6R9pX6ogKiCqoSrhayIrIquBQ75+t0WxffHjIz3mAaMBpcGjAb33/vRvZ370PfCBYwHjIwHl40F"
    "joulkI2MpJGNjKKTjYyhlI2Mn5aNjJ2XjYybmI2MmpqMjJmajI2Xm4yMlZ2MjAWUnouMkp+MjJCgi4yQoo2jjKSKpImji4yGoYuMhqCKjISfi4yCn4qMgZ2K"
    "jH+cBYqMfZuKjHyaiox6mYmMeZeJjHeWiYx2lYiMdZOIjHOSiItxkYiLcI+Ji26NiIwFbQaKBvu2ewaKigdv/fAG9+753hWliaSHooeghZ+DnoOdgZuAmn+Z"
    "fph8lnuVepN5kneRd491jXWMc4p0i4oFiXWHdoV2hHiDeYF7i4qAfH59fn58f4qLe4B6goqLeYN3hIqLdoV0hnOIcYluigX7l4yK+IeMjPeXBg756dv3AhWX"
    "gIyLmICYgYyKmYGMi5mCi4qag4yKmoOMigWbhIyKm4SMi5uEjYuchYyKnIaNi52GjIuehoyLnoiNip6IjYufiYyLoImMi6CKBY0GoAaNBqcGjQaljY2LpY2N"
    "i6OPjYujj42MoZCNjJ+Rjoyeko2MBZ2TjYyblI2MmpWNjJmWjIyXloyNlpeMjZSYjIySmYyNkZqLjI+bjIyNnIuMjJwFjAeMB4qdi4wFiZyLjIqMh5uLjYWb"
    "ioyEmoqMi4yCmYqMgZmKjX+YiYx+l4mMfJeKjHqWiox5lgWJjAZ3lIqMdpSJjHWTBYqLc5OJi3OSiYxxkYmLcJGKi26QcI9xkHOQdJB2kHeReJF6kXyTiot9"
    "kn6Uf5QFgZWCloqLhJiKi4WZhZqHnImdip+Mn4uMjZ6PnpGckpyTm5WZi4yWmJeYmZaZlgWblJyTi4ydkp6Rn5Cgj6KOoo2kjJuKm4ubiZqJmomaiJmHmYeZ"
    "h5mGmYWYhJiFBYuKmISYg5eCmIKXgZeAloCMi7ydf5eKi3+Xiot+ln2Vi4x9lIqLfpSKjH2TiosFfZOKi3ySiox8kYqMe5CKjHyQiYx7j4qLe4+JjHuOiYt6"
    "joqLeo2JjHqMiYt5jAWKBnkGiQZwBogGcYmJinKIiItzhomKBYqLdYaIinaEiIp3g4mKeIGJinqBiYp7f4mKfH6Kin19iop/fIuKgHuKioJ6i4oFg3mLioV5"
    "i4qHd4uKiHeLiop1jHaLio53i4qPeYuKkXuMiZJ8jIqUfYyKlX6MiwWMiouKl4CMipmAjYqZgY2KnIKNipyDjYqehI2Ln4SMiqGFjIuihYyLo4aMi6SGBaaG"
    "poanh6WFpIWihaGEoIOeg52CnIKagJmBl3+Wf5V/i4qUfpF9jIuQfI97jXoFjHqKe4uKiXyHfIZ9hH6Df4GAgYF/gouKfYN9g3uEeoV5hXeGdoh1h3OJcopx"
    "igV3jHiMeIx5jXmOeY57j4qLe497kHyRiot8kX2RfJJ9k36TfZN/lH6Vf5V/llt3BQ75yNH53xX3yoqM/d7F+d6MjPfKrf08aQYO+d7d99kVjGuNbYuKj2+M"
    "ipBwk3KLipRzi4qWdYuKl3cFjIqYeIyKmnqNipt7jYqdfYyJn3+NiqCAjoqhgo6Ko4SOiqSFjoulho+LpomOigWpBo0GqQaOjKaNj4ulkI6LpJGOjKOSjoyh"
    "lI6MoJaNjAWfl4yNnZmNjJubjYyanIyMmJ6MjJefi4yWoYuMlKOLjJOkkKaMjI+ni4yNqYyrBfirUfyrB4psiW4Fh2+FcYRzg3SBdn94f3qLin18fHx7f4qL"
    "eoB5gXeEdoR1hnSHiotziYqLcYpxjAWKi3ONiot0j3WQdpJ3knmVepaKi3uXfJp9mouMf5x/noGgg6KEo4Wlh6eJqIqqBfirUQcO+gDR+ewV9+j98IyJjYmN"
    "iY2JjomOio+Kj4oFjgaPBo8GjwaPjI6MjoyOjY6NjY2MjYyN9+j58FOT+8v9pgWKBooG+8v5plODBQ4cBIfN+e0V92b974uKjImMiY2JjYmNioyLjYmPio6K"
    "j4oFjgaPBo8GjwaOjI6MjIuOjI2Njo2NjIuMjY2MjfeK+NkFjAaMBveK/NmMiYyJi4qNio6JjomOio6Kj4oFkgaPBo8GjoyPjI6Mjo2NjI6NjI2NjYuO92b5"
    "71KRi4r7UP2WBYqKBoqM+4L4xoqNi4yJjYmMiI2JjYqLiIyIjIeMBYQGhwaHBoiKiouIioiKiImJiYqLiYqKiYuKion7gvzGBYqKioyKBvtQ+ZaLjFKFBQ75"
    "89GSFb99jIv3x/gfBYwGjAb3yPwfv5mLi/vc+DoFjAeMB/fc+DqLi1eZ+8j8HwWKBooG+8f4H4qLV3333fw6BYoHigf73fw6BQ75/dH56RX35fxIBYoH/DTF"
    "+DQHjIz35fhIVpn7zfwoBYqKB4qMBvvM+ChWfQUO+ff3CPnfFfjuBoyKi4r9GP3WiYmKiQWJB4gHjImMiY2JjYmNiY6JjoqPio6KBY8Gjwb5FK385AaKjIuM"
    "+Rj51o2NjI0FjQeOB4qNio2JjYmNiY2IjYiMh4yIjAWHBocG/R5pBg738Ov6KRWHBocGiIqIioqLiIqJiYqLiYqJiYmJiomLioqJBYkHHPtQB4kHjImLioyJ"
    "jYmNiY2KjIuNiY6KjIuOio6KBY8Gjwb3Ta37L4yKHASMjIz3L60GDvl33foTFfic/pDClfyc+pBUgQUO9/DO+ikVafcviowc+3SKivsvafdNB48GjwaOjI6M"
    "jIuOjI2NjIuNjI2NjY2MjYuMjI0FjQccBLAHjQeKjYuMio2JjYmNiYyKi4mNiIyKi4iMiIwFhwaHBvtNBg75h934rhW+e/eH95mMjAWMigb3iPuZvpuLjPui"
    "97WLjImMiouJjYiNiIyIjIeMBYcGhwaHBogGh4qIioeKiYmIiYmKi4r7ovu1BQ75fN0uFWn42K382AcO98TO+eoV9wf7UsGXi4v7BvdSiotVfwUO+c7f98AV"
    "jHKLio5zi4qQdIuKknSLipN1BYyKBpV2lnaMiph4jIqZeYyKBZt6jIqbfI2KnXyMip5+jYqfgI2KoIGOiqGDjoqhhI+KooaPi6OIjoukio+LpIwFjoujjo+L"
    "opCPjKGSjoyhk46MoJWNjJ+WjYyemIyMnZqNjJuajIybnIyMmZ2MjAWMBoyKBvsHxfjsUfsHB4qKB4oGiox9nYqMe5yKjHuaiYwFeZqKjHiYiYx3lomMdpWI"
    "jHWTiIx1koeMdJCHi3OOiItyjIeLcoqIi3OIh4t0hgWHinWEiIp1g4iKdoGJineAiYp4foqKeXyJint8iop7eoqKfXmKin54ioqAdoF2BYqKB4N1i4qEdIuK"
    "hnSLiohzi4qKcgXGoxWOoo+hi4ySoJOgi4yUn5ael52ZnJqbmpqMi5uYnZedlZ6UnpKfkZ+PBZ+OoIygip+In4efhZ6EnoKdgZ1/m36bfJp7mXqXeZZ4lHeL"
    "ipN2knaLio91jnQFjHOKc4h0h3WLioR2g3aLioJ3gHh/eX16fHt7fHt+eX95gXiCeIR3hXeHd4h2igV2jHeOd493kXiSeJR5lXmXe5iKi3yafJt9nH+dgJ6C"
    "n4uMg6CEoIuMh6GIooqjBQ758uH3wBV3B4oHjHiMio14i4qPeYuKkXiReYyKknmLipR6lXoFjIoGlnyLipd8jIqXfYyKBZl9jIqZfoyLm3+MipyAjIqcgY2K"
    "nYKNip2DjYqfhI2Ln4WNip+Hjoufh46Ln4kFjgafBo4GnwaOBp6Njoyfjo2Ln4+NjJ6QjYyekY2MnZONi52UjIyclY2Mm5WMjJuXjIyamAWMjJmYjIyZmouM"
    "mJuXnIuMlZyMjJSci4ySnYyLkZ2LjJCejIuOnoyMjZ2LjI2eBYwHnQeMB4qei4yInYuMh56HngWKjAaFnYSdioyDnIqMgpyKjICcfpuLjH6aiot+mYqMBX2X"
    "iox8l4qMe5WKjHuViYx7lImMepKJjHqSiYx5kIiMeZCJi3iPiYt4joiLeI0FiAZ4BokGdwaJineKiYp4iIiLeIcFiYp4hoiKeYWJinmDiYt6gomKe4KJinuB"
    "iop8gIqKfX+Kin1/i4p+fYqKf32GhAWKBoqMBvgYUfzEB8adFY2djIuOnI+ckZySm5KblJqUmpaZlpmXl5eXmZaZlgWZlJqUm5KbkpuRnI+bj4yLm42bjZyM"
    "m4qbipuJm4ebh5uGm4WahJqDmoKZgZmBBYuKmICYfpd9l3yVfIuKlHuTe5J6i4qRepB5jnqLio55jHmLeYl5iXmHeYZ5hXoFhHqCeoJ7gHt/fH99fX59f32A"
    "fIKKinyDe4N7hHqFe4d6h3qIeop6inqMeox6jgV5j3qQepF6knuTe5R8lYuMfZZ9louMfpd/mYCZgZqBm4ObhJyFnIadh52InoqeBZ4HDvmf+Rr4fxW9nYKU"
    "ioyClIqLgpSKjIGTiouBk4qMgJKAkoqMgJGJjICQiox/kIqMf5CJi3+PBYyKB36Piot+j4mLfo6Ki32Oiot9jYmLfoyJi32MBYkGfQaIBnCKh4txiIiLcYaI"
    "i3KEiIpzg4iKBXWCiouJinV/iop2f4mJeH2KioqLeXyKint6iop7eYqKfniKin93ioqBdouKgnUFiooHhXSKioZzi4qIc4uKinGMcYuKjnOLipBzjIqRdAWM"
    "igaUdYuKlXaMipd3jIqYeIyKm3mMigWbeoyKnXyMi4yKnn2NiaB/jIqhf42KjIuhgo6Ko4OOiqSEjoulho6LpYiPi6aKBY4GmgaMBpqMjYuZjI2LmY2Ni5iO"
    "jYsFmI6Ni5iPjYuXj42Ll5CNi5eQjIyXkIyMl5CMjJaRjIyWkoyLlpKMjJWTjIuVkwWMjJSUjIuUlIyMlJRZnYKBgoOCg4KDgYSBhYGEgYaBhoCGgYeAh3+H"
    "gIl/iICJBX6Jf4p+i36Kc4x0jnSPdZCLjHaSdpN3lniWeZh6mnyafJx+nYuMgJ6Bn4KghaEFhqKIo4qjjKOOo5CikaGUoJWflp6LjJidmpyampyanZielp+W"
    "oJOgkouMoZCijwWijqOMl4qXi5eKl4mWiZeJloiVh5aHloeVhpWGlYaMi5SElYWVhJSDlIOUg5SBBQ75zvlA+fAV/BAHiooHigaKjIqMgJmKjH+Ziox+mIqL"
    "fpeKjHyXBYqMBnuViox7lImMe5SJjHmSiYx5kYmMeZCIjHmPiIx4joiLeY0FiAZ4BogGeQaIBnmJiIt5iIiKeYeJi3mGiYp5hYmKeoSKinqDiop7gomKfIGK"
    "inyAiop8fwWKigd+foqKfn2Lin99ioqAe4qKgXqCeouKg3mEeYuKhXmHeIh4i4qJeYuKingFeAeKB415i4qOeYuKj3iQeYuKkXkFjIoGknoFiowHk3qLipV6"
    "jIuWepd7jIqYfYuKBZl9jIqZf4yKmn+MipuBjIqcgYyLnIKNipyEjYqdhY2KnYaNip6HjYueiI6KnYkFjgaeBo4GngaOBp6Njoyejo6Lno8FjYyekY6LnZKN"
    "jJ2TjIyMi5yUjIyblI2MmpaMjJqXjIyZl4yMmJmMi5eajIuUlwWMjIqM+wPF+fBRB/zXBIl4iHmIeYZ5hXqEe4N7gnuBfIB9gH5+fn5/foB8gouKBX2Di4p7"
    "g3yFiop7hnuGe4d7iHuKe4p7jHuMfI57jnuQfJF8kYqLfZOKi32TfJUFfpZ9l3+Xiot/mYCagZqBm4OchJuFnYaciJ2InYmdi52MnY2djp2PnJCdkZyTnAWT"
    "m5WblZqXmZeYl5eZl5iVmZSalJqSmpGakZqPmo6Mi5qOmoybjJqKjIuaipuIBZuIm4aahYyLmoWag5qCmYKYgJiAl3+XfpZ9lX2UfJN8knqRe5F6j3mOeY15"
    "jHkFDvmm9yH3rRWMB4yMB/irBo8GjwaOjI+MjowFjo2NjY2Mi4yNjYyNjI2LjYqni4yJpYuMh6WGo4qMhaKLjIOgioyCn4uMgJ6LjAV/nYqMfpuKjH2aiox8"
    "mImMepeKjIqLepaJjHiUiIx4koeMd5GHi3aPh4x1jYeMBXQGhwZxioiLcoiHi3OGiIpzhQWIinWDiIp2gYiKd4CJinh+iYp5fYqKenuKint7i4qKin16iop+"
    "d4uKf3eLioF2BYuKg3WKioV0hnOLiohyinGMcY5yi4qQc5F0jIqTdYuKlXaLipd3i4qYd4yKmXoFjIqLipt7jIqce4yKnX2Nip5+jYqfgI6KoIGOiqGDjoqj"
    "hY6Ko4aPi6SIjouligWOBpkGjQaYjI2LmYyMi5mNjYuYjoyLmI6Ni5iPBYyLmI+Mi5iQjIuXkIyMl5CMjIyLlpCMjJeRjIyWkoyLlZKMjJaTjIuVk4yMlZMF"
    "i4yVlIuMlZRZnYKCgoKBg4KDgYSBhYGFgIWBhoCGgYeAh4CHgIh/iYCJgIl/igV/i3+KdYx1jnaPd5B3kniUiot5lXmXeph8mXybfpyKi3+dgZ6BoIOghaGH"
    "ooijBYy0FYqMBpEHjqOPopGhk6CVoJWel52Mi5icmpuamZyYnZedlYyLnpSfkp+QoI+hjqGMn4oFnomdh4yLnIechYuKm4SagpqBmH+Mi5d9jIuWfIyLlnuU"
    "eZR3k3eRdZBzj3KNcQWCB4qKBw74WsX46xVp1oqMiQeKcAX8q8X4qweMpYuOjIwF9z2t+zkGiowGjAeMlI+hkKGSn5Oek5yLjJWblpqWmJeXBZiVi4yYk4uM"
    "mZKZkZmQmo6bjZ2Mh613i4eKd4mHiniHiIp5hYiKeoSIiXuDiYkFe4GKinx/iop+foqJfn2Lin97i4qAeouKgniLioN4ioqEdouKhnWKi4dziX6KigU9Bg75"
    "zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiMipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qiho+Lo4iOi6SKj4ukjAWO"
    "i6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qMjJucjIyZnYyMBYwGjIoGWQeKbIluh2+FcYRzgnSBdoB4fnl+fIuKfH17fwV7gIqLeoF5g4qL"
    "eIV3hnaHdIl0inyMfYt9jIuMfYx+jX6Of41/j3+Pf4+Aj4CQBYCRgZCBkoGRgpKBk4KTg5NZeZWCi4qVg4uKlYOMi5WDjIuWg4yLloSMipaFjIoFl4WMi5iF"
    "jIuXho2KmIaMi5mHjIqZiIyKmYiNi5mIjYuaiIyLm4mMi5uKjIubigWNBpsGjQamBo6MpY2OjKOPjosFo5GOjKGSjoyglI2Nn5WNjJ6YjYycmY2Mm5uMjJmc"
    "jIyZnouMl5+MjJWhjIyTowWMjAaSpJGmi4yPp4yMjamMqwX4q1H7BweKigeKBoqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tz"
    "joiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIFxqMV"
    "jqKPoYuMkqCToIuMlJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5GfjwWfjqCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50BYxz"
    "inOIdId1i4qEdoN2i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdooFdox3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIehiKKK"
    "owUO+brfFsX3zgaSpJOkk6OMi5SilqCWn5eemJyYmpmZmpialpuVnJSckpyQnZCejp6NBYyLn4yhip+Jn4edh52EnISbgpqBmX+Yfph9i4qWfJV5lHmSd5J2"
    "kHSOdI1yjHEF+8rF98oHiqaJpYekhqOEoYuMg6CLjIGei4yAngWLjH+ciox+moqNfJmKjHuYiYx6l4mMeZWJjHeUiIx3koiMdZGIi3SPiIxzjYiMBXIGhwZ0"
    "BoiKdIkFiIt1h4iKdoaIineFiouJineDiYp5gYmKeYCJinp/iop7fYqKfHyKin16iop+egWKigeAegWKBoqMBvgcUf3wBw73Xsn5zxWMiIuIjIiMiIyIjYiN"
    "iYyJjYiOiY2KjYmOiY6KjoqOio6LjooFjgaOBo6MjouOjI6MjoyOjY2NjYyOjY2OjI2NjY2OjI6MjoyOi46MjgWOB44Hio6LjoqOio6KjomOiY2KjYmOiI2J"
    "jImNiI2IjIiMiIyIi4iMBYgGiAaIioiLiIqIioiKiImJiYmKiImJiIqJiYmJiIqIioiKiIuIiogFiAeV/dIVxfjsUfzsBg74F/eL+c8ViAeMiIyIjIiMiIyI"
    "jYmNiQWNiI2JjYqMi42JjomNio6KjoqOi46KkouOjI6LjoyOjI2Mjo2OjY2MjY2Njo2NBY2NjI6Mjo2Oi46MjouOjI6KjouOio6LjomOio6KjomNiY2JjomN"
    "iYyIjYiNiYwFiIyIjIiLiIyEi4iKiIuIioiKiYqIiYmJiouJiomJiYiJiYmJioiKiIqIioiKiAWIB4gH+1X+xRWOaaCLj4yfjY6Ln46OjJ2Qj4yckY6Mm5KO"
    "jJqUjowFmZWNjZiWjYyXmIyMlpmMjJWbjIyUm4uMk52MjJGejIyQn4uMkKGLjI6ijaSMpQX451H85weKcolziHWHdoV3hXmEeoJ7gn2AfoCAf4F/gn6Eiot+"
    "hHyGfId7iXqJd4oFDvmr1xbF94cGjIz3HOIFjAaMBvgi++m7n/wp9+8FioyMjAb3+fd5YaP8gfvRiouKjAX4yFH98AcO90bHFsX58FEGDhwEqN0WxffOBpCk"
    "kqSSopOilKCVn5Wdl5yWm5iZmJiYl4yLmZWZk4uMmpKakJqQm44Fm42cjJyKm4mZiIyLmYeYhpiEl4OXgZaAln+UfZR7k3qSeZB3kHaPdI5zjHKMcAX7z8X3"
    "zgeQpJKkkqKTopSglZ+VnZeclpuYmZiYmJeMi5mVmZOLjJqSmpCakJuOBZuNnIycipuJmYiMi5mHmIaYhJeDl4GWgJZ/lH2Ue5N6knmQd5B2j3SOc4xyjHAF"
    "+8/F988HiqaLjIqliKSHoouMhqGLjIWfioyEnouMg50FioyCm4qMgZqKjYCYiY1/l4mMfpaJjX2UiI18k4iMepKIjHmQiIx4j4eLeI2HjAV3BocGdgaIineJ"
    "h4p4iIiKBYqLeYWIinmEiIp6g4mKe4GJinuAiop8foqKfX2Kin58iop+eouKf3mAeIqKhoAFiooHiowGiJaKjISei4yDnQWKjIKbioyBmoqNgJiJjX+XiYx+"
    "lomNfZSIjXyTiIx6koiMeZCIjHiPh4t4jYeMBXcGhwZ2BoiKd4mHiniIiIqKi3mFiIp5hIiKeoOJinuBBYmKe4CKinx+iop9fYqKfnyKin56i4p/eYB4ioqB"
    "doJ1i4qCdYuKhHOLioRyhXAFiQf7zwcO+brfFsX3zgaSpJOkk6OMi5SilqCWn5eemJyYmpmZmpialpuVnJSckpyQnZCejp6NBYyLn4yhip+Jn4edh52EnISb"
    "gpqBmX+Yfph9i4qWfJV5lHmSd5J2kHSOdI1yjHEF+8rF98oHiqaJpYekhqOEoYuMg6CLjIGei4yAngWLjH+ciox+moqNfJmKjHuYiYx6l4mMeZWJjHeUiIx3"
    "koiMdZGIi3SPiIxzjYiMBXIGhwZ0BoiKdIkFiIt1h4iKdoaIineFiouJineDiYp5gYmKeYCJinp/iop7fYqKfHyKin16iop+egWKigd/d4qLf3aLioB2i4qB"
    "dIuKgnOLioNxg3AFiAf7zwcO+eLh98AVjHKLio5zi4qQdIuKknSLipR1i4qWdouKl3eLipl4i4qaeYyKm3qMigWcfI2KnXyNip5+jYqggI2KoYGNiqGCjouj"
    "hI6Ko4aOi6SIj4ukio+LpIyPi6SOBY6Lo5COjKOSjYuilI2MoZWNjKCWjYyemI2MnZqNjJyajIybnIyMmp2LjJmei4wFl5+LjJagi4yUoYuMkqKLjJCii4yO"
    "o4uMjKSKpIuMiKOLjIaii4yEoouMgqGLjAWAoIuMf5+LjH2ei4x8nYqMe5yKjHqaiYx5momMeJiJjHaWiYx1lYmMdJSJi3OSBYiMc5CIi3KOh4tyjIeLcoqH"
    "i3KIiItzhoiKc4SIi3WCiYp1gYmKdoCJinh+iYoFeXyJinp8iop7eoqKfHmLin14i4p/d4uKgHaLioJ1i4qEdIuKhnSLiohzi4qKcgXFFoyjBY6ikKGLjJGg"
    "k6CVoJaemJ2ZnJqbm5qcmJ2XnpWelIyLnpKgkZ+PoI6Mi6CMoIoFjIufiIyLn4eghZ6EjIuegp6BnX+cfpt8mnuZeph5lniVdpN2kXaLipB1jnSMcwWKc4h0"
    "hnWLioV2g3aBdoB4fnl9enx7e3x6fnl/eIF4goqLeIR2hXeHiot3iIqLBXaKdoyKi3aOd492kXiSiot4lHiVeZd6mHuafJt9nH6dgJ6BoIOghaCLjIahiKIF"
    "DvnO3/fAFfzExfgHjIyMigeUf4yLl3yMiwWYfYyKmX+Mipl/jYqagI2Km4KMipyCjYqdg42KnYSOi52Fjoqeh46LnoiOip6JBY4GngaOBqAGno2OjJ6OjYue"
    "j42MnZCNjJ2RjYycko2MnJSMi5yVjIyblYyMmpeMjAWZl4yMmZmLjJiZjIyXm5acjIuUnIyMk5yMjJKci4ySnYuMkJ2PnouMjp2LjI2dBYwHngeKnouMiZ2L"
    "jIieh56Ki4adi4yEnYOdi4yCnICci4yAm4qMf5kFjIoHf5mKjAV9mIuMfJeKjHyWiox8lYmMe5SKjHqTiox6komMeZGJjHmQiYt5j4iMeY6Ii3mNBYgGeQaI"
    "BngGiAZ4iYmLeIiIinmHiIp5homKeYWJioqLeoSJinuCiYp7goqKe4EFiooHfH+Kin1/fn6Kin99ioqAfYqKgXyKioJ8ioqDe4qKhHoFiooGhXqFeYuKh3mK"
    "ioh5i4qJeIp4BcacFY2djp2PnZGckZuSnJOalJqVmZaZl5iXl5iWmJYFmZSalJqTmpGbkZuQm46bjpqMjIuajJuKmoqaiJuImoeahZqFmoSagpmCmIGZfwWX"
    "f5d+l32VfJV7k3uTepF6kHmPeo55jXmMeYt5iXmIeYh5hnqFeYR7g3qKi4J7BYF8gHyKi399f399f36Aiot9gXyDfIN8hXyFe4Z7iHyIe4p7inuMe4x7jnuP"
    "e5AFepCLjHyRe5OLjH2Tiox9lH6WfpeKi3+YgJiAmYGagpuDm4SbhZyGnYediZ2JngWeBw75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiM"
    "ipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qiho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qM"
    "jJucjIyZnYyMBYwGjIoG/AvF+fBR+wcHiooHigaKjH2diox7nIqMe5qJjAV5moqMeJiJjHeWiYx2lYiMdZOIjHWSh4x0kIeLc46Ii3KMh4tyioiLc4iHi3SG"
    "BYeKdYSIinWDiIp2gYmKd4CJinh+iop5fImKe3yKint6iop9eYqKfniKioB2gXYFiooHg3WLioR0i4qGdIuKiHOLiopyBcajFY6ij6GLjJKgk6CLjJSflp6X"
    "nZmcmpuamoyLm5idl52VnpSekp+Rn48Fn46gjKCKn4ifh5+FnoSegp2BnX+bfpt8mnuZepd5lniUd4uKk3aSdouKj3WOdAWMc4pziHSHdYuKhHaDdouKgneA"
    "eH95fXp8e3t8e355f3mBeIJ4hHeFd4d3iHaKBXaMd453j3eReJJ4lHmVeZd7mIqLfJp8m32cf52AnoKfi4yDoISgi4yHoYiiiqMFDviyzxbF99kGjKONo4+h"
    "kKCRn5OelJyVnJaal5mMi5iXmJaalZqTm5KbkYyLm5Cdjp2Nn4wF9wit+wkGiQZ1BoiKdYmIi3aHiIqKi3eGiIp4hImKeIOJinqBiYp6gIqKe34Fiop9fouK"
    "iop+fYuKiop/e4qKgHqLioF4g3eLioR2i4qFdouKh3SLiolzi4qKcgX72QcO+YvZwhWWgYyLl4KMipeCmIKMi5iDBYyKBpiEjIqZhIyLBZmEjIuZhYyKmoWM"
    "i5uGjIqbh4yKm4eMi5yHjIuciIyKnImNi5yJjYqdioyLnooFjAaeBowGoQaMBqCMjIufjI2Lno2Mi56OjYucjgWNi5yPjYubj42MmpCNjJmQjYyXkY2MjIuW"
    "kY2Mi4yWkoyMlZOMjJOUjIySlIyNBZCVjIyPlouMjpaLjYyWi42KmIuMiZeLjYeXi4yHloqNhZWLjYSVioyDlYqLio0FgpSKjICUiox/lImMfpOJjH2SiYx8"
    "komMepKKi3mRiox3kYqLdpGKi3aRiYt1kAWKi3KQco90kHaQdpB4kHmQe5F8kX2Ri4x/kX+SgZODk4qLhJOElYaVhpWIl4mXBZkHmgeOmo6ZkJiSmJKXlJaV"
    "lpWUjIuWlJiUmZKZkpuRBZuQnI+cjp6OnoyfjJaKl4uWipaJjIuVioyLlYiWiZWIloeViIuKlYeUhoyLlIYFlIWVhZOEjIuThJSDk4OTgr6bg5WKi4KUi4yB"
    "k4uMgZOKjIKSioyAkoqLgZKKiwWAkYqMgJCKjH+Qiox/j4qMf46JjH+Oiox+joqLfo6Ki36NiYt+jYmLfoyKi32MBYoGfQaJBnUGiQZ1iYmLdoiIi3eHiYp4"
    "h4iKeYWJi3mEiYp7gwWJinuDiYp9goqKiop+gYqKf4CKioB/ioqBf4qKg32KioR9i4qFfYuKh3yLiYh8BYoHewd7B4oHjn2Lio5+jIqQf4uJkoCLipOAjIqU"
    "gYyJBZaCjIqXgoyKmIONipmDjYqahI2Lm4SNip2FjIuehYyKoIaMiqCGjIuihoyLooYFjIukhqKHoYaghp6GnYWchZqFmYWYhJeElYOMi5SEi4qTg5KDkYKL"
    "ipCCj4GLigWOgY1/jH+KgYmBiIOLioaEi4qGhISEhIWChYKGgIZ/hn+HfYd8iHuJeoh5iniJBXiLd4p6jHqLe417jXuNfI58jn2PfY99kH2QfpF+kX6Sf5J/"
    "kn+Tf5OAlICUXHcFDviGx/j9FWn3EYqM/D8HeweNfIuKjX6Lio5+jIqPf4uKkICLipGAjIqRgoyJBZOCjIqTg4yKlIOMi4yKlYWNiZaGjYmXh42KmIaNi5iH"
    "jIuNipmJjoqZiY6LmooFjgabBowGwq1VBnyMBX6Mf4yAjYGOgo6Ki4OPiouDkIOQhJGEkoWShZSGlIaViJaLjIiWi4yJl4uMiZgFmgf4P4yM9zmt+zmMivc2"
    "Ufs2ior7EQcO+bDf97YVjG6NcIuKj3GRc4uKknSLipN2jIqVeIuKlnkFjIqLiph7jIqZe4yKmn2Nipt/jYqdgI6KnoGNiqCDjoqghY6KooaOi6OHjouliAWN"
    "BqYGjQamBo0GpY6Oi6OPjouikI6MoJGOjKCTjYyelY6MnZYFjYybl42MmpmMjJmbjIyYm4uMjIyWnYuMlZ6MjJOgi4ySoouMkaOPpYuMjaaMqAX3ylH7ygeK"
    "b4lxh3KGdIR1g3eCeIF6f3x+fX1/i4p9gYqLfIF6g3mEeIV3h3aIdYkFc4pzjHWNdo53j3iReZJ6k3yViot9lYuMfZd+mX+agZyCnoOfhKGGooekiaWKpwX3"
    "ylEHDvnD0fjnFffK/OyMiY2JjomNiY6Kj4mOi4+KBY8GjwaOBo+Mj4uOjY6Mjo2NjY2NjI33yvjsVJX7rfy0BYqKiowG+674tFSBBQ4cBHPN+OgV92v87IyJ"
    "jYmMiYyLjYmOio6JjoqOigWPBo8GjwaPBo6Mj4yOjI6Mjo2NjY2NjI33e/hMBYyMjIoG93z8TIyJjYmNiY6JjoqOio6KjIuOigWPBo8GkgaPjI6Mjo2OjI2N"
    "jY2NjYyN92v47FKT+1P8qQWKigeKjAb7dvhBio2JjYiNiY2IjIeNiIuHjAWHBocGiAaHioeLiImIioiJiYmJiYqJ+3b8QQWKigeKjAb7U/ipUoMFDvmq0fjj"
    "Ffe2+7YFiouKB/u2+7a9efem96YFjIwHjIr3pvumvZ37tve2BYqMBoyMB/e297ZZnfum+6aKigWKjAb7pvemWXkFDvnD0fjnFffF/OyTjAWMBoyKBk/7DIWA"
    "hYAFhYCEgouKhIKEg4SDi4qDhISEg4WDhouKg4eChoKIg4iBiIKJgYqAin+KiouOaQWZBo4GmIyOjJiMjYyYjY2Ml46NjJePjYyWj42MlZCNjJSRjYwFlJKM"
    "i5SSjIyUk4yMk5OMjJOTi4yTlIuMkpWMi5KWkZaMi5GXkZeMi/gA+W5Ulfur/MIFigaKBvuw+MJUgQUO+ZP3K/j9FWn4Y4qMioqKB/yu/M+JiYuKiomKiQWJ"
    "B4kHjIiMiY2JjYmOio2Jj4qOio6KBY8Gjwb4xK38kIyKjIyMBviu+M+NjYuMjI2MjQWNB40Hio6KjYmNiY2IjImNh4yIjIeMBYgGhwb8lwYO+Fb3MPe/FYqM"
    "jIyMi4yMlZCMjJSRjYyTkYyMk5GMjIuMkpKMjJGSi4yMjJCTjIyQlIyMj5UFjIwGjpWMjI6Wi4yOl42Yi4yNmIuMjJkFmwf3dQecB4ycjZuNmouMj5mOmZCY"
    "kJeRlpKVi4ySlJOTBZSSlJGUkJWPlo6WjoyLl4ybjIeteouHinuKh4p8iIeKfYeHin6FiIp/hImKgIQFi4qJioCDiomCgoqJgoGKioSAioqFfoqKhn6Ki4d9"
    "ioqHfYuKiHyLioh7iXqKeQV5B/t1B3wHin2Jfol/iYCIgIeCi4qHg4aDhoOGhYWFhYWEhoSHhIeDhwWCiIKIgIiIioqLiImJioqLiYmJiYqJiomLioqJi4mM"
    "iYuKjImMiY2JjYmMi42KBY6JjIuOipaIlIiUiJOHkoeSh5KGkYWRhZCFkIOQg4+Di4qPgo6AjYCNf41+jH0FfAf7dQd5B4x5jXqOe4uKjnyLio99jIqPfYyL"
    "kH6MipF+BYyKkoCMipSBjImUgoyJloONiouKloSNipeEjoqYhY+KmYePipqIj4qbio+KnIsFj617jH+MiouAjoCOgY+CkIKRgpKDk4SUi4yElYWWhpeGmIiZ"
    "h5mLjImaiZuKnAWcB/d1B5sHipmLjImYi4yJmIiXi4yIloqMiJUFjIoHh5WKjIaUioyGk4qMi4yFkoqMhJKLjIqMg5GKjIORiYyCkYqMgZCKjAUO91TO+8AV"
    "xRwEsFEGDvhWzvoHFZuKl4qXiJWIlYeVhpSFk4STg5OCi4qRgZGAkX+Pfo99jn2Lio58jHuMeox6Bft1B3sHjH2Lio1+i4qNfo1/BYyKBo6ABYuKj4GLipCB"
    "i4qRgouKkYOMiouKkYSMipKEi4qMipOFjIqThY2Kk4WNipSGjYoFjIoGiooHiYoFgoaJioOFiYqDhYqKg4WKiouKhISKioWEi4qKioWDi4qFgouKhoGLioeB"
    "i4qIgAWKigeJf4l+i4qJfouKin0Fewf7dQeKeop6inuIfIuKiH2HfYd+hX+FgIWBi4qDgoODg4QFgoWBhoGHgYh/iH+Ke4qOaZ2Lj4ybjI+Mmo6PjJmPjoyY"
    "kY6Ml5KMi42MlpKLjAWNjJWTjY2UlIyNk5WMjJOWi4ySmIuMkZiQmYuMj5mMjI6ai4yNm4yLjJyMnYydBfd1B5oHjJmMmI2XjpaOlo+Ui4yPk4+TkJORkZCR"
    "jIuRkZGQBZKPk4+Tj5SOlI6Wjo6Mjo2OjI6NjY2MjYyNi4yMjYuNio2LjIqNio2JjYiNiIwFiI2IjICOgo6CjoOPg4+Ej4WQhZGKi4aRhZGGk4eTh5OLjIeU"
    "iJaIlomXipiKmQWaB/d1B4qdip2KnIqLiZuLjIiaioyHmYuMhpmFmIuMhJiLjIOWBYqMg5WKjYKUiY2Bk4mMi4yAkomMiot/koiMfpGIjH2Ph4x8joeMe4yH"
    "jHmLiGkFDvmG3ffJFb15mZmYl5iVi4yXk5eSi4yWkJWPlI6TjJOMk4qTipSIlYeWhouKl4QFl4OLipiBmH+ZfZl9jIqZfoyLmH+Ni5iBjYqYg42JmIWOiZiG"
    "j4qYh5CKmImQigWYBpEGmAaQjJiNkIyYj46MmZCNjZmRjY2Yk42MmJWMi5mXjIsFmZiMjJmZWZ19fX5/foGLin+Df4SLioCGgYeCiIOKg4qDjIOMgo6Bj4CQ"
    "i4x/kgV/k4uMfpV+l32ZfZmKjH2Yiot9l4qLfpWJjH6TiY19kYmNfZCIjH6Phox+jYaMBX4GhQZ+BoaKfomGin6Hh4p+hoiJfoWJiX6DiYp+gYmLfn+Ki31+"
    "iop9fQUO994O91TOTxXF+PZRBg75o/gS+aAV+zkHiooHh4qIi3GGiItyhIiKc4OIigV1goqLiYp1f4qKdn+JiXh9ioqKi3l8iop7eoqKe3mKin54iop/d4qK"
    "gXaLioJ1BYqKB4V0ioqGc4uKiHOLiopxjHGLio5zi4qQc4yKkXQFjIoGlHWLipV2jIqXd4yKmHiMigWbeYyKm3qMip18jIuMip59jYmgf4yKoX+NioyLoYKO"
    "iqODjoqkhI6LpYaOi4+KBYyKBlTFv4yMlgeMBpqMjYuZjI2LmY2Ni5iOjYuYjo2LBZiPjYuXj42Ll5CNi5eQjIyXkIyMl5CMjJaRjIyWkoyLlpKMjJWTjIuV"
    "k4yMlJQFjIuUlIyMlJRZnYKBgoOCg4KDgYSBhYGEgYaBhoCGgYeAh3+HgIl/iICJfol/igV+BoMGiowG+MYHjIwHkgaXBpeKlomWiZaJloiVh5aIlYaVh5WF"
    "BZWGlIWVhJSElIOUg5SDlIG9nYKUi4yBlIGUiouClIqLgZOKi4GTiouAkoqMgZEFiouAkYqLioyAkIqMf4+KjH+PiYx/j4mLf4+Ji3+OiYt+jomLfo2Ji32M"
    "iot9jAWJBoGMivc2UQb9iQSKinyOdZCLjHaSdpN3lniWeZh6mnyafJx+nYuMgJ6Bn4KghaGGogWIo4qjjKOOo5CikaGUoJWflp6LjJidmpyampyanZielp+W"
    "oJOgkouMoZCajoyKBQ75mfch9/8Vw4qMBplCjXUFdwd3B4l4BYl5iHqHe4Z7hXyFfouKhH6Df4N/goCBgYGCi4qBg4CCgIR/g4mJiYmJiYqJiokFiQeIB4yJ"
    "jImNiY2JjoqNiY+KjoqPigWOBo8G+Nit/JYGiowFjAeMB5GQjIyWlQWMjJWWi4yVloyMk5iMi5OZjIuSmYuMkpmLjJGbjIuQm4uMj5yLjI+ci4yNno2fBaAH"
    "oAeMB4mhi4x90AWMjIz3q637soyKB2T3X4mhi6GNn46fj52QnZKbk5uUmZWYlpeYlQWLjJiUmZOakpqQjIubkJ2Ono2fjJiKmIuXiZeJl4mXiJaIloeWhpaG"
    "loWVhZWEBZWDlYOVgpSBk4GUgIuKk3/BmYKXi4yCloqMgpaLjIGVioyBlYqLgZSKjICUiosFf5OKjICSiYx/kYqMf5GJjH6Qiot+kImMfY+Ki32PiYt9jomL"
    "fY2Ji32NiIt9jAWJBnwGiAZ0BoiKdImIi3aHh4p4hoeKeYSIinmDiYp7gQWJioqLfICKin1/iop+foqKgHyKioF8i4qCeouKhHqKioZ4ioqHeIuKiHeLiol2"
    "BYoHdQeMiox0i4qy+1sFioqKWmkHDvma+Mb5MRW9edvbWZ07OwX8rgTbO72dO9tZeQX8QfehFYx4i4qNeYuKj3qLipB6i4qRe4yKknuMipN8jIqUfYyKBZZ9"
    "jIqWfoyKmICMipmAjIqagYyKmoKMi42KmoSOipuFjoqcho6KnYePi52JjooFngaPBp0Gj4ydjY+LnI+PjJyQjYwFnJGNjJuSjYyalI2MmZWNjJiWjYyXloyM"
    "l5iMjJaZi4yVmYyMk5qMjJKbi4ySmwWLjJCci4yPnIuMjZ2LjIyeip6LjImdi4yHnIuMhpyLjISbi4yEm4qMg5qKjIGZBYuMgJmKjH+Yiox/lomMfpaJjH2V"
    "iYx8lImMe5KJjHqRiYx6kIeMeo+Hi3mNh4wFeQaHBngGiIp5iYeLeYeIinqGiIp7hYiKfISJioqLfIKKinyBiooFfYCKin6AioqAfoqKgH2KioJ9ioqDfIqK"
    "hHuKioV7i4qGeouKh3qLiol5i4qKeAXQzxWRmpKak5mUmZWXlpeWlpeVmJOYk5mRmZCZj5mOmY2ajJmKBZqJmYiZh5mGmIWYg4yLl4OXgYyLloCWf5V/lH2T"
    "fZJ8kXyQe4uKjnuNeox5inkFiXqIe4uKhnuFfIR8g32CfYF/gH+AgIqLf4F/g4qLfoN+hX2GfYd9iHyJfYp8jAV9jX2OfY99kH2RfpN+k3+VgJaAl4GXgpmD"
    "mYSahZqGm4uMiJuJnIqdjJ2NnI6bBYwH+wf3qxXbO72dO9tZeQX9KgS9edvbWZ07OwUO+bvd+ekV97j8LwWMigZRior7Umn3UoqMKIqK+1Jp91KKjPtoxfdo"
    "B4yMB/dSrftSBoqMBu4HjIwH91Kt+1IGiowGxQeMB/e4+C9Wmfuf/AyKigWKjAb7oPgMVn0FDvdy3fdIFfx0xfh0UQf5ZAT8dMX4dFEHDvlr9xb3ChWUf4uK"
    "lYCVgIyLlYGLipaCi4qWg4yKloOLigWXhIyKloSMipeFjIqXho2Kl4aNipeHjYqXh42LmIeNi5iIjouYiY2LmYmNi5mKBY0GmQaNBp8GjgaejY6LnY6Oi5yP"
    "jouckI2Lm5GNi5uSBY2MmZKNjJiSjY2Xk42MlpWMjJWVjIyMi5SWjIyTlouNkpeMjJGXi4yPmYyMjpgFjAeMB42Zi4yMmQWMaAeKjAaMB42Oi4yRmQWMjAaQ"
    "mouMj5sFi4yNnIuMjJ2Km4uMiZmLjIiai4yGmIuMhZiLjISXio2DlouMgpeKjICVioyAlgWKjH6Uiox9lIqMfJSKjHuTiox6k4qLeZOJi3iSiox3koqLdpGK"
    "jHWRiot0kXaRBXaReJJ4kXqSe5N9kn2Tf5OAlIGVgpSLjIOVhJaFl4uMh5eImYmaipuMmoyYjIwFjpiPmJCXkZaTlpOVlJSVk5aTl5KXkZmQmZCZj4yLmo6c"
    "jZyMnYyXipaLloqWiQWWipaIlYmViJWHlYiVhpWGlIaVhZSFlISUhJSDlIOUgb2dgpSKjIKUgZSKi4KTBYqMBoGSioyBkoqLgJKKiwWAkYqMgJCKjICQiot/"
    "kIqLf4+KjH+OiYx/jomLf46Ji3+NiYt+jYqLfoyJi36MBYkGfgaJBnYGiQZ3iYiLeIiJi3iHiYt6hoiLBXuFiIp8hYmKfISJin6DiYp+goqKf4GKioGBioqB"
    "gIuKgn+LioR+ioqFfouKhn0Fi4qIfYuKiXyLiop7jHqLio17i4qPfIuKkH2LipF9jIqTf4uJlICMiZWAjIqXgQWMipeBjYqZgoyKmoKMipuDjYqcg4yLnYON"
    "i56EjIqfhYyKoYWMi6GEjIuihaCFBZ+FnoWdhJyEmoSag5iDmIKWg5aBlIKTgZKAkoCQf49/i4qOfox+i4qMfYp6iXwFi4qIfYZ9hX6LioV/iouEgIqLg4GB"
    "gYCDi4p/hH+EfoV9hX2Hiot8h3uIeol5igV5iniMeYx5jXuPe458kHyQi4x+kX6Sf5KAlIGUgpWDlYuMhJaGl4aXiJmJmIuMBYqZi5iNl42WjpaOlpCVkJWR"
    "lZGUk5OLjJOTk5OVk5WSlpKWkpiRl5KMi5iRmpAFi4xrp3yFiYt9hIqLfYSKi32Eiop/hIqKf4OKioCDiouAgoqKgoKKi4KBiouDgQWKigeEgYuKhIGLioWA"
    "i4qGgId/ioqIf4uKiX+Liol+BX0HigeMfIuKjXyLio99i4qQfYyKkX6MiZJ/jIqUf4yKlYCMipaAjYqNigWKB4oHiooHd4YF8WUVioyMjIwGmoaOip2HjYqe"
    "iI6Ln4iNi6CJBY0GoQaNBqAGjQafjY6Lno2NjJ6Ojoycj46MnJCNjJuRjoyako2MmZONjJiUjYyXlYyMlZQFjAaMigaKiod/hn+FgISBgoKLioKDi4qBg4GE"
    "BX+Ef4V+hn2GfYd8iHuJeop6in+Mf4uAjX+MgI6AjoCOgI+Bj4CQgZGAkYGSgZIFDvgu95/5rhWMiAaMiIyIjImNiIyJjYiNiY2JjomNio6JjYqOio6KBY6K"
    "jgaOBo4GjoyOBo6MjoyNjI6NjYyOjY2NjY2NjoyNjY6MjYyOjI4FjoyOB44HjgeOio4Hio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmMiIyIjAWIjIgGiAaIBoiK"
    "iAaIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIBYiKiAeIB4gH+1yIFYyIBoyIjIiMiY2IjImNiI2JjYmOiY2KjomNio6KjooFjoqOBo4GjgaOjI4GjoyO"
    "jI2Mjo2NjI6NjY2NjY2OjI2NjoyNjI6MjgWOjI4HjgeOB46KjgeKjoqOio2JjoqNiY6JjYmNiI2JjIiNiYyIjIiMBYiMiAaIBogGiIqIBoiKiIqJioiJiYqI"
    "iYmJiYmJiIqJiYiKiYqIiogFiIqIB4gHiAcO+i7d+EIVjGePaJBpi4qTa4uKlWuXbYuKmG6Mippwi4qccQWMip1zjIqfdIyKoHeMiqF4jYqie42JpH2NiqV/"
    "joqlgo+JpoWPiqeHkIqoio+LBaiMkIynj4+MppGPjaWUjoyll42MpJmNjaKbjYyhnoyMoJ+MjJ+ijIydo4yMm6UFjIwGmqYFjIwHmKiLjJeplauLjJOri4yQ"
    "rY+ujK+Kr4euhq2LjIOri4yBq3+pi4x+qAWKjAZ8pgWMigd7pYqMeaOKjHeiiox2n4qMdZ6JjHSbiY1ymYmMBXGXiIxxlIeNcJGHjG+PhoxujIeLboqGim+H"
    "h4pwhYeJcYKIinF/iYpyfYmJdHsFiYp1eIqKdneKind0iop5c4qKenGLinxwiop+bouKf22Ba4uKg2uLioZph2iKZwXJzxWRrJKrlaqWqJinmqabpJ2inaGf"
    "nqCdoZoFopiilqKTopGjjqONo4mjiKKFooOigKJ+oXygeZ94nXWddJtymnCYb5ZulWySawWQaoyLjmqMaIpoiGqKi4ZqhGuBbIBufm98cHtyeXR5dXd4dnl1"
    "fHR+dIB0g3SFBXOIc4lzjXOOdJF0k3SWdJh1mnadd555oXmie6R8pn6ngKiBqoSrhayIrIqujK4F+Hj7ShWEg4SDg4ODhYSEg4WDhoOGg4aCh4OIg4eDiIqL"
    "g4mDiYqLg4mCigWCioKLgop7jHuNfI59jn2Qi4x+kX6Sf5SAlIuMgJWLjIGXgpiCmoSahZyGnYadBYifip+LjIqgjKCLjIyfjp+QnZCdkZySmpSalJiVl4uM"
    "lpWLjJaUl5SYkpiRi4wFmZCZjpqOm42bjJSKlIuUipSKk4mMi5OJk4mMi5OIk4eTiJSHk4aThpOGk4WShAWThZODkoOSg7+bg5SDlIqLg5OLjIKSi4yCkoqL"
    "g5KKi4KRioyCkYqLgpGJi4KQBYqMBoGPiYuCj4mMgY6JjIGOiYuBjomLgY2Ji4CNiYuAjImLgIwFiQaABogGdwaIiniJh4t5h4iLeYaIigV7hYqLiYp7g4mK"
    "fIKJin2Biop+gImKf3+KiYB+i4qBfYqKgnuLioN7i4qEeYV4BYuKh3iLioh3i4qJdouKinWMdYuKjXaLio53i4qPeIuKkXiSeYuKk3uLipR7jIoFlX2LipZ+"
    "jImXf42KmICMipmBjYqago2Km4ONioyLm4WOip2Gjoudh4+LnomOigWfBo4GlgaNBpaMjYuWjI2Llo2Ni5WNjYuVjo2LlY6NjJWOjYyUj42LlY8FjIwHlJCN"
    "i5SRjIuUkYyMlJGMi5OSjIuUkouMlJKLjJOTjIuTlJOUV5sFDviK0/j2FX8HjIqMf4uKjoCLio6Ai4qQgYuKkIGLipGBjIoFkYKMipKCjIqTg4yKk4ONipSE"
    "jIqVhYyKlYWOipWHjoqWh42Kl4eOi5eIjouXiQWPBpcGjwaXBo8Gl42Oi5eOjouXj42Mlo+OjJWPjYyMi5WRjIyVkYyMk5EFjAaMigZ1xfe2UXUHiooHigaD"
    "kYqMgZGKjIGRiouJjIGPiIyAj4mMf4+Ii3+OiIt/jQWHBn8GhwZ/BocGf4mIi3+IiIt/h4mKgIeIioGHiIqBhYqKgYUFioqChImKg4OKioODioqEgoqKhYKK"
    "ioWBi4qGgYuKhoGLioiAi4qIgIuKin+KigV/B8WWFY2WjZWOlY+Vj5SRlJCTkpOSkpKSkpGMi5KQjIuTj5OPk46UjpOMBZOMlIyTioyLk4qTipSIk4iTh5OH"
    "jIuShpOFkoSShJKDkIORgo+Cj4GOgY2BjYAFgAeAB4mAiYGIgYeBh4KFgoaDhIOEhISEg4WEhoqLg4eDh4OIgoiDioOKBYqLg4qCjIOMg4yCjoOOg4+Dj4qL"
    "hJCKi4SRhJKEkoSThpOFlIeUh5WIlYmViZYFlgcO+TH31vfDFYkHiQeJB4yJjYn3cPusv5n7afekBYqMBoyMB/dp96RXmftw+6yJiQX7j4kViQeJB4kHjImN"
    "ifdw+6y/mftp96QFiowGjIwH92n3pFeZ+3D7rImJBQ75Xd33/xX4foqM+1TF92YGjQeKjYuMio2JjYmNiYyKi4mNiIyKi4iMiIwFhwaHBvycaQYO+Krs98cV"
    "afforfvoBw76Lt34QhWMZ49okGmLipNri4qVa5dti4qYboyKmnCLipxxBYyKnXOMip90jIqgd4yKoXiNiqJ7jYmkfY2KpX+OiqWCj4mmhY+Kp4eQiqiKj4sF"
    "qIyQjKePj4ymkY+NpZSOjKWXjYykmY2NopuNjKGejIygn4yMn6KMjJ2jjIybpQWMjAaapgWMjAeYqIuMl6mVq4uMk6uLjJCtj66Mr4qvh66GrYuMg6uLjIGr"
    "f6mLjH6oBYqMBnymBYyKB3uliox5o4qMd6KKjHafiox1nomMdJuJjXKZiYwFcZeIjHGUh41wkYeMb4+GjG6Mh4tuioaKb4eHinCFh4lxgoiKcX+JinJ9iYl0"
    "ewWJinV4iop2d4qKd3SKinlziop6cYuKfHCKin5ui4p/bYFri4qDa4uKhmmHaIpnBc/wFZKrlaqWqJinmqabpJ2inaGfnqCdoZqimAWilqKTopGjjqONo4mj"
    "iKKFooOigKJ+oXygeZ94nXWddJtymnCYb5ZulWySa5BqBYyLjmqMaIpoiGqKi4ZqhGuBbIBufm98cHtyeXR5dXd4dnl1fHR+dIB0g3SFc4gFc4lzjXOOdJF0"
    "k3SWdJh1mnadd555oXmie6R8pn6ngKiBqoSrhayIrIqujK6OrAX3Hvu+FcX3O4yM9xSKBvc7+0S9nfss9zUFioyMjAaYjY6Ml46OjJaPjoyWkI2MBZWQjYyU"
    "kY2Mk5KNjJOTjIySk4yMkZOLjZGUi4yQlIuMj5aLjI6Vi4yNl4uMjZYFjAeYB5gHjAeJl4mXi4yIlouMh5WLjIaVi4yFlYuMhZSKjAWEk4qMg5OKjIKSioyC"
    "koqLioyBkYiMgZCKi4mMf4+JjH6OiYx+joiLfY2IjHyMBYkGewaKBvs+BocGhwaIioiKiouIiomJiouJiomJiYmKiYuKiokFiQf8YAf3aPhOFZeKloqViYuK"
    "lImUh5SIk4aShpKFkoWQhIyLkIOPg4yLj4KOgY6BjICNgAV/B38HiYCKgYuKiIKLioiCh4OLioaEhoOFhYWFhIWDh4uKhIeCiIKIgoiAiYCKf4p9igX7H4yK"
    "94OMjPcfBg74UNP5zxVp98Ct+8AHDvh23fl4FYAHigeNgYuKjYGLio6Bi4qPgouKkIKLipCCjIqQgwWMipGDjIqShIyKkoSMipOFjYqThY2Kk4aNipSHjIuN"
    "iZSIjoqViI6KlomOi5aJBY4GlgaPBpYGjgaWjY6Llo2OjJWOjoyUjo2NlY+NjJOQjYyTkQWNjJORjIySkoyMkpKMjJGTjIyQk4yMkJSLjJCUi4yPlIuMjpWL"
    "jI2Vi4yMlYyMBZYHlgeKjIqVi4yJlYuMiJWLjIeUi4yGlIuMhpSKjAWGk4qMhZOKjISSioyEkoqMg5GJjIORiYyDkImMgY+JjYKOiIyBjoiMgI2Ii4CNBYgG"
    "gAaHBoAGiAaAiYiLgImIioGIiIqCiImJiouCh4mKg4aJigWDhYmKg4WKioSEioqEhIqKhYOKioaDioqGgouKhoKLioeCi4qIgYuKiYGLiomBBYoHgAfFFpUH"
    "jZSMlY6UjpOPlJCSkJOQkpGRkZGSkJGPko+Sj5KNko2RjAWMi5GMkoySipGKjIuRipKJkomSh5KHkYeShpGFkYWQhJCDkISPgo6DjoKMgY2CBYEHgQeJgoqB"
    "iIKIg4eChoSGg4aEhYWFhYSGhYeEh4SHhImEiYWKBYqLhYqEioSMhYyKi4WMhI2EjYSPhI+Fj4SQhZGFkYaShpOGkoeUiJOIlIqViZQFDvlA3ZMV+Jyt/JwG"
    "+CQEafd6ioz7gcX3gYyM93qt+3qMiveBUfuBior7egcO+IrE+XcVpIiPlpCVkJSRlJGTkpOSk5ORk5KUkJSQlZCUj5WPlo6WjZaNloyXjAWWBpsGmoqZiZiI"
    "mIeWhpeGlYWVhYuKlISTg5KCkoKRgZCBi4qPgI6AjX+NfgV+B4AHioCJgYmAiIGHgoeBhoKFgYWChYKEgoODi4qDg4KCgoKBg4GCgYKAg/uA+04FioqKB4qK"
    "BYoHigeKB4oHjIqMigWKjQeMioyKBY0GjAaNigWNBvgCm/vmBoqMjAb3bvdABZaUlZSMi5WUlZSUlJSUlJSTlJKVkpSRlYyLkJSLjJGUi4yPlY+VjpaOlo2W"
    "jJYFjAeWB5kHiZiJmIeXi4yHlouMhpWKjIWVBYuMhJWDlIqMgpOLjIGTgJOKi4CSiot/kYqLfpCKjH2Piot8joqMfI2Ji3uMiowFegaKBn4GigZ+ioqLf4mK"
    "i3+Jiot/iIqLgIgFioqAh4qLgIeLioCGgYWKi4KFiouDhIqKg4SKi4SDiouEgoSChYKLioWBhoCHgAUO+JXE+BMVkoUFi4qThZOFk4WUhpSGlIaUhoyLlIeM"
    "i5WHlYiMi5WIi4qWiYyLlYiMi5aJjIuWiQWMi5aKjIuXioyLl4qYi5uMjIubjIyLmYyMi5qNjIyYjYyMmI6Mi5ePjIyWj4yMBZaQjIuVkYyMlJGLjJOSjIuS"
    "k4yMkZOMjJCUkJWPlo6Wi4yMlouMjJeKmIqXi4wFiZaLjIeWh5WLjIaUi4yFlISUhJOKjIOSgpKKjIKQioyBkIqMgI+KjH+PiouAjgWKjAaMjAeQjIyMlY6M"
    "jJWPjIyUkIyLlJGMi5KRjIySkYyMkZKMjJGTkJOLjJCUjpWMi42VBYyMB42WjJYFjAeWB4wHmAeJl4uMiZaLjIiWi4yHloqLhpWLjIWVBYSUi4yDlIKTi4yB"
    "kouMgZGKjICRiot/kIqMf4+JjH6Oiox9joqLfI2Ji3yMiowFegaKBoEGigaBioqLgYqKi4GKiouBiYqLgomKi4KIiouCiQWKigeCiIqLgoiLioKIi4qCh4OG"
    "iouDhoOGg4WDhYSFBYuKhIWEhKGDkpKRkZKRkpCSkZOPi4ySj5OPk4+Tj5OOk46UjZSNlI2UjZSMlIwFlQaVBpoGmYqZiZiIl4iWh5aGlYWUhZSFi4qThJKD"
    "kYKRgo+Bj4GLio6Bi4qOgIx/BX8HfweKgImBiIGIgoeDhoOFg4WFhYSDhoSGgoaDiIuKgYiBiIGJgIl/in6KBYoGiQaKiomLioqKioqLioqKigWKB4oHigeK"
    "B4oHjIqMioyKjIuMio2LjIoFjQaMBpqKmIqXiZeIloiVh5WHlIaUhpOFkoSShJGDkIOMi4+Ci4qPgo6ABY2AjH+Mf4p/ioCIgYiBh4KGgoWEi4qEhIOFg4WC"
    "hoGGgYeAh3+If4h+iX2KfIoFfAZ/BoCMgIuAjYCMgY2BjYGOgY2Cj4KOgo+Dj4KPg5CDkISQhJCDkYWSiot2ggUO99rO+U4Vv3v3IPc+V5v7IPs+BQ75l9/7"
    "mBXF98gGjIwHjAaVgo2KmIKMiZmDjYqahI2Km4SNipyGjoqcho6LnYeOi56IjoufiQWNBqAGjQahBo4GoY2Ni6COjoyfjwWNjJ6QjoydkY6MnJONi5yUjYyb"
    "lYyLjIybloyMmpeMjJmYjIuZmoyLmJqLjJOVBYwGjIoG+wbF93UHjQeKjYuMio2JjYmNiYyKi4mNh4yIjIeMBYgGhwaHBocGiIqHioiKiImJiomJiYkFioiC"
    "doF4gXiAeYB7f3t+fX99iot+f32AfYF8gnyEe4R7hXqHeod5iYuKeYqKiwV4inmMeYx7jXuPfI59kH2RfpF/koCUgZSBlIuMgpWEl4SXi4yFmIeZiJqJm4qc"
    "BfguUf3wBw75j/gP+gEVcwaHinSJh4t1h4eKdoaIigV2hImKd4OJinmCiYp6gImKe3+Kinx+iop+fIqKf3yKioB6i4qCeoqKg3iLioR4BYuKhneLiod2i4qJ"
    "dYuKinSMdIuKjXWLio92i4qQd4uKkniLipN4jIqUeouKlnoFjIqXfIyKmHyMipp+jIqbf42KnICNip2CjYqfg42KoISOiqCGj4qhh4+LoomPigWjBo0GvYqM"
    "+9bF9+gGjQeKjYuMio2JjYmNiYyKi4mNiIyKi4iMiIwFhwaHBjwGdIx2jXiOeJB5kXqSfJOKi3yUfZZ+l3+YgJmBm4KbhJ2EnYafiJ+LjImgBYqhjKGNoIuM"
    "jp+Qn5Kdkp2Um5WblpmXmJiXmZaalIyLmpOckp2RnpCejqCNoowF90eKjP3exfnejIzHrfvABg73Zs73yhWIB4iMiAeMiIyIjImNiIyJjYiNiY2JjomNio6J"
    "jYqOio6KBY6KjgaOBo4GjoyOBo6MjoyNjI6NjYyOjY2NjY2NjoyNjY6MjYyOjI4FjoyOB44HjgeOio4Hio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmMiIyIjAWI"
    "jIgGiAaIBoiKiAaIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIBYiKiAeIBw73xPdEfRWJfYqLiH+HgYaBhYKFgoSDhIOEg4ODhIMFiooGhIOKi4SCi4qE"
    "gouKhYKKioaAi4qGgIuKiH6Lioh+BYkHgweJB4yDi4oFjISMiY2EjIqOhYuKjIqPhY2Jj4eLio2KkYaNipGHj4mRiI+KkomPipOJj4qTigWQBpMGjwaUBo+M"
    "lIyOi5WNjYyVjo6LlY+Ni5WPi4xrp4uMgoeDiIOJhIqFigWGBoYGhwaHjIiMiIyIjYiOiI6Ij4mPiZGKkYqRi5ONl46Xj5WQlJCUBZGTkZOSk5OTkpOTk5OU"
    "kpSMjJKUi4ySlYuMkZaLjJCXi4yPmIuMjpmLjIybUY0FDveExPmGFaGD2NgFjIoG/Iul+KkHjAeMB4qMi4yKjIqMiYuKjImLiowFiQaKBokGiQaKiomLioqJ"
    "i4qKioolJQUO+IrT+PYVfweMiox/i4qOgIuKjoCLipCBi4qQgYuKkYGMigWRgoyKkoKMipODjIqTg42KlISMipWFjIqVhY6KlYeOipaHjYqXh46Ll4iOi5eJ"
    "BY8GlwaPBpcGjwaXjY6Ll46Oi5ePjYyWjwWOjJWPjYyMi5WRjIyVkYyMlJKNjJOTjIyTk4yMkpSMjJGUjIyRlYuMkJWLjJCVBYuMjpaLjI6Wi4yMl4uMjJeK"
    "l4uMipeLjIiWi4yIlouMhpWLjIaVi4yFlYqMhZQFioyElIqMg5OKjIOTiYyCkoqMgZGKjIGRiouJjIGPiIyAj4mMf4+Ii3+OiIt/jQWHBn8GhwZ/BocGf4mI"
    "i3+IiIt/h4mKgIeIioGHiIqBhYqKgYUFioqChImKg4OKioODioqEgoqKhYKKioWBi4qGgYuKhoGLioiAi4qIgIuKin+KigV/B8UWlgeNlo2VjpWPlY+UkZSQ"
    "k5KTkpKSkpKRjIuSkIyLk4+Tj5OOlI6TjAWTjJSMk4qMi5OKk4qUiJOIk4eTh4yLkoaThZKEkoSSg5CDkYKPgo+BjoGNgY2ABYAHgAeJgImBiIGHgYeChYKG"
    "g4SDhISEhIOFhIaKi4OHg4eDiIKIg4qDigWKi4OKgoyDjIOMgo6DjoOPg4+Ki4SQiouEkYSShJKEk4aThZSHlIeViJWJlYmWBQ75MffW+NEV92r7pAWKB4oH"
    "+2r7pL9993D3rI2NjI0FjQeNB40Hio2Jjftw96xXfQX7jhb3avukBYoHigf7avukv333cPesjY2MjQWNB40HjQeKjYmN+3D3rFd9BQ76OPmqQhWf9wqMBsGY"
    "VQaK99AGjAeKjIuMioyKi4qMBYoGigaKjAWIBooGigaJBoqKiouKioqLior7m/vXBYqKigeKB4oHjIqMioyKjIuMigWMBo0GjIoFjAb3kAaM+woG+3/3FxWK"
    "jIwG9333swWMBoz7tIoG+34G9wT5shVjk/xM/fCzgwX7NPmgFZ2FzM0FjIoG/Dug+FMHjIqMB4yKjIoHioyKi4qMBYkGigaKBokGigaKBoqKiYuKioqKNTUF"
    "Dvoq+Lf3qBWOlI+UkJOPkpGSkJKMi5GRkZGSkJKQjIuSj5OPk46UjpSOlI2UjZWMBZSLlYyZipeKl4mWiZWIjIuUh4yLlIeUhpOFkoWShJKEkIOQgo+Cj4KN"
    "gY2BjYEFgAeBB4qCiYOLiomDBYmCiIOHg4eDhoOGhIaDhYSEg4SEhISDhIuKg4SDhIKEgoP7WvswioqLioqKi4qMigWKjIoHjIqMi4yKBY0GjAaMBo2KBffG"
    "jAaX+7AHigaMjPdL9yWVkgWUk5OSlJOSkoyLkpOSkouMkpKRk5GTkJOQk5CUjpOMi46UjZSMi42UjJSLjIyUBZUHlgeKlouMiZWIlYuMh5SLjIeUBYqLhpSF"
    "lISThJKKjIORi4yCkYqLgpGKi4GQiouAj4qMgI6Ki36Oiot/jYqLfY0FigZ9BooGgAaKBoGKiouBigWKigaBioqLgYiKi4KIiouCiAWKi4KHiouDh4qKg4eL"
    "ioOGiouEhYSFiouFhIqLhYSGg4qLhoOHgoqLiIKKi4iCBfcg+NUVY5P8TP3ws4MF+zT5oBWdhczNBYyKBvw7oPhTB4yKjAeMioyKB4qMiouKjAWJBooGigaJ"
    "BooGigaKiomLioqKijU1BQ76OPmqQhWf9wqMBsGYVQaK99AGjAeKjIuMioyKi4qMBYoGigaKjAWIBooGigaJBoqKiouKioqLior7m/vXBYqKigeKB4oHjIqM"
    "ioyKjIuMigWMBo0GjIoFjAb3kAaM+woG+3/3FxWKjIwG9333swWMBoz7tIoG+34G9wT5shVjk/xM/fCzgwX7NPhpFZGGkYaMi5GGkoYFkoaMi5KHkoeMi5KH"
    "jIuTiIuKk4iMi5OIjIuTiYyLk4iMi5SJlYqLipWKlYqVigWMBpUGjAaVioyMBZgGjAaYjJiMjIuWjYyLl42WjoyLBZWPjIuUj4yLlJCMi5OQjIuSkYyLkpGL"
    "jJGRjIyQkpCTj5OLjI6Ti4yNlIyLjJUFlQeMB5UHjAeKlQWJlYiUi4yIk4uMhpOHk4qLhpKKjIWRi4yEkYORg5CKi4OQiouCj4qLgY+Ki4GOBYqMjAaQjYyL"
    "lI4FjIuTj4yMko+Mi5KQjIuRkIyMkZCLjJGRkJKQkouMjpKMi46Ti4yNk4uMjZSMlAWMB5UHlgeKlYmVi4yIlIuMiJSKi4eUBYaTi4yFk4WSioyEkYuMg5GK"
    "i4ORiouCkIqMgY+Aj4qMgI6Ki3+Niox/jIqLfo0FigZ9BooGggaCigWKBoMGioqDioqLg4qKi4OJg4mKi4OJi4qDiYOIhIeKi4SIi4qEiISHi4qEh4SGhYYF"
    "hYaLioWGhYWdhZCRkZCQkJGPkZCSj5GPkY6Sj5KNi4ySjZKOko2SjZOMkoyTjAWTjJOMk4uYipeKl4qWiJWJlIeMi5OHjIuThpOGkoaShJGEkISQg4+DjoKN"
    "go2BBYyBjIGLgYqCiYKJg4iDh4SHhYeEhoaKi4aGhYaEh4SHhIiDiIOIgomBioKKgIoFigaJBoqKigaKBoqKioqKigWKB4oHigeKB4yKjIqMigWMBowGjIoF"
    "jQaMBpeKloqVipWIlImUiJSHkoeTh5GGkoWQhZCEkISPg46DjoKMgoyBBYyBioGKgYmDiIKIhIaEhoSGhYSGi4qEh4uKhIeDh4KHgoiKi4KJgImAiX+Kf4oF"
    "fgaBBoIGgYyCjIKMg42CjYONg42DjoOOhI6EjoSPhI+Ej4WPhZCFkIWQeoQFDvmv+Cb48RWAB4qBiYGJgouKiIKIgoaCBYuKh4OFgoWChYOKi4SDg4OCg4KD"
    "gYOAhH+Df4R+hHuDi4p8g4qLfYKKi32CiosFfoKKin6Ci4p/goqKf4GLioCBioqBgIuKgYCLioKAi4qDfouKhH6LioV+i4qGfQWLiod8i4qIe4uKinuLiop5"
    "jHaLio13i4qQd5B4jIqSeYuKlHmMipV7i4qXe4yKBZh8jIqafoyKi4qbf4yKnX+Nip2BjYqfgo2KoYONiqKEjYujho6KpIeOi6WJjYoFpwaNBp4GjQadjI2L"
    "nY2NjJyNjYucj42Lm4+NjJuQjYubkYyMmpGNjJqSjIyZk4yLBZmUjIyYlIyMl5WMjJeWjIyWloyMlZeMjJWYi4yUmYyMk5lVl4J9i4qCfoJ/gX8FgYCAgYCC"
    "f4OLin+EfoR+hH6GfYZ9hn2IfIh7iHyKe4p6inOMdI2Ki3aOiox3jwV3kXiSepOKi3uUe5WLjH2Wfpd/mYCZgZuDm4Schp2Hnomeip+MnIybjpqOmZCZBZCY"
    "kZeSlpOWk5aVlZWVlZSXlJeUl5OZlJiTjIuZk5qTjIuYk4yLmJOMi5eTjIwFl5OMi5aUlpSVlJSUjIyTlJOVkpWRlYyMkJSLjJCVi4yPlY6WjpaLjI2Vi4yM"
    "lgWMB5ZRBw7549uPFcSD9wD3vAWMjPf3iowG9wD7vMST+8/58IqNiY2JjYmNiIyIjYiMiIuKjAWIBocGhwaIBoeKiAaHioiJiYqIiYqJiouKiYqJ+8/98AX4"
    "kPfZFYoHiooF+9yMioyMBvc3+FMFjIyMigb3OPxTBfuV+YEV9zn7IK2Z+zn3IAUO+ePbjxXEg/cA97wFjIz394qMBvcA+7zEk/vP+fCKjYmNiY2JjYiMiI2I"
    "jIiLiowFiAaHBocGiAaHiogGh4qIiYmKiImKiYqLiomKifvP/fAF+JD32RWKB4qKBfvcjIqMjAb3N/hTBYyMjIoG9zj8UwUg+Y8V+zn7IK199zn3IAUO+ePb"
    "jxXEg/cA97wFjIz394qMBvcA+7zEk/vP+fCKjYmNiY2JjYiMiI2IjIiLiowFiAaHBocGiAaHiogGh4qIiYmKiImKiYqLiomKifvP/fAF+JD32RWKB4qKBfvc"
    "jIqMjAb3N/hTBYyMjIoG9zj8UwVc+PUVrpn7G/cWiYyJjImMiYyIjH2LiYqIiomKiYqKivsb+xauffcJ9wUFDvnj248VxIP3APe8BYyM9/eKjAb3APu8xJP7"
    "z/nwio2JjYmNiY2IjIiNiIyIi4qMBYgGhwaHBogGh4qIBoeKiImJioiJiomKi4qJion7z/3wBfiQ99kVigeKigX73IyKjIwG9zf4UwWMjIyKBvc4/FMF+6n5"
    "ORWSkpKQkpCRjpCOkIyPjJKLkIqPipGIkYiRhpKGkoSSg5ODkoGMiwWSgoyLkoOMi5OEjIuThYyKk4eNipOIjoqSiI+Kk4qNiqOLjYyTjI6Mk46OjJOOBY2M"
    "ko+NjJORjIuSkoyLk5OTlIyLkpVmlYODhIOEhISGhYaFiIWIh4qGioSLh4wFhoyGjoWOhJCEkISShJOEk4SViouDlIOTiouEkoqLg5GJjISPiYyDjoiMg46I"
    "jAWDjImMc4uJioOKh4qEiIiKg4iJioOHioqDhYqLg4SKi4SDiouEgoqLhIGwgZKTBQ754/cdhxX3APe8BYyM9/eKjAb3APu8xJP7z/nwio2JjYmNiY2IjIiN"
    "iIyIi4qMBYgGhwaHBogGh4qIBoeKiImJioiJiomKi4qJion7z/3wBfiQ99kVigeKigX73IyKjIwG9zf4UwWMjIyKBvc4/FMFI/krFYyIjIiMiY2JjIiNiY2J"
    "jYmNio2JjYqOio2KjoqOipaLjoyNjI6MjYyOjI2NjYwFjY2NjY2NjI6MjY2NjI6LjoyNi5eKjYuOio6JjYqNio6JjYmNiY2JjImNiIyJjAWIjImMiIyAi4iK"
    "iIqJioiKiYqJiYmKiYmJiYmJioiJiYqJioiKiIuJioiLhYyIBftSFokHjIiMiIyJjYmMiI2JjYmNiY2KjYmNio6KjYqOio6KlouOjI2MjoyNjI6MBY2NjYyN"
    "jY2NjY2MjoyNjY2MjouOjI2Ll4qNi46KjomNio2KjomNiY2JjYmMiY0FiIyJjIiMiYyIjICLiIqIiomKiIqJiomJiYqJiYmJiYmKiImJiomKiIqIi4mKiAWF"
    "Bw7549uPFcSD9wD3vAWMjPf3iowG9wD7vMST+8/58IqNiY2JjYmNiIyIjYiMiIuKjAWIBocGhwaIBoeKiAaHioiJiYqIiYqJiouKiYqJ+8/98AX4kPfZFYoH"
    "iooF+9yMioyMBvc3+FMFjIyMigb3OPxTBfuj+TMVjISLioyEjISLio2Fi4qOhY6Fi4qOhYyLjoWMio+GjIuPhoyKj4eMipCHjYoFkIeMi5GHjIuRiI2KkYmN"
    "ipKJjYuSiY2Lkoqhi5KMjouRjY6LkY2NjJKNjIyRjgWNi5CPjYuQj4yMkI+MjJCPjIyPkIyLj5CLjI+Rj5GLjI6RjZGMjIyRjIyMkoySBZMHkweKkoqSioyK"
    "kYqMiZGIkYuMh5GHkYuMh5CKi4eQioyGj4qMho+KjIaPiYuGjwWJi4WOioyEjYmMhY2Ii4WNiIuEjHWLhIqJi4SJiYuEiYmKhYmJioWIiouFh4qLBYaHiYqG"
    "h4qKh4eKioeGiouHhoqKiIWKi4iFi4qIhYiFi4qJhYuKioSKhIuKioQFtZEVjJKMkQWNkI2RjpCOkI6QjpCPj4+Oj4+Qjo+Nj42QjY+Mj4yci4+Kj4qPiZCJ"
    "j4mPiJCHBY+IjoePho6GjoaNho6FjIaMhYyEi3+KhIqFioaIhYmGiIaIhoeGiIeHiIaHh4gFh4mGiYeJh4qHinqLh4yHjIaNh42HjYaOh4+HjoePiJCIkIiQ"
    "iJCJkYmQipGKkgWRBw76M92OFcSF8/fOjIwFjIoGeffIB4yK7fu7po4FjIoGe/c5rfsgjIoHMfelBYwHjIwHla10jIoGcN6LjIyMBfdXrftkjIoG+xb4HQWM"
    "jIz4Iq38RweKBocGh4qIi4eJiIqJiYiKiYmKiYqI+7H98AX3zPmQFYyMB4wGjIr3JfxKBYqKivu3jIqMBw755vlk9yMVgH6Af3+Af4F/gX6CfoJ+g32EBX6F"
    "fYV9hoqLfYZ9h3yIfIh7iXyKe4p8ioqLcIxxj3OQc5J1lHWWd5l3mnmdep8Fe6F9o36li4yAp4Gpg6uFrYeuiLGKsoyyjrGPrpGtk6uVqZani4yYpZmjm6Gc"
    "nwWdnZ+an5mhlqGUo5KjkKWPpoyMi5qKm4qaipuJmoiaiJmHmYaMi5mGmYWYhZmEBZiDmIKYgpeBl4GXgJZ/ln6/m3+Yi4x/l4qMf5eKi36Wi4x+loqLfZWK"
    "jH2UiosFfZSKi3yTiox8komMfJGKjHuQiYx7kIqLepCKi3qPiYt6joqMeY2Ki3mNiYt6jAWJBnkGiAZsioeLboeHi2+Fh4pwg4iKcoCIinN+iYp0fImKdnqK"
    "ind4i4oFiop5doqKe3SKinxyi4p9cIuKf26LioFsg2qLioRpi4qHZ4uKiGWKY4xjjmWLigWPZ4uKkmmLipNqlWyLipdui4qZcIuKmnKMipt0jIqddoyKi4qf"
    "eIyKoHqNiqJ8BY2Ko36OiqSAjoqmg4+Kp4WPi6SIiH6IfoaAhoGFgoWChIOEg4SDg4ODgoSDiosFhIKLioSChYGLioWBi4qHgIuKh3+Liol9i3mMioyDi4qO"
    "hIuKjoWMio+FjIqQhgWMipGGjYqRh42KkoiNipOJjoqSio6KlIqki5WNjYuVjY2LlY6MjJaOjIuVkHSfBYKIgoiDiYOJhIp/i4aMhouGjYeNh42IjoePiI+I"
    "kImRipGKkoyTjZiOl4+Vj5UFkZSRk5KTkpOSk5OTk5OLjJKTjIuSlIuMkpSLjJKVi4yQlYyMj5eMjI6YjIyNmQWMBp0GjQacjI2LnY2Mi52NjIycjo2LnI+M"
    "i5yQjIubkI2Mm5AFjIyakY2MmpKMjJqTjIuZlIyLmZSMjJmVjIuYlouMmJaMi5eXjIyXl4uMl5hXmwUO+Y346fhTFfxgjIr4HoyM+Lqt/NgGhwaHBoiKiIqK"
    "i4iKiYmKi4mKiYmJiYqJi4qKiQWJB/3wB4kHjImLioyJjYmNiY2KjIuNiY6KjIuOio6KBY8Gjwb42K38uoyK+B6MjPhgrQb7x/kLFfc5+yCumfs59yAFDvmN"
    "+On4UxX8YIyK+B6MjPi6rfzYBocGhwaIioiKiouIiomJiouJiomJiYmKiYuKiokFiQf98AeJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+Nit/LqM"
    "ivgejIz4YK0G+zH5GRX7Ofsgrn33OfcgBQ75jfjp+FMV/GCMivgejIz4uq382AaHBocGiIqIioqLiIqJiYqLiYqJiYmJiomLioqJBYkH/fAHiQeMiYuKjImN"
    "iY2JjYqMi42JjoqMi46KjooFjwaPBvjYrfy6jIr4HoyM+GCtBir4fxWumfsb9xaKjImMiIyJjIiMfouIiomKiIqJioqK+xv7Fq599wr3BQUO+Y33HfhTFYyK"
    "+B6MjPi6rfzYB4cGhwaIioiKiouIiomJiouJiomJiYmKiYuKiokFiQf98AeJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+Nit/LqMivgejIz4YK0G"
    "+y34tRWIB4yIjImNiYyIjYmNiY2JjYqNiY2KjoqNio6KjoqWi46MjoyNjI6MjYyNjY2MBY2NjY2NjYyOjY2MjYyOi46MjYuOjI6KjouOio2LjoqOio2JjYqO"
    "iY2JjYmNiYwFiY2JjIiMiYyIjIiMgIuIioiKiYqIiomKiYmJiomJiYmJiYqIiYmKiYqIi4iKiQV/B/tSFoyJi4iMiAWMiY2JjIiNiY2JjYmNio2JjYqOio2K"
    "joqOipaLjoyOjI2MjoyNjI2NjYyNjY2NBY2NjI6NjYyNjI6LjoyNi46MjoqOi46KjYuOio6KjYmNio6JjYmNiY2JjImNiYwFiIyJjIiMiIyAi4iKiIqJioiK"
    "iYqJiYmKiYmJiYmJioiJiYqJioiLiIqJi4iKiAWMiAYO99PH+soV9zn7IK2Z+zn3IAWo/tgVxfnwUf3wBg730/d1+tgV+zn7IK199zn3IAX7Df7KFcX58FH9"
    "8AYO+D333vo+Fa6Z+xv3FoqMiYyJjIiMiYx9i4iKiYqJiomKiYr7G/sWrn33CvcFBW7+rxXF+fBR/fAGDvgQ94/6dBWMiIyIjImMiY2IjImNiY2JjYqOiY2K"
    "jYqOio6KjYqXi42MjoyOjI2MBY2Mjo2NjI2NjY2MjY2OjI2MjYyOjI6MjYuXio2KjoqOio2KjYmOio2JjYmNiYwFiI2JjImMiIyIjImMf4uJioiKiIqJiomK"
    "iImJiomJiYmKiYmIiomKiYqIioiKiQV/BzL+dhXF+fBR/fAGJvp2FYyJjIiMiIyJjImNiIyJjYmNiY2KjomNio2KjoqOio2Kl4uNjI6MjoyNjAWNjI6NjYyN"
    "jY2NjI2NjoyNjI2MjoyOjI2Ll4qNio6KjoqNio2JjoqNiY2JjYmMBYiNiYyJjIiMiIyJjH+LiYqIioiKiYqJioiJiYqJiYmJiomJiIqJiomKiIqIiokFggcO"
    "+gzd+FMVab2KjPwwpweMigZ794QHjQaujI6LrY+Oi6uRj4ypk46MqJWOjIuMBaaXjYykmo2No5uNjaCejYyfoIyMnaKMjZukjIyZpoyMl6iMi5aqi4yUrIuM"
    "kq0Fi4yQr4uMjrCLjIyzirOLjIiwi4yGr4uMhK2LjIKsi4yAqoqLf6iKjH2miox7pAWKjXmiiox3oImMdp6JjXObiY1ymomMcJeLjIiMbpWIjG2Th4xrkYiL"
    "aY+Ii2iMBYkG+4R7BoqKB2/8MIqKWQb3AWcVjIyM91Kt+1KMivgejIz3ZQesiamIqIamhKSBpICifouKoHyMi595nnecdZtzmXGXb5Ztk2uSaZBoi4qOZoxk"
    "BYpkiGaLioZohGmDa4Btf299cXtzenV4d3d5iot2fIuKdH5ygHKBcIRuhm2IaokF+2WMiowGDvne3RbF+ZgGjIwHjAb4xv2gjYmNiY6KjomOio6Kj4oFkgaP"
    "Bo8GjoyPjI6Mjo2NjI2NjY2MjYuMjI0FjQf58FH9mAeKigeKBvzG+aCJjYmNiIyIjYiMiIyHjAWEBocGhwaIioeKiIqIiYmKiYmJiYqJi4qKiQWJB/3wB/d3"
    "+oIVkpKSkJGQkY6RjpCMj4ySi4+KkIqRiAWRiJGGkoaShJKDmnmTgpODjIuThIyLk4WMipOHjYqTiI2Kk4iPipKKjYqki42MBZKMj4yTjo2Mk46NjJOPjIyT"
    "kYyLk5KMi5OTk5STlWWVhIOEg4SEhIaFhoWIhYgFhoqHioSLh4yGjIWOhY6FkISQhJKEk3ydg5SDk4qLg5KKi4ORioyDj4mMg46JjAWDjoeMhIyJjHKLiYqE"
    "ioeKg4iJioOIiYqDh4qKg4WKi4OEiouDg4OCg4GxgZKTBQ76Gt34QhWMZ45ojIuQaYuKk2uLipRsi4qXbYuKmG6Lippwi4qccYuKnXMFjIqedYyKn3aMiqF4"
    "jYqhe42Ko3yOiqN/joqlgo+KpYSQiqaHkIqnio+Lp4yQjAWmj5CMpZKPjKWUjYyMi6OXjoyjmoyMjIuhm42MoZ6MjJ+gjIyeoYyMnaOLjJylBYuMmqaLjJio"
    "i4yXqYuMlKqLjJOri4yQrY+ujK+Kr4euhq2LjIOri4yCqouMf6kFi4x+qIuMfKaLjHqli4x5o4qMeKGKjHegiox1nomMdZuKi4qMc5qIjHOXiouJjAVxlIeM"
    "cZKGjHCPhoxvjIeLb4qGinCHhopxhIeKcYKIinN/iIpzfImKdXuJinV4BYqKd3aKinh1iop5c4uKenGLinxwi4p+bouKf22LioJsi4qDa4uKhmmKi4hoimcF"
    "xRaMro6skaySq5SqlqiYqJmlm6Sco52gBZ+fn5ygmqGYoZaik6GRoo6ijaKJooihhaKDoYChfqB8n3qfd512nHObcplxmG4Flm6UbJJrkWqOaoxoimiIaoVq"
    "hGuCbIBufm59cXtyenN5dnd3d3p2fHV+dYB0gwV1hXSIdIl0jXSOdZF0k3WWdZh2mnecd595oHqje6R9pX6ogKiCqoSrhayIrIquBfdv+RwV9zn7IK2Z+zn3"
    "IAUO+hrd+EIVjGeOaIyLkGmLipNri4qUbIuKl22Liphui4qacIuKnHGLip1zBYyKnnWMip92jIqheI2KoXuNiqN8joqjf46KpYKPiqWEkIqmh5CKp4qPi6eM"
    "kIwFpo+QjKWSj4yllI2MjIujl46Mo5qMjIyLoZuNjKGejIyfoIyMnqGMjJ2ji4ycpQWLjJqmi4yYqIuMl6mLjJSqi4yTq4uMkK2Proyviq+Hroati4yDq4uM"
    "gqqLjH+pBYuMfqiLjHymi4x6pYuMeaOKjHihiox3oIqMdZ6JjHWbiouKjHOaiIxzl4qLiYwFcZSHjHGShoxwj4aMb4yHi2+Khopwh4aKcYSHinGCiIpzf4iK"
    "c3yJinV7iYp1eAWKind2iop4dYqKeXOLinpxi4p8cIuKfm6Lin9ti4qCbIuKg2uLioZpiouIaIpnBcUWjK6OrJGskquUqpaomKiZpZuknKOdoAWfn5+coJqh"
    "mKGWopOhkaKOoo2iiaKIoYWig6GAoX6gfJ96n3eddpxzm3KZcZhuBZZulGySa5FqjmqMaIpoiGqFaoRrgmyAbn5ufXF7cnpzeXZ3d3d6dnx1fnWAdIMFdYV0"
    "iHSJdI10jnWRdJN1lnWYdpp3nHefeaB6o3ukfaV+qICogqqEq4WsiKyKrgX4BfkqFfs5+yCtffc59yAFDvoa3fhCFYxnjmiMi5Bpi4qTa4uKlGyLipdti4qY"
    "bouKmnCLipxxi4qdcwWMip51jIqfdoyKoXiNiqF7jYqjfI6Ko3+OiqWCj4qlhJCKpoeQiqeKj4unjJCMBaaPkIylko+MpZSNjIyLo5eOjKOajIyMi6GbjYyh"
    "noyMn6CMjJ6hjIydo4uMnKUFi4yapouMmKiLjJepi4yUqouMk6uLjJCtj66Mr4qvh66GrYuMg6uLjIKqi4x/qQWLjH6oi4x8pouMeqWLjHmjiox4oYqMd6CK"
    "jHWeiYx1m4qLioxzmoiMc5eKi4mMBXGUh4xxkoaMcI+GjG+Mh4tvioaKcIeGinGEh4pxgoiKc3+IinN8iYp1e4mKdXgFiop3doqKeHWKinlzi4p6cYuKfHCL"
    "in5ui4p/bYuKgmyLioNri4qGaYqLiGiKZwXFFoyujqyRrJKrlKqWqJiomaWbpJyjnaAFn5+fnKCaoZihlqKToZGijqKNoomiiKGFooOhgKF+oHyfep93nXac"
    "c5tymXGYbgWWbpRskmuRao5qjGiKaIhqhWqEa4JsgG5+bn1xe3J6c3l2d3d3enZ8dX51gHSDBXWFdIh0iXSNdI51kXSTdZZ1mHaad5x3n3mgeqN7pH2lfqiA"
    "qIKqhKuFrIisiq4F+ED4kBWvmfsb9xaJjImMiYyJjIiMfYuIiomKiYqJiomK+xv7Fq999wn3BQUO+hrd+EIVjGeOaIyLkGmLipNri4qUbIuKl22Liphui4qa"
    "cIuKnHGLip1zBYyKnnWMip92jIqheI2KoXuNiqN8joqjf46KpYKPiqWEkIqmh5CKp4qPi6eMkIwFpo+QjKWSj4yllI2MjIujl46Mo5qMjIyLoZuNjKGejIyf"
    "oIyMnqGMjJ2ji4ycpQWLjJqmi4yYqIuMl6mLjJSqi4yTq4uMkK2Proyviq+Hroati4yDq4uMgqqLjH+pBYuMfqiLjHymi4x6pYuMeaOKjHihiox3oIqMdZ6J"
    "jHWbiouKjHOaiIxzl4qLiYwFcZSHjHGShoxwj4aMb4yHi2+Khopwh4aKcYSHinGCiIpzf4iKc3yJinV7iYp1eAWKind2iop4dYqKeXOLinpxi4p8cIuKfm6L"
    "in9ti4qCbIuKg2uLioZpiouIaIpnBcUWjK6OrJGskquUqpaomKiZpZuknKOdoAWfn5+coJqhmKGWopOhkaKOoo2iiaKIoYWig6GAoX6gfJ96n3eddpxzm3KZ"
    "cZhuBZZulGySa5FqjmqMaIpoiGqFaoRrgmyAbn5ufXF7cnpzeXZ3d3d6dnx1fnWAdIMFdYV0iHSJdI10jnWRdJN1lnWYdpp3nHefeaB6o3ukfaV+qICogqqE"
    "q4WsiKyKrgX3W/jUFZKSkpCRkJGOkY6QjI+MkouPipCKkYiRiJGGBZKGkoSSg5KDk4GTgpODjIuThIyLk4WMipOHjYqTiI2Kk4iPipKKjYqki42MkowFj4yT"
    "jo2Mk46NjJOPjIyTkYyLk5KMi5OTk5STlWWVhIOEg4SEhIaFhoWIhYiGigWHioSLh4yGjIWOhY6FkISQhJKEk4STg5WDlIOTiouDkoqLg5GKjIOPiYyDjomM"
    "BYOOh4yEjImMcouJioSKh4qDiImKg4iJioOHioqDhYqLg4SKi4ODg4KDgbGBkpMFDvoa3fhCFYxnjmiMi5Bpi4qTa4uKlGyLipdti4qYbouKmnCLipxxi4qd"
    "cwWMip51jIqfdoyKoXiNiqF7jYqjfI6Ko3+OiqWCj4qlhJCKpoeQiqeKj4unjJCMBaaPkIylko+MpZSNjIyLo5eOjKOajIyMi6GbjYyhnoyMn6CMjJ6hjIyd"
    "o4uMnKUFi4yapouMmKiLjJepi4yUqouMk6uLjJCtj66Mr4qvh66GrYuMg6uLjIKqi4x/qQWLjH6oi4x8pouMeqWLjHmjiox4oYqMd6CKjHWeiYx1m4qLioxz"
    "moiMc5eKi4mMBXGUh4xxkoaMcI+GjG+Mh4tvioaKcIeGinGEh4pxgoiKc3+IinN8iYp1e4mKdXgFiop3doqKeHWKinlzi4p6cYuKfHCLin5ui4p/bYuKgmyL"
    "ioNri4qGaYqLiGiKZwXGrhWOrJGskquUqpaomKiZpZuknKOdoAWfn5+coJqhmKGWopOhkaKOoo2iiaKIoYWig6GAoX6gfJ96n3eddpxzm3KZcZhuBZZulGyS"
    "a5FqjmqMaIpoiGqFaoRrgmyAbn5ufXF7cnpzeXZ3d3d6dnx1fnWAdIMFdYV0iHSJdI10jnWRdJN1lnWYdpp3nHefeaB6o3ukfaV+qICogqqEq4WsiKyKrgX4"
    "CPjGFYyIjIiMiYyJjYiMiY2JjYmNio6JjYqNio6KjoqNipeLjYyOjI6MjYwFjYyOjY2MjY2NjYyNjY6MjYyNjI6MjoyNi5eKjYqOio6KjYqNiY6KjYmNiY2J"
    "jAWIjYmMiYyIjIiMiYx/i4mKiIqIiomKiYqIiYmKiYmJiYqJiYiKiYqJioiKiIqJBX8H+1IWjImMiIyIjImMiY2IjImNiY2JjYqOiY2KjYqOio6KjYqXi42M"
    "joyOjI2MBY2Mjo2NjI2NjY2MjY2OjI2MjYyOjI6MjYuXio2KjoqOio2KjYmOio2JjYmNiYwFiI2JjImMiIyIjImMf4uJioiKiIqJiomKiImJiomJiYmKiYmI"
    "iomKiYqIioiKiQWCBw75Xt3aFb1593T3dAWMjAeMivd0+3S9nfuE94QFiowGjIwH94T3hFmd+3T7dIqKBYqMBvt093RZefeE+4QFiouKB/uE+4QFDvoa3fhC"
    "FYxnBY5ojIuQaYuKk2uLipRsi4qXbYuKmG6Lippwi4qccYuKnXOMio+Gi4oi+yXBfeH3DAWMjIyKBqF4jYqhe42Ko3yOiqN/joqlggWPiqWEkIqmh5CKp4qP"
    "i6eMkIymj5CMpZKPjKWUjYyMi6OXjoyjmoyMjIuhm42MBaGejIyfoIyMnqGMjJ2ji4ycpYuMmqaLjJioi4yXqYuMlKqLjJOri4yQrY+ujK8Fiq+Hroati4yD"
    "q4uMgqqLjH+pi4x+qIuMfKaLjHqli4x5o4qMh5CLjPT3JVWZNfsMBYqKiowGdZ6JjHWbiouKjHOaiIxzl4qLiYwFcZSHjHGShoxwj4aMb4yHi2+Khopwh4aK"
    "cYSHinGCiIpzf4iKc3yJinV7iYp1eAWKind2iop4dYqKeXOLinpxi4p8cIuKfm6Lin9ti4qCbIuKg2uLioZpiouIaIpnBfjq96UVjIwFjIoGm3KZcZhulm6U"
    "bJJrkWqOaoxoimiIaoVqhGuCbIBufm4FfXF7cnpzeXZ3d3d6dnx1fnWAdIN1hXSIdIl0jXSOdZF0k3WWdZh2mnecd5+KjAWMB2u2FYqKBYqMBnukfaV+qICo"
    "gqqEq4WsiKyKroyujqyRrJKrlKqWqJioBZmlm6Sco52gn5+fnKCaoZihlqKToZGijqKNoomiiKGFooOhgKF+oHyfep93jIoFigcO+d7d99kVjGuNbYuKj2+M"
    "ipBwk3KLipRzi4qWdYuKl3cFjIqYeIyKmnqNipt7jYqdfYyJn3+NiqCAjoqhgo6Ko4SOiqSFjoulho+LpomOigWpBo0GqQaOjKaNj4ulkI6LpJGOjKOSjoyh"
    "lI6MoJaNjAWfl4yNnZmNjJubjYyanIyMmJ6MjJefi4yWoYuMlKOLjJOkkKaMjI+ni4yNqYyrBfirUfyrB4psiW4Fh2+FcYRzg3SBdn94f3qLin18fHx7f4qL"
    "eoB5gXeEdoR1hnSHiotziYqLcYpxjAWKi3ONiot0j3WQdpJ3knmVepaKi3uXfJp9mouMf5x/noGgg6KEo4Wlh6eJqIqqBfirUfyrB/eL+YUV9zn7IK2Z+zn3"
    "IAUO+d7d99kVjGuNbYuKj2+MipBwk3KLipRzi4qWdYuKl3cFjIqYeIyKmnqNipt7jYqdfYyJn3+NiqCAjoqhgo6Ko4SOiqSFjoulho+LpomOigWpBo0GqQaO"
    "jKaNj4ulkI6LpJGOjKOSjoyhlI6MoJaNjAWfl4yNnZmNjJubjYyanIyMmJ6MjJefi4yWoYuMlKOLjJOkkKaMjI+ni4yNqYyrBfirUfyrB4psiW4Fh2+FcYRz"
    "g3SBdn94f3qLin18fHx7f4qLeoB5gXeEdoR1hnSHiotziYqLcYpxjAWKi3ONiot0j3WQdpJ3knmVepaKi3uXfJp9mouMf5x/noGgg6KEo4Wlh6eJqIqqBfir"
    "UfyrB/gh+ZMV+zn7IK199zn3IAUO+d7d99kVjGuNbYuKj2+MipBwk3KLipRzi4qWdYuKl3cFjIqYeIyKmnqNipt7jYqdfYyJn3+NiqCAjoqhgo6Ko4SOiqSF"
    "joulho+LpomOigWpBo0GqQaOjKaNj4ulkI6LpJGOjKOSjoyhlI6MoJaNjAWfl4yNnZmNjJubjYyanIyMmJ6MjJefi4yWoYuMlKOLjJOkkKaMjI+ni4yNqYyr"
    "BfirUfyrB4psiW4Fh2+FcYRzg3SBdn94f3qLin18fHx7f4qLeoB5gXeEdoR1hnSHiotziYqLcYpxjAWKi3ONiot0j3WQdpJ3knmVepaKi3uXfJp9mouMf5x/"
    "noGgg6KEo4Wlh6eJqIqqBfirUfyrB/hc+PkVr5n7G/cWiYyJjImMiYyIjH2LiIqJiomKiYqJivsb+xavffcJ9wUFDvne3ffZFYxrjW2Lio9vjIqQcJNyi4qU"
    "c4uKlnWLipd3BYyKmHiMipp6jYqbe42KnX2MiZ9/jYqggI6KoYKOiqOEjoqkhY6LpYaPi6aJjooFqQaNBqkGjoymjY+LpZCOi6SRjoyjko6MoZSOjKCWjYwF"
    "n5eMjZ2ZjYybm42MmpyMjJiejIyXn4uMlqGLjJSji4yTpJCmjIyPp4uMjamMqwX4q1H8qweKbIluBYdvhXGEc4N0gXZ/eH96i4p9fHx8e3+Ki3qAeYF3hHaE"
    "dYZ0h4qLc4mKi3GKcYwFiotzjYqLdI91kHaSd5J5lXqWiot7l3yafZqLjH+cf56BoIOihKOFpYeniaiKqgX4q1H8qwf4JPkvFYyIjIiMiYyJjYiMiY2JjYmN"
    "io6JjYqNio6KjoqNipeLjYyOjI6MjYwFjYyOjY2MjY2NjYyNjY6MjYyNjI6MjoyNi5eKjYqOio6KjYqNiY6KjYmNiY2JjAWIjYmMiYyIjIiMiYx/i4mKiIqI"
    "iomKiYqIiYmKiYmJiYqJiYiKiYqJioiKiIqJBX8H+1IWjImMiIyIjImMiY2IjImNiY2JjYqOiY2KjYqOio6KjYqXi42MjoyOjI2MBY2Mjo2NjI2NjY2MjY2O"
    "jI2MjYyOjI6MjYuXio2KjoqOio2KjYmOio2JjYmNiYwFiI2JjImMiIyIjImMf4uJioiKiIqJiomKiImJiomJiYmKiYmIiomKiYqIioiKiQWFB4gHDvn90fnp"
    "Fffl/EgFigf8NMX4NAeMjPfl+EhWmfvN/CgFiooHiowG+8z4KFZ9Bfg894MV+zn7IK199zn3IAUO+c/dFsXOjIz3mAaMBqkGjganjo6Lpo8FjoulkI2MpJGN"
    "jKKTjYyhlI2Mn5aNjJ2XjYybmI2MmpqMjJmajI2Xm4yMlZ2MjAWUnouMkp+MjJCgi4yQoo2jjKSKpImji4yGoYuMhqCKjISfi4yCn4qMgZ2KjH+cBYqMfZuK"
    "jHyaiox6mYmMeZeJjHeWiYx2lYiMdZOIjHOSiItxkYiLcI+Ji26NiIwFbQaKBvuYjIr3clH98AbF+O0VjIyM95cHp4qliaSHooeghZ+DnoOdgZuAmn+Zfph8"
    "lnuVepN5kneRd491jXWMc4p0i4oFiXWHdoV2hHiDeYF7i4qAfH59fn58f4qLe4B6goqLeYN3hIqLdoV0hnOIcYluigX7l4yKjAYO+bHdFsX4yQaMpo2lkKOL"
    "jAWQopOgk6CVnpacl5uMi5iamZial5uVnJOLjJySnZGekJ6On42fjJ+KnomdiJyIBZyGmoWMipmEmoOYgpeAl4CVfoyLlH2TfYyLknuRepB5jniNeIx2i3uJ"
    "fYl9iH0FiH+LioZ/hoCFgIWAhIGDgYKCgoKCgoCCgIN/g3+DfoOLin6DiIqJiYmJiomKiAWJB4kHjImMiYuKjYmNiouKjYqOiY6KnIWbhZqEmYSZhJiDl4OW"
    "gpWCjIqUgpSBBYuKk4GSf5KAkH6Qfo5+i4qOfYx9i4qMfIp6iXuIfYuKh36LioV+hH+EgIKAgoIFgIKAhH+EfoV+hX2HfIeKi3yIeol5inmKgIyBi4GMgYuC"
    "jYKMgo2CjYONg42DjgWDjYOOg4+EjoOPhI+Ej4SQhI+LjF9zk4aMi5KGjIuThoyKk4eMi5OGjIuTh4yLBZSHjIqUiIyLlIeMi5SIjIuViIyLlYiMi5WJjIqW"
    "ioyKlYqNi5aJjIuWioyLlooFjQaWBo0GlgaNBqAGjQafjY6Lno2NjJ6OjYydj42MnJAFjYybkY2MmpKNjJmTjYyYlIyMjIuWlY2MlZaNjJSXjIyUmIuMk5iL"
    "jJGZi4yQmgWMjIwHjpqLjI2ci4yMnYqcipuLjIiai4yHmYuMhpmKjIaYioyEmIOYBYqMg5aKjIGWioyBloqMf5WKjH+Uiox+lIqMfZOKjHyTiox7koqMe5KK"
    "jImLiowFjAeMjAeYlJiUl5SMjJaUjIsFlZWMjJSUjIyUlYyMk5WMjJOWi4ySlouMkpeRmIuMj5eMjI+ZjpmLjI2ai4yNmgWMB5sHiqGJoIuMh56LjIaei4yE"
    "nYuMg5uLjIKbioyBmgWKjH+Ziox/mIqLiox9l4qMfJWJjXuUiYx6k4iMeZKIjHiRiIx3j4iMdo6IjHWNBYgGdAaIBnQGh4p0iYiLdYeIinWGiIp3hIiKd4KJ"
    "iniBiYp5gImKBXt+iYp8fIqKfHuKin56ioqAeYqKgXeKioJ2i4qDdIuKhXOLiodyi4qIcYuKim8F/MkHDvnO3/fAFYxyi4qOc4uKkHSLipJ0i4qTdQWMigaV"
    "dpZ2jIqYeIyKmXmMigWbeoyKm3yNip18jIqefo2Kn4CNiqCBjoqhg46KoYSPiqKGj4ujiI6LpIqPi6SMBY6Lo46Pi6KQj4yhko6MoZOOjKCVjYyflo2MnpiM"
    "jJ2ajYybmoyMm5yMjJmdjIwFjAaMigb7B8X47FH7BweKigeKBoqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KK"
    "iItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIFxqMVjqKPoYuMkqCT"
    "oIuMlJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OBaCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50jHMFinOIdId1i4qE"
    "doN2i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdop2jAV3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIehiKKKo4yjBfdG+YYV"
    "9zn7IK2Z+zn3IAUO+c7f98AVjHKLio5zi4qQdIuKknSLipN1BYyKBpV2lnaMiph4jIqZeYyKBZt6jIqbfI2KnXyMip5+jYqfgI2KoIGOiqGDjoqhhI+KooaP"
    "i6OIjoukio+LpIwFjoujjo+LopCPjKGSjoyhk46MoJWNjJ+WjYyemIyMnZqNjJuajIybnIyMmZ2MjAWMBoyKBvsHxfjsUfsHB4qKB4oGiox9nYqMe5yKjHua"
    "iYwFeZqKjHiYiYx3lomMdpWIjHWTiIx1koeMdJCHi3OOiItyjIeLcoqIi3OIh4t0hgWHinWEiIp1g4iKdoGJineAiYp4foqKeXyJint8iop7eoqKfXmKin54"
    "ioqAdoF2BYqKB4N1i4qEdIuKhnSLiohzi4qKcgXGoxWOoo+hi4ySoJOgi4yUn5ael52ZnJqbmpqMi5uYnZedlZ6UnpKfkZ+Pn44FoIygip+In4efhZ6EnoKd"
    "gZ1/m36bfJp7mXqXeZZ4lHeLipN2knaLio91jnSMcwWKc4h0h3WLioR2g3aLioJ3gHh/eX16fHt7fHt+eX95gXiCeIR3hXeHd4h2inaMBXeOd493kXiSeJR5"
    "lXmXe5iKi3yafJt9nH+dgJ6Cn4uMg6CEoIuMh6GIooqjjKMF99z5lBX7OfsgrX33OfcgBQ75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiM"
    "ipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qiho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qM"
    "jJucjIyZnYyMBYwGjIoG+wfF+OxR+wcHiooHigaKjH2diox7nIqMe5qJjAV5moqMeJiJjHeWiYx2lYiMdZOIjHWSh4x0kIeLc46Ii3KMh4tyioiLc4iHi3SG"
    "BYeKdYSIinWDiIp2gYmKd4CJinh+iop5fImKe3yKint6iop9eYqKfniKioB2gXYFiooHg3WLioR0i4qGdIuKiHOLiopyBcajFY6ij6GLjJKgk6CLjJSflp6X"
    "nZmcmpuamoyLm5idl52VnpSekp+Rn4+fjgWgjKCKn4ifh5+FnoSegp2BnX+bfpt8mnuZepd5lniUd4uKk3aSdouKj3WOdIxzBYpziHSHdYuKhHaDdouKgneA"
    "eH95fXp8e3t8e355f3mBeIJ4hHeFd4d3iHaKdowFd453j3eReJJ4lHmVeZd7mIqLfJp8m32cf52AnoKfi4yDoISgi4yHoYiiiqOMowX4F/j6Fa+Z+xv3FomM"
    "iYyJjImMiIx9i4iKiYqJiomKiYr7G/sWr333CfcFBQ75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiMipl5jIoFm3qMipt8jYqdfIyKnn6N"
    "ip+AjYqggY6KoYOOiqGEj4qiho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qMjJucjIyZnYyMBYwGjIoG+wfF+OxR"
    "+wcHiooHigaKjH2diox7nIqMe5qJjAV5moqMeJiJjHeWiYx2lYiMdZOIjHWSh4x0kIeLc46Ii3KMh4tyioiLc4iHi3SGBYeKdYSIinWDiIp2gYmKd4CJinh+"
    "iop5fImKe3yKint6iop9eYqKfniKioB2gXYFiooHg3WLioR0i4qGdIuKiHOLiopyBcajFY6ij6GLjJKgk6CLjJSflp6XnZmcmpuamoyLm5idl52VnpSekp+R"
    "n4+fjgWgjKCKn4ifh5+FnoSegp2BnX+bfpt8mnuZepd5lniUd4uKk3aSdouKj3WOdIxzBYpziHSHdYuKhHaDdouKgneAeH95fXp8e3t8e355f3mBeIJ4hHeF"
    "d4d3iHaKdowFd453j3eReJJ4lHmVeZd7mIqLfJp8m32cf52AnoKfi4yDoISgi4yHoYiiiqOMowX3Mvk+FZKSkpCRkJGOkY6QjI+MkouPipCKkYiRiJGGBZKG"
    "koSSg5KDk4GTgpODjIuThIyLk4WMipOHjYqTiI2Kk4iPipKKjYqki42MkowFj4yTjo2Mk46NjJOPjIyTkYyLk5KMi5OTk5STlWWVhIOEg4SEhIaFhoWIhYiG"
    "igWHioSLh4yGjIWOhY6FkISQhJKEk4STg5WDlIOTiouDkoqLg5GKjIOPiYyDjomMBYOOh4yEjImMcouJioSKh4qDiImKg4iJioOHioqDhYqLg4SKi4ODg4KD"
    "gbGBkpMFDvnO3/fAFYxyi4qOc4uKkHSLipJ0i4qTdQWMigaVdpZ2jIqYeIyKmXmMigWbeoyKm3yNip18jIqefo2Kn4CNiqCBjoqhg46KoYSPiqKGj4ujiI6L"
    "pIqPi6SMBY6Lo46Pi6KQj4yhko6MoZOOjKCVjYyflo2MnpiMjJ2ajYybmoyMm5yMjJmdjIwFjAaMigb7B8X47FH7BweKigeKBoqMfZ2KjHuciox7momMBXma"
    "iox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaB"
    "dgWKigeDdYuKhHSLioZ0i4qIc4uKinIFyboVj6GLjJKgk6CLjJSflp6XnZmcmpuamoyLm5idl52VnpSekp+Rn4+fjgWgjKCKn4ifh5+FnoSegp2BnX+bfpt8"
    "mnuZepd5lniUd4uKk3aSdouKj3WOdIxzBYpziHSHdYuKhHaDdouKgneAeH95fXp8e3t8e355f3mBeIJ4hHeFd4d3iHaKdowFd453j3eReJJ4lHmVeZd7mIqL"
    "fJp8m32cf52AnoKfi4yDoISgi4yHoYiiiqOMowX33/kwFYyIjIiMiYyJjYiMiY2JjYmNio6JjYqNio6KjoqNipeLjYyOjI6MjYwFjYyOjY2MjY2NjYyNjY6M"
    "jYyNjI6MjoyNi5eKjYqOio6KjYqNiY6KjYmNiY2JjAWIjYmMiYyIjIiMiYx/i4mKiIqIiomKiYqIiYmKiYmJiYqJiYiKiYqJioiKiIqJBYUHhQf7UhaMiYyI"
    "jIiMiYyJjYiMiY2JjYmNio6JjYqNio6KjoqNipeLjYyOjI6MjYwFjYyOjY2MjY2NjYyNjY6MjYyNjI6MjoyNi5eKjYqOio6KjYqNiY6KjYmNiY2JjAWIjYmM"
    "iYyIjIiMiYx/i4mKiIqIiomKiYqIiYmKiYmJiYqJiYiKiYqJioiKiIqJBYUHiAcO+c7f98AVjHKLio5zi4qQdIuKknSLipN1BYyKBpV2lnaMiph4jIqZeYyK"
    "BZt6jIqbfI2KnXyMip5+jYqfgI2KoIGOiqGDjoqhhI+KooaPi6OIjoukio+LpIwFjoujjo+LopCPjKGSjoyhk46MoJWNjJ+WjYyemIyMnZqNjJuajIybnIyM"
    "mZ2MjAWMBoyKBvsHxfjsUfsHB4qKB4oGiox9nYqMe5yKjHuaiYwFeZqKjHiYiYx3lomMdpWIjHWTiIx1koeMdJCHi3OOiItyjIeLcoqIi3OIh4t0hgWHinWE"
    "iIp1g4iKdoGJineAiYp4foqKeXyJint8iop7eoqKfXmKin54ioqAdoF2BYqKB4N1i4qEdIuKhnSLiohzi4qKcgXGoxWOoo+hi4ySoJOgi4yUn5ael52ZnJqb"
    "mpqMi5uYnZedlZ6UnpKfkZ+Pn44FoIygip+In4efhZ6EnoKdgZ1/m36bfJp7mXqXeZZ4lHeLipN2knaLio91jnSMcwWKc4h0h3WLioR2g3aLioJ3gHh/eX16"
    "fHt7fHt+eX95gXiCeIR3hXeHd4h2inaMBXeOd493kXiSeJR5lXmXe5iKi3yafJt9nH+dgJ6Cn4uMg6CEoIuMh6GIooqjjKMF9zj5OBWDB4yEjYSLio2Fi4qN"
    "hYyLjYWMio6Fj4WMio+GkIaLipCHjIqQh4yKBZGHjIuRh4yLkYiNipGJjYqRiY6LkYmOi5KKoYuSjI6LkY2Ni5KNjYyRjY2MkY4FjIuRj4yLkY+MjJCPjIyQ"
    "j4uMkJCPkIyMj5GOkYyMjZGMi42Ri4yNkYuMjZKMkgWTB5MHipKJkouMiZGLjImRiouJkYqMiJGHkYqMh5CGkIuMho+KjIaPiowFhY+Ki4WPiouFjomMhY2J"
    "jISNiYuFjYiLhIx1i4SKiIuFiYiLhYmJioWJiYqFiAWKi4WHiouFh4qKhoeKioaHi4qGhoeGioqHhYiFioqJhYqLiYWLiomFi4qJhIqEBYMHtZEVjJKMkQWN"
    "kI2RjZCOkI+QjpCPj4+Oj4+PjpCNj42PjY+MkIybi4+KkIqPiY+JkImPiI+HBY+Ij4eOho+GjoaNho2FjYaMhYyEi3+KhIqFiYaJhYmGiIaHhoiGh4eHiIeH"
    "h4gFhomHiYeJhoqHinuLhoyHjIeNh42GjYeOh4+HjoePiJCHkIiQiZCJkYmQipGKkgWRBw4cBTTd97YVxQaMpY2kkKOQopOik6CWn5admJ2Zm5mam5icl52V"
    "nZSekp6QBZ+PoI6gjJ+Kn4meh52HnISchJyCmoCaf5l+mHyXepZ5lXiTdpJ1kXSPco1xjHAF+8DF9geMjAeMBpR/BYyKmnqMipt8jIqLipx9jYqdf42JnoCN"
    "iqCBjoqgg46KooWOiqKGj4ujiI+LpIoFjgaaBo0GmoyNi5qMjYuZjY2LmY6Ni5mOjYyYjo2MmI+NiwWYkIyLmJCMjJiRjIuXkoyLl5KMjJeSjIyWk4yLlpSM"
    "jJaUjIuWlYuMlpWMi5WWBVmdgYCAgYGCgYKAg4CEgYSAhYCFgIZ/hoCHf4iAiH+Iiot/iX6Kfol+i36KiosFdox2jnePeJCKi3mSeJR6lXqWi4x7l4uMfJl+"
    "m36cgJ2Ki4Geg5+DoYahhqKJowWRB4yMB/iWBo8GjwaOjI+MjoyOjY2NjYyLjI2NBYyNjI2LjYqmiaWLjIeji4yGo4qLhaKKjIOgi4yBn4uMgJ6LjH+diox+"
    "m4qMfJoFiox7mYqMepiJjHmVi4yJjHiUiYx3k4iMdpGHjHWPiIx0jYeLdIyHi3OKiIt0iQWHinWHiIp1hYiKd4OIineCiYp4gImKeX+Kinp9iop7fIqKfXqK"
    "in55iop/eIF2BYoGigaKjAaKjouMgZ+KjICeiowFfpyKjH2biox8moqMepiLjIqMeZaJjXmViIx4lIiMd5KIjHWRiIt1j4iMdI2HjAV0BocGcoqHi3OIh4t0"
    "hoiKdIWIinaDiouJiouKdoKJinh/iYp4foqKBXp8iYp7fIqJfXqKin55iop/d4qKgXaKi4J1i4qDdIuKhXOLiodyi4qIcouKinAF+QSvFY2kkKKLjJChk6GT"
    "n4uMlZ6WnZibi4yYmpqZmpeblpyVnJOdkp6QnY8FjIuejZ+Mn4qeiZ2HnYachJuDm4KagJl+jIuYfZd8l3qVeZR4k3aRdpB0j3ONcgWDB4qKB/x3BoqMBowH"
    "Dvmf+Rr4fxW9nYKUioyClIqLgpSKjIGTiouBk4qMgJKAkoqMgJGJjICQiox/kIqMf5CJi3+PBYyKB36Piot+j4mLfo6Ki32Oiot9jYmLfoyJi32MBYkGfQaI"
    "BnCKh4txiIiLcYaIi3KEiIpzg4iKBXWCiouJinV/iop2f4mJeH2KioqLeXyKint6iop7eYqKfniKin93ioqBdouKgnUFiooHhXSKioZzi4qIc4uKinGMcYuK"
    "jnOLipBzjIqRdAWMigaUdYuKlXaMipd3jIqYeIyKm3mMipt6jIqdfIyLjIqefY2JoH+MiqF/jYoFjIuhgo6Ko4OOiqSEjouhh4qIiX6HfoeAhoGFgoWChIOE"
    "g4ODhIODgoODhIKLigWEgoWBioqGgYuKh4CKioh/i4qJfYuKioOLioyDi4qNg4uKjYSMio6FjIqPhYyKBZCGjIqRhoyKkoeNipGIjoqSiY6Kk4qOipOKpYuV"
    "jY2LlI2Ni5WOjYyVjo2LlZAFdJ+BiIOIgomEiYSKfouGjIaLh42HjYeNh46Ij4iPiJCJkYqRipKLk42YjpePlQWQlZGUkZORk5KTk5OTk5KTi4yTk5KUjIyS"
    "lIuMkZWMjJCVi4yQl4uMj5iLjI6ZBaIGjgaaBowGmoyNi5mMjYuZjY2LmI6NiwWYjo2LmI+Ni5ePjYuXkI2Ll5CMjJeQjIyXkIyMlpGMjJaSjIuWkoyMlZOM"
    "i5WTBYyMlJSMi5SUjIyUlFmdgoGCg4KDgoOBhIGFgYSBhoGGgIaBh4CHf4eAiX+IgIkFfol/in6LfopzjHSOdI91kIuMdpJ2k3eWeJZ5mHqafJp8nH6di4yA"
    "noGfgqCFoQWGooijiqOMo46jkKKRoZSglZ+WnouMmJ2anJqanJqdmJ6Wn5agk6CSi4yhkKKPBaKOo4yXipeLl4qXiZaJl4mWiJWHloeWh5WGlYaVhoyLlISV"
    "hZWElIOUg5SDlIEFDvmm9yH3rhWMjAf4qwaPBo8GjoyPjI6MBY6NjY2NjIuMjY2MjYyNi42Kp4uMiaWLjIelhqOKjIWii4yDoIqMgp+LjICei4wFf52KjH6b"
    "iox9moqMfJiJjHqXioyKi3qWiYx4lIiMeJKHjHeRh4t2j4eMdY2HjAV0BocGcYqIi3KIh4tzhoiKc4UFiIp1g4iKdoGIineAiYp4fomKeX2Kinp7iop7e4uK"
    "iop9eoqKfneLin93i4qBdgWLioN1ioqFdIZzi4qIcopxjHGOcouKkHORdIyKk3WLipV2i4qXd4uKmHeMipl6BYyKi4qbe4yKnHuMip19jYqefo2Kn4COiqCB"
    "joqhg46Ko4WOiqOGj4ukiI6LpYoFjgaZBo0GmIyNi5mMjIuZjY2LmI6Mi5iOjYuYjwWMi5iPjIuYkIyLl5CMjJeQjIyMi5aQjIyXkYyMlpKMi5WSjIyWk4yL"
    "lZOMjJWTBYuMlZSLjJWUWZ2CgoKCgYOCg4GEgYWBhYCFgYaAhoGHgIeAh4CIf4mAiYCJf4oFf4t/inWMdY52j3eQd5J4lIqLeZV5l3qYfJl8m36ciot/nYGe"
    "gaCDoIWhh6KIowWQB4yvFYqMBpEHjqOPopGhk6CVoJWel52Mi5icmpuamZyYnZedlYyLnpSfkp+QoI+hjqGMn4oFnomdh4yLnIechYuKm4SagpqBmH+Mi5d9"
    "jIuWfIyLlnuUeZR3k3eRdZBzj3KNcQWCB4qKB/yMBvc0+Y0V9zn7IK2Z+zn3IAUO+ab3IfeuFYyMB/irBo8GjwaOjI+MjowFjo2NjY2Mi4yNjYyNjI2LjYqn"
    "i4yJpYuMh6WGo4qMhaKLjIOgioyCn4uMgJ6LjAV/nYqMfpuKjH2aiox8mImMepeKjIqLepaJjHiUiIx4koeMd5GHi3aPh4x1jYeMBXQGhwZxioiLcoiHi3OG"
    "iIpzhQWIinWDiIp2gYiKd4CJinh+iYp5fYqKenuKint7i4qKin16iop+d4uKf3eLioF2BYuKg3WKioV0hnOLiohyinGMcY5yi4qQc5F0jIqTdYuKlXaLipd3"
    "i4qYd4yKmXoFjIqLipt7jIqce4yKnX2Nip5+jYqfgI6KoIGOiqGDjoqjhY6Ko4aPi6SIjouligWOBpkGjQaYjI2LmYyMi5mNjYuYjoyLmI6Ni5iPBYyLmI+M"
    "i5iQjIuXkIyMl5CMjIyLlpCMjJeRjIyWkoyLlZKMjJaTjIuVk4yMlZMFi4yVlIuMlZRZnYKCgoKBg4KDgYSBhYGFgIWBhoCGgYeAh4CHgIh/iYCJgIl/igV/"
    "i3+KdYx1jnaPd5B3kniUiot5lXmXeph8mXybfpyKi3+dgZ6BoIOghaGHooijBZAHjK8ViowGkQeOo4+ikaGToJWglZ6XnYyLmJyam5qZnJidl52VjIuelJ+S"
    "n5Cgj6GOoYyfigWeiZ2HjIuch5yFi4qbhJqCmoGYf4yLl32Mi5Z8jIuWe5R5lHeTd5F1kHOPco1xBYIHiooH/IwG98r5mxX7OfsgrX33OfcgBQ75pvch964V"
    "jIwH+KsGjwaPBo6Mj4yOjAWOjY2NjYyLjI2NjI2MjYuNiqeLjImli4yHpYajioyFoouMg6CKjIKfi4yAnouMBX+diox+m4qMfZqKjHyYiYx6l4qMiot6lomM"
    "eJSIjHiSh4x3kYeLdo+HjHWNh4wFdAaHBnGKiItyiIeLc4aIinOFBYiKdYOIinaBiIp3gImKeH6Jinl9iop6e4qKe3uLioqKfXqKin53i4p/d4uKgXYFi4qD"
    "dYqKhXSGc4uKiHKKcYxxjnKLipBzkXSMipN1i4qVdouKl3eLiph3jIqZegWMiouKm3uMipx7jIqdfY2Knn6Nip+AjoqggY6KoYOOiqOFjoqjho+LpIiOi6WK"
    "BY4GmQaNBpiMjYuZjIyLmY2Ni5iOjIuYjo2LmI8FjIuYj4yLmJCMi5eQjIyXkIyMjIuWkIyMl5GMjJaSjIuVkoyMlpOMi5WTjIyVkwWLjJWUi4yVlFmdgoKC"
    "goGDgoOBhIGFgYWAhYGGgIaBh4CHgIeAiH+JgImAiX+KBX+Lf4p1jHWOdo93kHeSeJSKi3mVeZd6mHyZfJt+nIqLf52BnoGgg6CFoYeiiKMFkAeMrxWKjAaR"
    "B46jj6KRoZOglaCVnpedjIuYnJqbmpmcmJ2XnZWMi56Un5KfkKCPoY6hjJ+KBZ6JnYeMi5yHnIWLipuEmoKagZh/jIuXfYyLlnyMi5Z7lHmUd5N3kXWQc49y"
    "jXEFggeKigf8jAb4BfkBFa+Z+xv3FomMiYyJjImMiIx9i4iKiYqJiomKiYr7G/sWr333CfcFBQ75pvch964VjIwH+KsGjwaPBo6Mj4yOjAWOjY2NjYyLjI2N"
    "jI2MjYuNiqeLjImli4yHpYajioyFoouMg6CKjIKfi4yAnouMBX+diox+m4qMfZqKjHyYiYx6l4qMiot6lomMeJSIjHiSh4x3kYeLdo+HjHWNh4wFdAaHBnGK"
    "iItyiIeLc4aIinOFBYiKdYOIinaBiIp3gImKeH6Jinl9iop6e4qKe3uLioqKfXqKin53i4p/d4uKgXYFi4qDdYqKhXSGc4uKiHKKcYxxjnKLipBzkXSMipN1"
    "i4qVdouKl3eLiph3jIqZegWMiouKm3uMipx7jIqdfY2Knn6Nip+AjoqggY6KoYOOiqOFjoqjho+LpIiOi6WKBY4GmQaNBpiMjYuZjIyLmY2Ni5iOjIuYjo2L"
    "mI8FjIuYj4yLmJCMi5eQjIyXkIyMjIuWkIyMl5GMjJaSjIuVkoyMlpOMi5WTjIyVkwWLjJWUi4yVlFmdgoKCgoGDgoOBhIGFgYWAhYGGgIaBh4CHgIeAiH+J"
    "gImAiX+KBX+Lf4p1jHWOdo93kHeSeJSKi3mVeZd6mHyZfJt+nIqLf52BnoGgg6CFoYeiiKMFkQeuBIwHkQeOo4+ikaGToJWglZ6XnYyLmJyam5qZnJidl52V"
    "jIuelJ+Sn5Cgj6GOoYyfigWeiZ2HjIuch5yFi4qbhJqCmoGYf4yLl32Mi5Z8jIuWe5R5lHeTd5F1kHOPco1xBYIHiooH/IwG9835NxWMiIyIjImMiY2IjImN"
    "iY2JjYqOiY2KjYqOio6KjYqXi42MjoyOjI2MBY2Mjo2NjI2NjY2MjY2OjI2MjYyOjI6MjYuXio2KjoqOio2KjYmOio2JjYmNiYwFiI2JjImMiIyIjImMf4uJ"
    "ioiKiIqJiomKiImJiomJiYmKiYmIiomKiYqIioiKiQV/B/tSFoyJjIiMiIyJjImNiIyJjYmNiY2KjomNio2KjoqOio2Kl4uNjI6MjoyNjAWNjI6NjYyNjY2N"
    "jI2NjoyNjI2MjoyOjI2Ll4qNio6KjoqNio2JjoqNiY2JjYmMBYiNiYyJjIiMiIyJjH+LiYqIioiKiYqJioiJiYqJiYmJiomJiIqJiomKiIqIiokFggcO99fJ"
    "+soV9zn7IK2Z+zn3IAWf+6AViAeMiIyIjIiNiI2JjImNiI6JjYqNiYyLjYmOio6KjoqOi46KBY4GjgaOjI6LjoyOjI6Mjo2NjY2Mjo2Njo2NjI2NjoyOjI6M"
    "joyOBY4HjgeOB44Hio6KjoqOio6JjoqNiY2JjoiNiYyJjYiNiIyIjIiMiIuIjAWIBogGiIqIi4iKiIqIiomJiouJiYmKiImJiIqJiYmJiIqIioiKiIuIiogF"
    "iAeIB8/9zxX47FH87AcO99f3d/rYFfs5+yCtffc59yAF+xb7khWIB4yIjIiMiI2IjYmMiY2IjomNio2JjIuNiY6KjoqOio6LjooFjgaOBo6MjouOjI6MjoyO"
    "jY2NjYyOjY2OjY2MjY2OjI6MjoyOjI4FjgeOB44HjgeKjoqOio6KjomOio2JjYmOiI2JjImNiI2IjIiMiIyIi4iMBYgGiAaIioiLiIqIioiKiYmKi4mJiYqI"
    "iYmIiomJiYmIioiKiIqIi4iKiAWIB4gHz/3PFfjsUfzsBw74Qffg+j4Vrpn7G/cWioyJjImMiIyJjH2LiIqJiomKiYqJivsb+xauffcK9wUFZPt3FYyIjIiM"
    "iIyIjYiMiY2JjYiNiY6KjYmOiY6KjYqOio6LjooFjwaOBo6MjouOjI6MjoyNjY6NjYyNjY2OjY2NjYyOjY6MjoyOi46MjgWOB44Hio6LjoqOio6JjoqOiY2J"
    "jYmOiY2JjIiNiY2IjIiMiIyIi4iMBYgGhwaIioiLiIqJioiKiImJiYiKiYmJiImJiomJiIqIioiKiIqIBYgHiAeIB8/9zxX47FH87AcO+BT3kfp0FYyIjIiM"
    "iYyJjYiMiY2JjYmNio6JjYqNio6KjoqNipeLjYyOjI6MjYwFjYyOjY2MjY2NjYyNjY6MjYyNjI6MjoyNi5eKjYqOio6KjYqNiY6KjYmNiY2JjAWIjYmMiYyI"
    "jIiMiYx/i4mKiIqIiomKiYqIiYmKiYmJiYqJiYiKiYqJioiKiIqJBX8HKfs+FYgHjIiMiIyIjYiNiYyJjYiOiY2KjYmOiY6KjoqOio6LjooFjgaOBo6MjouO"
    "jI6MjoyOjY2NjYyOjY2OjI2NjY2OjI6MjoyOi46MjgWOB44Hio6LjoqOio6KjomOiY2KjYmOiI2JjImNiI2IjIiMiIyIi4iMBYgGiAaIioiLiIqIioiKiImJ"
    "iYmKiImJiIqJiYmJiIqIioiKiIuIiogFiAeIB8/9zxX47FH87Acn+nQVjIiMiIyJjImNiIyJjYmNiY2KjomNio2KjoqOio2Kl4uNjI6MjoyNjAWNjI6NjYyN"
    "jY2NjI2NjoyNjI2MjoyOjI2Ll4qNio6KjoqNio2JjoqNiY2JjYmMBYiNiYyJjIiMiIyJjH+LiYqIioiKiYqJioiJiYqJiYmJiomJiIqJiomKiIqIiokFfwcO"
    "+crd98AVjHKLio5zi4qQdIuKknSLipN1BYyKBpV2lnaMiph4jIqZeYyKm3qMipt8jYoFnXyMip5+jYqfgI2KoIGOiqGDjoqhhI+KooaPi6OIjoukio+LpIyO"
    "i6OOj4uikAWPjKGSjoyhk46MoJWNjJ+WjYyemIyMnZqNjJuajIybnIyMmZ2MjJiei4yXoJWgBYyMB5Ohi4ySoouMkKKLjI6ji4yMpIqki4yIo4uMhqKLjISi"
    "i4yDoQWKjAaBoH+gi4x+noqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOIinaB"
    "iYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIFzdAVjAeSoJOgi4yUn5ael52ZnJqbmpqMi5uYnZedlZ6U"
    "npKfkZ+Pn46gjAWgip+In4efhZ6EnoKdgZ1/m36bfJp7mXqXeZZ4lHeLipN2knaLio91jnSMc4pzBYh0h3WLioR2g3aLioJ3gHh/eX16fHt7fHt+eX95gXiC"
    "eIR3hXeHd4h2inaMd44Fd493kXiSeJR5lXmXe5iKi3yafJt9nH+dgJ6Cn4uMg6CEoIuMh6GIooqjjKOOogX3HfiVFfdkPIyL1fsOwZdd2AWMB4yMB9htjIup"
    "qfsjwQWKjAZ4qmOCiov7OcqKi21tBQ75ut8WxffOBpKkk6STo4yLlKKWoJafl56YnJiamZmamJqWm5WclJySnJCdkJ6Ono0FjIufjKGKn4mfh52HnYSchJuC"
    "moGZf5h+mH2LipZ8lXmUeZJ3knaQdI50jXKMcQX7ysX3ygeKpomlh6SGo4Shi4yDoIuMgZ6LjICeBYuMf5yKjH6aio18mYqMe5iJjHqXiYx5lYmMd5SIjHeS"
    "iIx1kYiLdI+IjHONiIwFcgaHBnQGiIp0iQWIi3WHiIp2hoiKd4WKi4mKd4OJinmBiYp5gImKen+Kint9iop8fIqKfXqKin56BYqKB393iot/douKgHaLioF0"
    "i4qCc4uKg3GDcAWIB/vPB/dj+oIVkpKSkJGQkY6RjpCMj4ySi4+KkIqRiJGIkYYFkoaShJKDkoOTgZOCk4OMi5OEjIuThYyKk4eNipOIjYqTiI+KkoqNiqSL"
    "jYySjAWPjJOOjYyTjo2Mk4+MjJORjIuTkoyLk5OTlJOVZZWEg4SDhISEhoWGhYiFiIaKBYeKhIuHjIaMhY6FjoWQhJCEkoSThJODlYOUg5OKi4OSiouDkYqM"
    "g4+JjIOOiYwFg46HjISMiYxyi4mKhIqHioOIiYqDiImKg4eKioOFiouDhIqLg4ODgoOBsYGSkwUO+eLh98AVjHKLio5zi4qQdIuKknSLipR1i4qWdouKl3eL"
    "ipl4i4qaeYyKm3qMigWcfI2KnXyNip5+jYqggI2KoYGNiqGCjoujhI6Ko4aOi6SIj4ukio+LpIyPi6SOBY6Lo5COjKOSjYuilI2MoZWNjKCWjYyemI2MnZqN"
    "jJyajIybnIyMmp2LjJmei4wFl5+LjJagi4yUoYuMkqKLjJCii4yOo4uMjKSKpIuMiKOLjIaii4yEoouMgqGLjAWAoIuMf5+LjH2ei4x8nYqMe5yKjHqaiYx5"
    "momMeJiJjHaWiYx1lYmMdJSJi3OSBYiMc5CIi3KOh4tyjIeLcoqHi3KIiItzhoiKc4SIi3WCiYp1gYmKdoCJinh+iYoFeXyJinp8iop7eoqKfHmLin14i4p/"
    "d4uKgHaLioJ1i4qEdIuKhnSLiohzi4qKcgXFFoyjjqIFkKGLjJGgk6CVoJaemJ2ZnJqbm5qcmJ2XnpWelIyLnpKgkZ+PoI6Mi6CMoIqMiwWfiIyLn4eghZ6E"
    "jIuegp6BnX+cfpt8mnuZeph5lniVdpN2kXaLipB1jnSMc4pzBYh0hnWLioV2g3aBdoB4fnl9enx7e3x6fnl/eIF4goqLeIR2hXeHiot3iIqLdooFdoyKi3aO"
    "d492kXiSiot4lHiVeZd6mHuafJt9nH6dgJ6BoIOghaCLjIahiKKKowX3T/meFfc5+yCtmfs59yAFDvni4ffAFYxyi4qOc4uKkHSLipJ0i4qUdYuKlnaLipd3"
    "i4qZeIuKmnmMipt6jIoFnHyNip18jYqefo2KoICNiqGBjYqhgo6Lo4SOiqOGjoukiI+LpIqPi6SMj4ukjgWOi6OQjoyjko2LopSNjKGVjYyglo2MnpiNjJ2a"
    "jYycmoyMm5yMjJqdi4yZnouMBZefi4yWoIuMlKGLjJKii4yQoouMjqOLjIykiqSLjIiji4yGoouMhKKLjIKhi4wFgKCLjH+fi4x9nouMfJ2KjHuciox6momM"
    "eZqJjHiYiYx2lomMdZWJjHSUiYtzkgWIjHOQiItyjoeLcoyHi3KKh4tyiIiLc4aIinOEiIt1gomKdYGJinaAiYp4fomKBXl8iYp6fIqKe3qKinx5i4p9eIuK"
    "f3eLioB2i4qCdYuKhHSLioZ0i4qIc4uKinIFxRaMo46iBZChi4yRoJOglaCWnpidmZyam5uanJidl56VnpSMi56SoJGfj6COjIugjKCKjIsFn4iMi5+HoIWe"
    "hIyLnoKegZ1/nH6bfJp7mXqYeZZ4lXaTdpF2i4qQdY50jHOKcwWIdIZ1i4qFdoN2gXaAeH55fXp8e3t8en55f3iBeIKKi3iEdoV3h4qLd4iKi3aKBXaMiot2"
    "jnePdpF4koqLeJR4lXmXeph7mnybfZx+nYCegaCDoIWgi4yGoYiiiqMF9+X5rBX7OfsgrX33OfcgBQ754uH3wBWMcouKjnOLipB0i4qSdIuKlHWLipZ2i4qX"
    "d4uKmXiLipp5jIqbeoyKBZx8jYqdfI2Knn6NiqCAjYqhgY2KoYKOi6OEjoqjho6LpIiPi6SKj4ukjI+LpI4FjoujkI6Mo5KNi6KUjYyhlY2MoJaNjJ6YjYyd"
    "mo2MnJqMjJucjIyanYuMmZ6LjAWXn4uMlqCLjJShi4ySoouMkKKLjI6ji4yMpIqki4yIo4uMhqKLjISii4yCoYuMBYCgi4x/n4uMfZ6LjHydiox7nIqMepqJ"
    "jHmaiYx4mImMdpaJjHWViYx0lImLc5IFiIxzkIiLco6Hi3KMh4tyioeLcoiIi3OGiIpzhIiLdYKJinWBiYp2gImKeH6JigV5fImKenyKint6iop8eYuKfXiL"
    "in93i4qAdouKgnWLioR0i4qGdIuKiHOLiopyBcUWjKOOogWQoYuMkaCToJWglp6YnZmcmpubmpyYnZeelZ6UjIuekqCRn4+gjoyLoIygioyLBZ+IjIufh6CF"
    "noSMi56CnoGdf5x+m3yae5l6mHmWeJV2k3aRdouKkHWOdIxzinMFiHSGdYuKhXaDdoF2gHh+eX16fHt7fHp+eX94gXiCiot4hHaFd4eKi3eIiot2igV2jIqL"
    "do53j3aReJKKi3iUeJV5l3qYe5p8m32cfp2AnoGgg6CFoIuMhqGIooqjBfgg+RIVr5n7G/cWiYyJjImMiYyIjH2LiIqJiomKiYqJivsb+xavffcJ9wUFDvni"
    "4ffAFYxyi4qOc4uKkHSLipJ0i4qUdYuKlnaLipd3i4qZeIuKmnmMipt6jIoFnHyNip18jYqefo2KoICNiqGBjYqhgo6Lo4SOiqOGjoukiI+LpIqPi6SMj4uk"
    "jgWOi6OQjoyjko2LopSNjKGVjYyglo2MnpiNjJ2ajYycmoyMm5yMjJqdi4yZnouMBZefi4yWoIuMlKGLjJKii4yQoouMjqOLjIykiqSLjIiji4yGoouMhKKL"
    "jIKhi4wFgKCLjH+fi4x9nouMfJ2KjHuciox6momMeZqJjHiYiYx2lomMdZWJjHSUiYtzkgWIjHOQiItyjoeLcoyHi3KKh4tyiIiLc4aIinOEiIt1gomKdYGJ"
    "inaAiYp4fomKBXl8iYp6fIqKe3qKinx5i4p9eIuKf3eLioB2i4qCdYuKhHSLioZ0i4qIc4uKinIFxRaMo46iBZChi4yRoJOglaCWnpidmZyam5uanJidl56V"
    "npSMi56SoJGfj6COjIugjKCKjIsFn4iMi5+HoIWehIyLnoKegZ1/nH6bfJp7mXqYeZZ4lXaTdpF2i4qQdY50jHOKcwWIdIZ1i4qFdoN2gXaAeH55fXp8e3t8"
    "en55f3iBeIKKi3iEdoV3h4qLd4iKi3aKBXaMiot2jnePdpF4koqLeJR4lXmXeph7mnybfZx+nYCegaCDoIWgi4yGoYiiiqMF9zv5VhWSkpKQkZCRjpGOkIyP"
    "jJKLj4qQipGIkYiRhgWShpKEkoOSg5OBk4KTg4yLk4SMi5OFjIqTh42Kk4iNipOIj4qSio2KpIuNjJKMBY+Mk46NjJOOjYyTj4yMk5GMi5OSjIuTk5OUk5Vl"
    "lYSDhIOEhISGhYaFiIWIhooFh4qEi4eMhoyFjoWOhZCEkISShJOEk4OVg5SDk4qLg5KKi4ORioyDj4mMg46JjAWDjoeMhIyJjHKLiYqEioeKg4iJioOIiYqD"
    "h4qKg4WKi4OEiouDg4OCg4GxgZKTBQ754uH3wBWMcouKjnOLipB0i4qSdIuKlHWLipZ2i4qXd4uKmXiLipp5jIqbeoyKBZx8jYqdfI2Knn6NiqCAjYqhgY2K"
    "oYKOi6OEjoqjho6LpIiPi6SKj4ukjI+LpI4FjoujkI6Mo5KNi6KUjYyhlY2MoJaNjJ6YjYydmo2MnJqMjJucjIyanYuMmZ6LjAWXn4uMlqCLjJShi4ySoouM"
    "kKKLjI6ji4yMpIqki4yIo4uMhqKLjISii4yCoYuMBYCgi4x/n4uMfZ6LjHydiox7nIqMepqJjHmaiYx4mImMdpaJjHWViYx0lImLc5IFiIxzkIiLco6Hi3KM"
    "h4tyioeLcoiIi3OGiIpzhIiLdYKJinWBiYp2gImKeH6JigV5fImKenyKint6iop8eYuKfXiLin93i4qAdouKgnWLioR0i4qGdIuKiHOLiopyBcajFY6iBZCh"
    "i4yRoJOglaCWnpidmZyam5uanJidl56VnpSMi56SoJGfj6COjIugjKCKjIsFn4iMi5+HoIWehIyLnoKegZ1/nH6bfJp7mXqYeZZ4lXaTdpF2i4qQdY50jHOK"
    "cwWIdIZ1i4qFdoN2gXaAeH55fXp8e3t8en55f3iBeIKKi3iEdoV3h4qLd4iKi3aKBXaMiot2jnePdpF4koqLeJR4lXmXeph7mnybfZx+nYCegaCDoIWgi4yG"
    "oYiiiqMF9+j5SBWMiIyIjImMiY2IjImNiY2JjYqOiY2KjYqOio6KjYqXi42MjoyOjI2MBY2Mjo2NjI2NjY2MjY2OjI2MjYyOjI6MjYuXio2KjoqOio2KjYmO"
    "io2JjYmNiYwFiI2JjImMiIyIjImMf4uJioiKiIqJiomKiImJiomJiYmKiYmIiomKiYqIioiKiQV/B/tSFoyJjIiMiIyJjImNiIyJjYmNiY2KjomNio2KjoqO"
    "io2Kl4uNjI6MjoyNjAWNjI6NjYyNjY2NjI2NjoyNjI2MjoyOjI2Ll4qNio6KjoqNio2JjoqNiY2JjYmMBYiNiYyJjIiMiIyJjH+LiYqIioiKiYqJioiJiYqJ"
    "iYmJiomJiIqJiomKiIqIiokFggcO+Tbd99EVafiSrfySB/dt93cVjIgGjIiMiIyJjYiMiY2IjYmNiY6JjYqOiY2KjoqOigWOio4GjgaOBo6MjgaOjI6MjYyO"
    "jY2Mjo2NjY2NjY6MjY2OjI2MjoyOBY6MjgeOB44HjoqOB4qOio6KjYmOio2JjomNiY2IjYmMiI2JjIiMiIwFiIyIBogGiAaIiogGiIqIiomKiImJioiJiYmJ"
    "iYmIiomJiIqJioiKiAWIiogHiAeIB4z8jhWMiIyIjImNiIyJjYiNiY2JjomNio6JjYqOio6KBY6KjgaOBo4GjoyOBo6MjoyNjI6NjYyOjY2NjY2NjoyNjY6M"
    "jYyOjI4FjoyOB44HjgeOio4Hio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmMiIyIjAWIjIgGiAaIBoiKiAaIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIBYiK"
    "iAeIB4gHiIwHDvna3ffAFYxyi4qOc4uKkHSLipJ0i4qUdYuKlnaLipd3i4qZeIuKmnmMipt6jIqNiQWMioqKBishvXvf5wWMBowGm4CNiqCAjYqhgY2KoYIF"
    "joujhI6Ko4aOi6SIj4ukio+LpIyPi6SOjoujkI6Mo5KNi6KUjYyhlY2MoJaNjAWemI2MnZqNjJyajIybnIyMmp2LjJmei4yXn4uMlqCLjJShi4ySoouMkKKL"
    "jI6jBYuMjKSKpIuMiKOLjIaii4yEoouMgqGLjICgi4x/n4uMfZ6LjHydiox7nIqMiI4FioyMBur1WZs3MIqLfJWJjHaWiYx1lYmMdJSJi3OSBYiMc5CIi3KO"
    "h4tyjIeLcoqHi3KIiItzhoiKc4SIi3WCiYp1gYmKdoCJinh+iYoFeXyJinp8iop7eoqKfHmLin14i4p/d4uKgHaLioJ1i4qEdIuKhnSLiohzi4qKcgX4sfdR"
    "FYyMBoyKBY2JmXqYeZZ4lXaTdpF2i4qQdY50jHOKc4h0hnWLioV2g3aBdoB4fnl9enx7e3wFen55f3iBeIKKi3iEdoV3h4qLd4iKi3aKdoyKi3aOd492kXiS"
    "iot4lHiVeZeBkwWMB4wHaKwVigaKBoqMfZx+nYCegaCDoIWgi4yGoYiiiqOMo46ikKGLjJGgk6CVoJaemJ2ZnJqbm5oFnJidl56VnpSMi56SoJGfj6COjIug"
    "jKCKjIufiIyLn4eghZ6EjIuegp6BnX+ThAWMigaKBw75sN/3thWMbo1wi4qPcZFzi4qSdIuKk3aMipV4i4qWeQWMiouKmHuMipl7jIqafY2Km3+Nip2Ajoqe"
    "gY2KoIOOiqCFjoqiho6Lo4eOi6WIBY0GpgaNBqYGjQaljo6Lo4+Oi6KQjoygkY6MoJONjJ6VjoydlgWNjJuXjYyamYyMmZuMjJibi4yMjJadi4yVnoyMk6CL"
    "jJKii4yRo4+li4yNpoyoBffKUfvKB4pviXGHcoZ0hHWDd4J4gXp/fH59fX+Lin2Biot8gXqDeYR4hXeHdoh1iQVzinOMdY12jnePeJF5knqTfJWKi32Vi4x9"
    "l36Zf5qBnIKeg5+EoYaih6SJpYqnBffKUfvKB/dy+agV9zn7IK2Z+zn3IAUO+bDf97YVjG6NcIuKj3GRc4uKknSLipN2jIqVeIuKlnkFjIqLiph7jIqZe4yK"
    "mn2Nipt/jYqdgI6KnoGNiqCDjoqghY6KooaOi6OHjouliAWNBqYGjQamBo0GpY6Oi6OPjouikI6MoJGOjKCTjYyelY6MnZYFjYybl42MmpmMjJmbjIyYm4uM"
    "jIyWnYuMlZ6MjJOgi4ySoouMkaOPpYuMjaaMqAX3ylH7ygeKb4lxh3KGdIR1g3eCeIF6f3x+fX1/i4p9gYqLfIF6g3mEeIV3h3aIdYkFc4pzjHWNdo53j3iR"
    "eZJ6k3yViot9lYuMfZd+mX+agZyCnoOfhKGGooekiaWKpwX3ylH7ygf4CPm2Ffs5+yCtffc59yAFDvmw3/e2FYxujXCLio9xkXOLipJ0i4qTdoyKlXiLipZ5"
    "BYyKi4qYe4yKmXuMipp9jYqbf42KnYCOip6BjYqgg46KoIWOiqKGjoujh46LpYgFjQamBo0GpgaNBqWOjoujj46LopCOjKCRjoygk42MnpWOjJ2WBY2Mm5eN"
    "jJqZjIyZm4yMmJuLjIyMlp2LjJWejIyToIuMkqKLjJGjj6WLjI2mjKgF98pR+8oHim+JcYdyhnSEdYN3gniBen98fn19f4uKfYGKi3yBeoN5hHiFd4d2iHWJ"
    "BXOKc4x1jXaOd494kXmSepN8lYqLfZWLjH2Xfpl/moGcgp6Dn4ShhqKHpImliqcF98pR+8oH+EP5HBWvmfsb9xaJjImMiYyJjIiMfYuIiomKiYqJiomK+xv7"
    "Fq999wn3BQUO+bDf97YVjG6NcIuKj3GRc4uKknSLipN2jIqVeIuKlnkFjIqLiph7jIqZe4yKmn2Nipt/jYqdgI6KnoGNiqCDjoqghY6KooaOi6OHjouliAWN"
    "BqYGjQamBo0GpY6Oi6OPjouikI6MoJGOjKCTjYyelY6MnZYFjYybl42MmpmMjJmbjIyYm4uMjIyWnYuMlZ6MjJOgi4ySoouMkaOPpYuMjaaMqAX3ylH7ygeK"
    "b4lxh3KGdIR1g3eCeIF6f3x+fX1/i4p9gYqLfIF6g3mEeIV3h3aIdYkFc4pzjHWNdo53j3iReZJ6k3yViot9lYuMfZd+mX+agZyCnoOfhKGGooekiaWKpwX3"
    "ylH7ygf4C/lSFYyIjIiMiYyJjYiMiY2JjYmNio6JjYqNio6KjoqNipeLjYyOjI6MjYwFjYyOjY2MjY2NjYyNjY6MjYyNjI6MjoyNi5eKjYqOio6KjYqNiY6K"
    "jYmNiY2JjAWIjYmMiYyIjIiMiYx/i4mKiIqIiomKiYqIiYmKiYmJiYqJiYiKiYqJioiKiIqJBX8H+1IWjImMiIyIjImMiY2IjImNiY2JjYqOiY2KjYqOio6K"
    "jYqXi42MjoyOjI2MBY2Mjo2NjI2NjY2MjY2OjI2MjYyOjI6MjYuXio2KjoqOio2KjYmOio2JjYmNiYwFiI2JjImMiIyIjImMf4uJioiKiIqJiomKiImJiomJ"
    "iYmKiYmIiomKiYqIioiKiQWCBw75w9H45xX3xfzsk4wFjAaMigZP+wyFgIWABYWAhIKLioSChIOEg4uKg4SEhIOFg4aLioOHgoaCiIOIgYiCiYGKgIp/ioqL"
    "jmkFmQaOBpiMjoyYjI2MmI2NjJeOjYyXj42Mlo+NjJWQjYyUkY2MBZSSjIuUkoyMlJOMjJOTjIyTk4uMk5SLjJKVjIuSlpGWjIuRl5GXjIv4APluVJX7q/zC"
    "BYoGigb7sPjCVIEF+CD4hRX7OfsgrX33OfcgBQ75yt358BX+9MX4B4yMjIoHlH+Mi5d8jIsFmH2Mipl/jIqZf42KmoCNipuCjIqcgo2KnYONip2EjoudhY6K"
    "noeOi56IjoqeiQWOBp4GjgagBp6Njoyejo2Lno+NjJ2QjYydkY2MnJKNjJyUjIuclYyMm5WMjJqXjIwFmZeMjJmZi4yYmYyMl5uWnIyLlJyMjJOcjIySnIuM"
    "kp2LjJCdj56LjI6di4yNnQWMB54Hip6LjImdi4yInoeeiouGnYuMhJ2DnYuMgpyAnIuMgJuKjH+ZBYyKB3+ZiowFfZiLjHyXiox8loqMfJWJjHuUiox6k4qM"
    "epKJjHmRiYx5kImLeY+IjHmOiIt5jQWIBnkGiAZ4BogGeImJi3iIiIp5h4iKeYaJinmFiYqKi3qEiYp7gomKe4KKinuBBYqKB3x/iop9f35+iop/fYqKgH2K"
    "ioqKBYoGiowG+BBRB8b8shWNnY6dj52RnJGbkpyTmpSalZmWmZeYl5eYlpiWBZmUmpSak5qRm5GbkJuOm46ajIyLmoybipqKmoibiJqHmoWahZqEmoKZgpiB"
    "mX8Fl3+Xfpd9lXyVe5N7k3qRepB5j3qOeY15jHmLeYl5iHmIeYZ6hXmEe4N6iouCewWBfIB8iot/fX9/fX9+gIqLfYF8g3yDfIV8hXuGe4h8iHuKe4p7jHuM"
    "e457j3uQBXqQi4x8kXuTi4x9k4qMfZR+ln6Xiot/mICYgJmBmoKbg5uEm4Wchp2HnYmdiZ4FngcO+cP4C4YVk4wFjAaMigZP+wyFgIWABYWAhIKLioSChIOE"
    "g4uKg4SEhIOFg4aLioOHgoaCiIOIgYiCiYGKgIp/ioqLjmkFmQaOBpiMjoyYjI2MmI2NjJeOjYyXj42Mlo+NjJWQjYyUkY2MBZSSjIuUkoyMlJOMjJOTjIyT"
    "k4uMk5SLjJKVjIuSlpGWjIuRl5GXjIv4APluVJX7q/zCBYoGigb7sPjCVIEF+CP4IRWMiIuIjYmMiY2IjImNiY2JjYqNiY6KjYqOio2KjoqWi46MjoyNjI6M"
    "jYyNjQWOjIyNjY2NjY2OjI2MjYyOjI6LjYyOi5GKjouNio6KjoqNio2JjomNiY2KjYiMBYmNiYyIjImMiIyIjICLiIqJioiKiYqIiomJiYqJiYmJiomJiIqJ"
    "iYmLiIqIiokFfwf7UhaMiYyIi4iNiYyJjYiMiY2JjYmNio2JjoqNio6KjYqOipaLjoyOjI2MjoyNjI2NBY6MjI2NjY2NjY6MjYyNjI6MjouNjI6LkYqOi42K"
    "joqOio2KjYmOiY2JjYqNiIwFiY2JjIiMiYyIjIiMgIuIiomKiIqJioiKiYmJiomJiYmKiYmIiomJiYuIioiKiQWFB4gHDvnj248VxIP3APe8BYyM9/eKjAb3"
    "APu8xJP7z/nwio2JjYmNiY2IjIiNiIyIi4qMBYgGhwaHBogGh4qIBoeKiImJioiJiomKi4qJion7z/3wBfiQ99kVigeKigX73IyKjIwG9zf4UwWMjIyKBvc4"
    "/FMFfPlFFfvAcffABg75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiMipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qi"
    "ho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qMjJucjIyZnYyMBYwGjIoG+wfF+OxR+wcHiooHigaKjH2diox7nIqM"
    "e5qJjAV5moqMeJiJjHeWiYx2lYiMdZOIjHWSh4x0kIeLc46Ii3KMh4tyioiLc4iHi3SGBYeKdYSIinWDiIp2gYmKd4CJinh+iop5fImKe3yKint6iop9eYqK"
    "fniKioB2gXYFiooHg3WLioR0i4qGdIuKiHOLiopyBcajFY6ij6GLjJKgk6CLjJSflp6XnZmcmpuamoyLm5idl52VnpSekp+Rn4+fjgWgjKCKn4ifh5+FnoSe"
    "gp2BnX+bfpt8mnuZepd5lniUd4uKk3aSdouKj3WOdIxzBYpziHSHdYuKhHaDdouKgneAeH95fXp8e3t8e355f3mBeIJ4hHeFd4d3iHaKdowFd453j3eReJJ4"
    "lHmVeZd7mIqLfJp8m32cf52AnoKfi4yDoISgi4yHoYiiiqOMowX4OPlKFfvAcffABg7549uPFcSD9wD3vAWMjPf3iowG9wD7vMST+8/58IqNiY2JjYmNiIyI"
    "jYiMiIuKjAWIBocGhwaIBoeKiAaHioiJiYqIiYqJiouKiYqJ+8/98AX4kPfZFYoHiooF+9yMioyMBvc3+FMFjIyMigb3OPxTBfvc+VUVlXyMipd+jIqZf42K"
    "moKNiZyEjoqdhY6KnoaNiwWfiI6Ln4qOi5+Mjouejo6LnpCOjJ2RjYycko6NmpSNjJmXjIyXmIyMlJqMjZKbBWKRhXuCfoGAf4F+g3yEfIZ7h3qJeop7jHqN"
    "e497kH2SfpN/lYCWg5iFm2KFknsFDvnO3/fAFYxyi4qOc4uKkHSLipJ0i4qTdQWMigaVdpZ2jIqYeIyKmXmMigWbeoyKm3yNip18jIqefo2Kn4CNiqCBjoqh"
    "g46KoYSPiqKGj4ujiI6LpIqPi6SMBY6Lo46Pi6KQj4yhko6MoZOOjKCVjYyflo2MnpiMjJ2ajYybmoyMm5yMjJmdjIwFjAaMigb7B8X47FH7BweKigeKBoqM"
    "fZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqK"
    "e3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIFxqMVjqKPoYuMkqCToIuMlJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OBaCMoIqf"
    "iJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50jHMFinOIdId1i4qEdoN2i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdop2jAV3"
    "jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIehiKKKo4yjBfb5WhWUfI2Kln6Niph/jYqbgo2JnISNip6FjYqeho6LBZ+Ijoueio+LnoyO"
    "i5+OjouekI2MnpGNjJySjY2blI2MmJeNjJaYjYyUmoyNkZsFY5GEe4N+gYB/gX2DfYR7hnuHe4l6inqMe417j3uQfZJ9k3+VgZaDmISbY4WRewUO+ePbjxXE"
    "g/cA97wFjIz394qMBvcA+7zEk/vP+fCKjYmNiY2JjYiMiI2IjIiLiowFiAaHBocGiAaHiogGh4qIiYmKiImKiYqLiomKifvP/fAF+JD32RWKB4qKBfvcjIqM"
    "jAb3N/hTBYyMjIoG9zj8UwX7D/vNFYN+g36LioR/i4qFf4V+hn+Gf4h/iouIf4mABYuKiYCKgIqAi4CMgYyBi4qNgoyKjYKMio6Ci4qQgoyKkIOMipGEjIqS"
    "hIyKk4YFjIqTho2Kk4iOipOIjouUiY6Kk4qai5SMjouUjI2MlI2Ni5SOjYyTjo2Mk4+NjAWSj42MkpCMjJKRaZmFhoWHhIeEiIWIhImFiYSKf4uGjIWMho2F"
    "jYWOhpCFkIaRBYaShpOIlImTiZSKlYqUi5+MlY2WjpWOlo6XkJaPl5GXkZeSmJKXk5iUmGSVgn0FDvnO3/fAFYxyi4qOc4uKkHSLipJ0i4qTdQWMigaVdpZ2"
    "jIqYeIyKBZl5jIqbeoyKm3yNip18jIqefo2Kn4CNiqCBjoqhg46KoYSPiqKGj4ujiI6LpIoFjwagBoeFi4qFf4qLhn6Gf4Z/h3+If4iAi4qKgIqLioAFgAd2"
    "jAeMgYuKjYKLio6Ci4qPgouKkIKMipCDBYyKkYSMipKEjIqSho2Kk4aNipOIjoqTiI6Lk4mOipSKmouUjI6LlIyNjJSNjYsFlI6NjJOOjYyTj4yMk4+MjJOQ"
    "jIySkWmZhYaEh4WHhIiEiIWJhImFin+LhYyGjAWFjYaNhY6FkIaQhZGHkoaTiJSIk4qUiZWKlIuVjJWMlY2WjZWOlo+Xj5aQl5GXBZGXkpiPk5aNj4yhko6M"
    "oZOOjKCVjYyflo2MnpiMjJ2ajYybmoyMm5yMjJmdjIwFjAaMigb7B8X47FH7BweKigeKBoqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKH"
    "jHSQh4tzjoiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uK"
    "inIFxqMVjqKPoYuMkqCToIuMlJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OoIygigWfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qP"
    "dY50jHOKc4h0BYd1i4qEdoN2i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEe4Z9j4WDh4p3iHaKdowFd453j3eReJJ4lHmVeZd7mIqLfJp8m32cf52AnoKfi4yD"
    "oISgi4yHoYiiiqOMowUO+eb5ZPcjFYB+gH9/gH+Bf4F+gn6CfoN9hAV+hX2FfYaKi32GfYd8iHyIe4l8inuKfIqKi3CMcY9zkHOSdZR1lneZd5p5nXqfBXuh"
    "faN+pYuMgKeBqYOrha2HroixirKMso6xj66RrZOrlamWp4uMmKWZo5uhnJ8FnZ2fmp+ZoZahlKOSo5Clj6aMjIuaipuKmoqbiZqImoiZh5mGjIuZhpmFmIWZ"
    "hAWYg5iCmIKXgZeBl4CWf5Z+v5t/mIuMf5eKjH+Xiot+louMfpaKi32Viox9lIqLBX2Uiot8k4qMfJKJjHyRiox7kImMe5CKi3qQiot6j4mLeo6KjHmNiot5"
    "jYmLeowFiQZ5BogGbIqHi26Hh4tvhYeKcIMFiIpygIiKc36JinR8iYp2eoqKd3iLioqKeXaKint0iop8couKfXCLin9ui4qBbAWDaouKhGmLiodni4qIZYpj"
    "jGOOZYuKj2eLipJpi4qTapVsi4qXbouKmXCLippyBYyKm3SMip12jIqLip94jIqgeo2KonyNiqN+joqkgI6KpoOPiqeFj4uoh4+LqooFjgadBo0GnIyNi52N"
    "jIudjYyMnI6Ni5yPjIuckIyLm5CNjJuQBYyMmpGNjJqSjIyak4yLmZSMi5mUjIyZlYyLmJaLjJiWjIuXl4yMl5eLjJeYV5sF+3/6SRX7OfsgrX33OfcgBQ75"
    "n/ka+H8VvZ2ClIqMgpSKi4KUioyBk4qLgZOKjICSgJKKjICRiYyAkIqMf5CKjH+QiYt/jwWMigd+j4qLfo+Ji36Oiot9joqLfY2Ji36MiYt9jAWJBn0GiAZw"
    "ioeLcYiIi3GGiItyhIiKc4OIigV1goqLiYp1f4qKdn+JiXh9ioqKi3l8iop7eoqKe3mKin54iop/d4qKgXaLioJ1BYqKB4V0ioqGc4uKiHOLiopxjHGLio5z"
    "i4qQc4yKkXQFjIoGlHWLipV2jIqXd4yKmHiMipt5jIoFm3qMip18jIuMip59jYmgf4yKoX+NioyLoYKOiqODjoqkhI6LpYaOi6WIj4umigWOBpoGjAaajI2L"
    "mYyNi5mNjYuYjo2LBZiOjYuYj42Ll4+Ni5eQjYuXkIyMl5CMjJeQjIyWkYyMlpKMi5aSjIyVk4yLlZMFjIyUlIyLlJSMjJSUWZ2CgYKDgoOCg4GEgYWBhIGG"
    "gYaAhoGHgId/h4CJf4iAiQV+iX+Kfot+inOMdI50j3WQi4x2knaTd5Z4lnmYepp8mnycfp2LjICegZ+CoIWhBYaiiKOKo4yjjqOQopGhlKCVn5aei4yYnZqc"
    "mpqcmp2YnpaflqCToJKLjKGQoo8Foo6jjJeKl4uXipeJlomXiZaIlYeWh5aHlYaVhpWGjIuUhJWFlYSUg5SDlIOUgQX7WPjtFfs5+yCtffc59yAFDvnm+WT3"
    "IxWAfoB/f4B/gX+BfoJ+gn6DfYQFfoV9hX2Giot9hn2HfIh8iHuJfIp7inyKiotwjHGPc5BzknWUdZZ3mXeaeZ16nwV7oX2jfqWLjICngamDq4Wth66IsYqy"
    "jLKOsY+uka2Tq5WplqeLjJilmaOboZyfBZ2dn5qfmaGWoZSjkqOQpY+mjIyLmoqbipqKm4maiJqImYeZhoyLmYaZhZiFmYQFmIOYgpiCl4GXgZeAln+Wfr+b"
    "f5iLjH+Xiox/l4qLfpaLjH6Wiot9lYqMfZSKiwV9lIqLfJOKjHySiYx8kYqMe5CJjHuQiot6kIqLeo+Ji3qOiox5jYqLeY2Ji3qMBYkGeQaIBmyKh4tuh4eL"
    "b4WHinCDBYiKcoCIinN+iYp0fImKdnqKind4i4qKinl2iop7dIqKfHKLin1wi4p/bouKgWwFg2qLioRpi4qHZ4uKiGWKY4xjjmWLio9ni4qSaYuKk2qVbIuK"
    "l26Liplwi4qacgWMipt0jIqddoyKi4qfeIyKoHqNiqJ8jYqjfo6KpICOiqaDj4qnhY+LqIePi6qKBY4GnQaNBpyMjYudjYyLnY2MjJyOjYucj4yLnJCMi5uQ"
    "jYybkAWMjJqRjYyakoyMmpOMi5mUjIuZlIyMmZWMi5iWi4yYloyLl5eMjJeXi4yXmFebBftE+a8Vr5n7G/cWiYyJjImMiIyJjH2LiIqJiomKiYqJivsb+xav"
    "ffcJ9wUFDvmf+Rr4fxW9nYKUioyClIqLgpSKjIGTiouBk4qMgJKAkoqMgJGJjICQiox/kIqMf5CJi3+PBYyKB36Piot+j4mLfo6Ki32Oiot9jYmLfoyJi32M"
    "BYkGfQaIBnCKh4txiIiLcYaIi3KEiIpzg4iKBXWCiouJinV/iop2f4mJeH2KioqLeXyKint6iop7eYqKfniKin93ioqBdouKgnUFiooHhXSKioZzi4qIc4uK"
    "inGMcYuKjnOLipBzjIqRdAWMigaUdYuKlXaMipd3jIqYeIyKm3mMigWbeoyKnXyMi4yKnn2NiaB/jIqhf42KjIuhgo6Ko4OOiqSEjoulho6LpYiPi6aKBY4G"
    "mgaMBpqMjYuZjI2LmY2Ni5iOjYsFmI6Ni5iPjYuXj42Ll5CNi5eQjIyXkIyMl5CMjJaRjIyWkoyLlpKMjJWTjIuVkwWMjJSUjIuUlIyMlJRZnYKBgoOCg4KD"
    "gYSBhYGEgYaBhoCGgYeAh3+HgIl/iICJBX6Jf4p+i36Kc4x0jnSPdZCLjHaSdpN3lniWeZh6mnyafJx+nYuMgJ6Bn4KghaEFhqKIo4qjjKOOo5CikaGUoJWf"
    "lp6LjJidmpyampyanZielp+WoJOgkouMoZCijwWijqOMl4qXi5eKl4mWiZeJloiVh5aHloeVhpWGlYaMi5SElYWVhJSDlIOUg5SBBfsd+FMVrpn7G/cWioyJ"
    "jIiMiYyIjH6LiIqJioiKiYqKivsb+xauffcK9wUFDvnm+WT3IxWAfoB/f4B/gX+BfoJ+gn6DfYQFfoV9hX2Giot9hn2HfIh8iHuJfIp7inyKiotwjHGPc5Bz"
    "knWUdZZ3mXeaeZ16nwV7oX2jfqWLjICngamDq4Wth66IsYqyjLKOsY+uka2Tq5WplqeLjJilmaOboZyfBZ2dn5qfmaGWoZSjkqOQpY+mjIyLmoqbipqKm4ma"
    "iJqImYeZhoyLmYaZhZiFmYQFmIOYgpiCl4GXgZeAln+Wfr+bf5iLjH+Xiox/l4qLfpaLjH6Wiot9lYqMfZSKiwV9lIqLfJOKjHySiYx8kYqMe5CJjHuQiot6"
    "kIqLeo+Ji3qOiox5jYqLeY2Ji3qMBYkGeQaIBmyKh4tuh4eLb4WHinCDBYiKcoCIinN+iYp0fImKdnqKind4i4qKinl2iop7dIqKfHKLin1wi4p/bouKgWwF"
    "g2qLioRpi4qHZ4uKiGWKY4xjjmWLio9ni4qSaYuKk2qVbIuKl26Liplwi4qacgWMipt0jIqddoyKi4qfeIyKoHqNiqJ8jYqjfo6KpICOiqaDj4qnhY+LqIeP"
    "i6qKBY4GnQaNBpyMjYudjYyLnY2MjJyOjYucj4yLnJCMi5uQjYybkAWMjJqRjYyakoyMmpOMi5mUjIuZlIyMmZWMi5iWi4yYloyLl5eMjJeXi4yXmFebBfvf"
    "+eoViAeMiIyIjIiMiY2IjImNiI2JjYmNiY6KjYmOio6KjoqOipeLjoyOjI6MjYyOjY2MBY6NjY2NjY2OjI2NjoyNjI6MjouOjI6LkYqOi46KjoqOio2JjoqN"
    "iY6JjYmNiI0FiYyIjYmMiIyIjIiMf4uIioiKiIqIiomJiIqJiYmJiYmJiIqJiYiKiYqIioiKiAWFBw75n/ka+H8VvZ2ClIqMgpSKi4KUioyBk4qLgZOKjICS"
    "gJKKjICRiYyAkIqMf5CKjH+QiYt/jwWMigd+j4qLfo+Ji36Oiot9joqLfY2Ji36MiYt9jAWJBn0GiAZwioeLcYiIi3GGiItyhIiKc4OIigV1goqLiYp1f4qK"
    "dn+JiXh9ioqKi3l8iop7eoqKe3mKin54iop/d4qKgXaLioJ1BYqKB4V0ioqGc4uKiHOLiopxjHGLio5zi4qQc4yKkXQFjIoGlHWLipV2jIqXd4yKmHiMipt5"
    "jIoFm3qMip18jIuMip59jYmgf4yKoX+NioyLoYKOiqODjoqkhI6LpYaOi6WIj4umigWOBpoGjAaajI2LmYyNi5mNjYuYjo2LBZiOjYuYj42Ll4+Ni5eQjYuX"
    "kIyMl5CMjJeQjIyWkYyMlpKMi5aSjIyVk4yLlZMFjIyUlIyLlJSMjJSUWZ2CgYKDgoOCg4GEgYWBhIGGgYaAhoGHgId/h4CJf4iAiQV+iX+Kfot+inOMdI50"
    "j3WQi4x2knaTd5Z4lnmYepp8mnycfp2LjICegZ+CoIWhBYaiiKOKo4yjjqOQopGhlKCVn5aei4yYnZqcmpqcmp2YnpaflqCToJKLjKGQoo8Foo6jjJeKl4uX"
    "ipeJlomXiZaIlYeWh5aHlYaVhpWGjIuUhJWFlYSUg5SDlIOUgQX7uPiOFYgHjIgFi4iNiIyJjIiNiY2IjYmNiY2JjoqNiY6KjoqNio6KmIuOjI6MjYyOjI6N"
    "jYyNjQWNjY2NjY6NjYyOjY2MjouOjI6LjoyOio6LjoqOi46KjomNio6JjYmOiY2JjYmNBYmMiI2IjImMiIyIjH6LiIqJioiKiIqJiYiKiYmJiYmJiYiJiYqI"
    "iomJiIuIiogFhQcO+eb5ZPcjFYB+gH9/gH+Bf4F+gn6CfoN9hAV+hX2FfYaKi32GfYd8iHyIe4l8inuKfIqKi3CMcY9zkHOSdZR1lneZd5p5nXqfBXuhfaN+"
    "pYuMgKeBqYOrha2HroixirKMso6xj66RrZOrlamWp4uMmKWZo5uhnJ8FnZ2fmp+ZoZahlKOSo5Clj6aMjIuaipuKmoqbiZqImoiZh5mGjIuZhpmFmIWZhAWY"
    "g5iCmIKXgZeBl4CWf5Z+v5t/mIuMf5eKjH+Xiot+louMfpaKi32Viox9lIqLBX2Uiot8k4qMfJKJjHyRiox7kImMe5CKi3qQiot6j4mLeo6KjHmNiot5jYmL"
    "eowFiQZ5BogGbIqHi26Hh4tvhYeKcIMFiIpygIiKc36JinR8iYp2eoqKd3iLioqKeXaKint0iop8couKfXCLin9ui4qBbAWDaouKhGmLiodni4qIZYpjjGOO"
    "ZYuKj2eLipJpi4qTapVsi4qXbouKmXCLippyBYyKm3SMip12jIqLip94jIqgeo2KonyNiqN+joqkgI6KpoOPiqeFj4uoh4+LqooFjgadBo0GnIyNi52NjIud"
    "jYyMnI6Ni5yPjIuckIyLm5CNjJuQBYyMmpGNjJqSjIyak4yLmZSMi5mUjIyZlYyLmJaLjJiWjIuXl4yMl5eLjJeYV5sF+8n5rhWNio2KjYqOipmLjYyOjI2M"
    "jYyNjPcb9xZnmfsJ+wX7CfcFZ333G/sWBQ75n/ka+H8VvZ2ClIqMgpSKi4KUioyBk4qLgZOKjICSgJKKjICRiYyAkIqMf5CKjH+QiYt/jwWMigd+j4qLfo+J"
    "i36Oiot9joqLfY2Ji36MiYt9jAWJBn0GiAZwioeLcYiIi3GGiItyhIiKc4OIigV1goqLiYp1f4qKdn+JiXh9ioqKi3l8iop7eoqKe3mKin54iop/d4qKgXaL"
    "ioJ1BYqKB4V0ioqGc4uKiHOLiopxjHGLio5zi4qQc4yKkXQFjIoGlHWLipV2jIqXd4yKmHiMipt5jIoFm3qMip18jIuMip59jYmgf4yKoX+NioyLoYKOiqOD"
    "joqkhI6LpYaOi6WIj4umigWOBpoGjAaajI2LmYyNi5mNjYuYjo2LBZiOjYuYj42Ll4+Ni5eQjYuXkIyMl5CMjJeQjIyWkYyMlpKMi5aSjIyVk4yLlZMFjIyU"
    "lIyLlJSMjJSUWZ2CgYKDgoOCg4GEgYWBhIGGgYaAhoGHgId/h4CJf4iAiQV+iX+Kfot+inOMdI50j3WQi4x2knaTd5Z4lnmYepp8mnycfp2LjICegZ+CoIWh"
    "BYaiiKOKo4yjjqOQopGhlKCVn5aei4yYnZqcmpqcmp2YnpaflqCToJKLjKGQoo8Foo6jjJeKl4uXipeJlomXiZaIlYeWh5aHlYaVhpWGjIuUhJWFlYSUg5SD"
    "lIOUgQX7o/hSFY2KjoqNio6KmIuOjI2MjoyNjIyM9xv3FmiZ+wn7BfsK9wVoffcb+xYFDvn33RanBoyKBnv3jgeNBrCMjouvjo6MrZEFjourk46MqZWOjKiY"
    "jYymmo2MpJyNjKKejIygoIyMnqKMjJykjIyapoyMmKiLjAWWqoyMlKyLjJKtjIyQsI6yjLSKtIiyhrCKjISti4yCrIqMgKqLjH6oiox8poqMBXqkiox4ooqM"
    "dqCKjHSeiYxynImMcJqJjG6YiIxtlYiMa5OIi2mRiIxnjoiLZowFiQb7jnsGiooHb/3wBsX53hWMjPdvBq6KrIephqiEp4KkgKR9onygeZ94nXWbc5pxmG+W"
    "bZRrkmkFkGeOZYxjimOIZYZnhGmCa4Btfm98cXtzeXV3eHZ5dHxyfXKAb4JuhG2GaodoigX7b4yK+csG96nrFY2KjoqNio6KmIuOjI2MjoyNjIyM9xv3FmiZ"
    "+wr7BfsJ9wVoffcb+xYFDvnO+UD58BX8EAeKigeKBoqMioyAmYqMf5mKjH6Yiot+l4qMfJcFiowGe5WKjHuUiYx7lImMeZKJjHmRiYx5kIiMeY+IjHiOiIt5"
    "jQWIBngGiAZ5BogGeYmIi3mIiIp5h4mLeYaJinmFiYp6hIqKeoOKinuCiYp8gYqKfICKinx/BYqKB35+iop+fYuKf32KioB7ioqBeoJ6i4qDeYR5i4qFeYd4"
    "iHiLiol5i4qKeAV4B4oHjXmLio55i4qPeJB5i4qReQWMigaSegWKjAeTeouKlXqMi5Z6l3uMiph9i4oFmX2Mipl/jIqaf4yKm4GMipyBjIucgo2KnISNip2F"
    "jYqdho2KnoeNi56IjoqdiQWOBp4GjgaeBo4Gno2OjJ6OjouejwWNjJ6Rjoudko2MnZOMjIyLnJSMjJuUjYyaloyMmpeMjJmXjIyYmYyLl5qMi5SXBYyMioz7"
    "A8X58FEH/NcEiXiIeYh5hnmFeoR7g3uCe4F8gH2Afn5+fn9+gHyCi4oFfYOLinuDfIWKinuGe4Z7h3uIe4p7inuMe4x8jnuOe5B8kXyRiot9k4qLfZN8lQV+"
    "ln2Xf5eKi3+ZgJqBmoGbg5yEm4WdhpyInYidiZ2LnYydjZ2OnY+ckJ2RnJOcBZOblZuVmpeZl5iXl5mXmJWZlJqUmpKakZqRmo+ajoyLmo6ajJuMmoqMi5qK"
    "m4gFm4ibhpqFjIuahZqDmoKZgpiAmICXf5d+ln2VfZR8k3ySepF7kXqPeY55jXmMeQV4B/uz+SQVjYqNio2KjoqZi42MjoyNjI2MjYz3G/cWZ5n7CfsF+wn3"
    "BWd99xv7FgUO+gzd+FMVab2KjPwwpweMigZ794QHjQaujI6LrY+Oi6uRj4ypk46MqJWOjIuMBaaXjYykmo2No5uNjaCejYyfoIyMnaKMjZukjIyZpoyMl6iM"
    "i5aqi4yUrIuMkq0Fi4yQr4uMjrCLjIyzirOLjIiwi4yGr4uMhK2LjIKsi4yAqoqLf6iKjH2miox7pAWKjXmiiox3oImMdp6JjXObiY1ymomMcJeLjIiMbpWI"
    "jG2Th4xrkYiLaY+Ii2iMBYkG+4R7BoqKB2/8MIqKWQb3AWcVjIyM91Kt+1KMivgejIz3ZQesiamIqIamhKSBpICifouKoHyMi595nnecdZtzmXGXb5Ztk2uS"
    "aZBoi4qOZoxkBYpkiGaLioZohGmDa4Btf299cXtzenV4d3d5iot2fIuKdH5ygHKBcIRuhm2IaokF+2WMiowGDvoq+Lb5hBVp9xuKjPuBB4qKB4oGioyKjICZ"
    "iox/mYqMfpiKi36Xiox8lwWKjAZ7lYqMe5SJjHuUiYx5komMeZGJjHmQiIx5j4iMeI6Ii3mNBYgGeAaIBnkGiAZ5iYiLeYiIinmHiYt5homKeYWJinqEiop6"
    "g4qKe4KJinyBiop8gIqKfH8FiooHfn6Kin59i4p/fYqKgHuKioF6gnqLioN5hHmLioV5h3iIeIuKiXmLiop4BXgHigeNeYuKjnmLio94kHmLipF5BYyKBpJ6"
    "BYqMB5N6i4qVeoyLlnqXe4yKmH2LigWZfYyKmX+Mipp/jIqbgYyKnIGMi5yCjYqchI2KnYWNip2GjYqeh42LnoiOip2JBY4GngaOBp4GjgaejY6Mno6Oi56P"
    "BY2MnpGOi52SjYydk4yMjIuclIyMm5SNjJqWjIyal4yMmZeMjJiZjIuXmoyLlJcFjIyKjPsDxflhB4yMBeqtLAaKjAX2USCKivsbB/ca/H4ViHmIeYZ5hXqE"
    "e4N7gnuBfIB9gH5+fn5/foB8gouKBX2Di4p7g3yFiop7hnuGe4d7iHuKe4p7jHuMfI57jnuQfJF8kYqLfZOKi32TfJUFfpZ9l3+Xiot/mYCagZqBm4OchJuF"
    "nYaciJ2InYmdi52MnY2djp2PnJCdkZyTnAWTm5WblZqXmZeYl5eZl5iVmZSalJqSmpGakZqPmo6Mi5qOmoybjJqKjIuaipuIBZuIm4aahYyLmoWag5qCmYKY"
    "gJiAl3+XfpZ9lX2UfJN8knqRe5F6j3mOeY15jHkFeAcO+Y346fhTFfxgjIr4HoyM+Lqt/NgGhwaHBoiKiIqKi4iKiYmKi4mKiYmJiYqJi4qKiQWJB/3wB4kH"
    "jImLioyJjYmNiY2KjIuNiY6KjIuOio6KBY8Gjwb42K38uoyK+B6MjPhgrQZL+M8V+8Bx98AGDvmm9yH3rhWMjAf4qwaPBo8GjoyPjI6MBY6NjY2NjIuMjY2M"
    "jYyNi42Kp4uMiaWLjIelhqOKjIWii4yDoIqMgp+LjICei4wFf52KjH6biox9moqMfJiJjHqXioyKi3qWiYx4lIiMeJKHjHeRh4t2j4eMdY2HjAV0BocGcYqI"
    "i3KIh4tzhoiKc4UFiIp1g4iKdoGIineAiYp4fomKeX2Kinp7iop7e4uKiop9eoqKfneLin93i4qBdgWLioN1ioqFdIZzi4qIcopxjHGOcouKkHORdIyKk3WL"
    "ipV2i4qXd4uKmHeMipl6BYyKi4qbe4yKnHuMip19jYqefo2Kn4COiqCBjoqhg46Ko4WOiqOGj4ukiI6LpYoFjgaZBo0GmIyNi5mMjIuZjY2LmI6Mi5iOjYuY"
    "jwWMi5iPjIuYkIyLl5CMjJeQjIyMi5aQjIyXkYyMlpKMi5WSjIyWk4yLlZOMjJWTBYuMlZSLjJWUWZ2CgoKCgYOCg4GEgYWBhYCFgYaAhoGHgIeAh4CIf4mA"
    "iYCJf4oFf4t/inWMdY52j3eQd5J4lIqLeZV5l3qYfJl8m36ciot/nYGegaCDoIWhh6KIowWQB4yvFYqMBpEHjqOPopGhk6CVoJWel52Mi5icmpuamZyYnZed"
    "lYyLnpSfkp+QoI+hjqGMn4oFnomdh4yLnIechYuKm4SagpqBmH+Mi5d9jIuWfIyLlnuUeZR3k3eRdZBzj3KNcQWCB4qKB/yMBvgm+VEV+8Bx98AGDvmN+On4"
    "UxX8YIyK+B6MjPi6rfzYBocGhwaIioiKiouIiomJiouJiomJiYmKiYuKiokFiQf98AeJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+Nit/LqMivge"
    "jIz4YK0G/A343xWUfIyKl36Mipl/jYqago6Jm4SOip2Fjoqeho6LBZ6Ijoufio6Ln4yOi56OjouekI6MnZGOjJuSjo2alI2MmZeMjJeYjIyUmoyNkpsFYpGF"
    "e4N+gIB/gX6DfIR8hnuHeol7inqMeo17j3yQfJJ+k3+VgJaDmIWbYoWSewUO+ab3IfeuFYyMB/irBo8GjwaOjI+MjowFjo2NjY2Mi4yNjYyNjI2LjYqni4yJ"
    "pYuMh6WGo4qMhaKLjIOgioyCn4uMgJ6LjAV/nYqMfpuKjH2aiox8mImMepeKjIqLepaJjHiUiIx4koeMd5GHi3aPh4x1jYeMBXQGhwZxioiLcoiHi3OGiIpz"
    "hQWIinWDiIp2gYiKd4CJinh+iYp5fYqKenuKint7i4qKin16iop+d4uKf3eLioF2BYuKg3WKioV0hnOLiohyinGMcY5yi4qQc5F0jIqTdYuKlXaLipd3i4qY"
    "d4yKmXoFjIqLipt7jIqce4yKnX2Nip5+jYqfgI6KoIGOiqGDjoqjhY6Ko4aPi6SIjouligWOBpkGjQaYjI2LmYyMi5mNjYuYjoyLmI6Ni5iPBYyLmI+Mi5iQ"
    "jIuXkIyMl5CMjIyLlpCMjJeRjIyWkoyLlZKMjJaTjIuVk4yMlZMFi4yVlIuMlZRZnYKCgoKBg4KDgYSBhYGFgIWBhoCGgYeAh4CHgIh/iYCJgIl/igV/i3+K"
    "dYx1jnaPd5B3kniUiot5lXmXeph8mXybfpyKi3+dgZ6BoIOghaGHooijBZAHjK8ViowGkQeOo4+ikaGToJWglZ6XnYyLmJyam5qZnJidl52VjIuelJ+Sn5Cg"
    "j6GOoYyfigWeiZ2HjIuch5yFi4qbhJqCmoGYf4yLl32Mi5Z8jIuWe5R5lHeTd5F1kHOPco1xBYIHiooH/IwG5PlhFZR8jYqWfo2KmH+NipuCjYmchI2KnoWN"
    "ip6GjosFn4iOi56Kj4uejI6Ln46Oi56QjYyekY2MnJKNjZuUjYyYl42MlpiNjJSajI2RmwVjkYR7g36BgH+BfYN9hHuGe4d7iXqKeox7jXuPe5B9kn2Tf5WB"
    "loOYhJtjhZF7BQ75jfjp+FMV/GCMivgejIz4uq382AaHBocGiIqIioqLiIqJiYqLiYqJiYmJiomLioqJBYkH/fAHiQeMiYuKjImNiY2JjYqMi42JjoqMi46K"
    "jooFjwaPBvjYrfy6jIr4HoyM+GCtBvuQ+LoViAeMiIuIjIiNiYyIjYmNiI2JjYmNiY2KjomOio2KjoqOipiLjoyOjI2MjoyOjY2MBY2NjY2NjY2OjY2Mjo2N"
    "jI6LjoyOi5eKjouOio6JjYqOiY2JjomNiY2JjYmMiI0FiIyJjIiMiIx+i4iKiIqJioiKiImJiomJiYmJiYmIiYmKiImJioiLiIqIi4iKiAWMBg75pvch964V"
    "jIwH+KsGjwaPBo6Mj4yOjAWOjY2NjYyLjI2NjI2MjYuNiqeLjImli4yHpYajioyFoouMg6CKjIKfi4yAnouMBX+diox+m4qMfZqKjHyYiYx6l4qMiot6lomM"
    "eJSIjHiSh4x3kYeLdo+HjHWNh4wFdAaHBnGKiItyiIeLc4aIinOFBYiKdYOIinaBiIp3gImKeH6Jinl9iop6e4qKe3uLioqKfXqKin53i4p/d4uKgXYFi4qD"
    "dYqKhXSGc4uKiHKKcYxxjnKLipBzkXSMipN1i4qVdouKl3eLiph3jIqZegWMiouKm3uMipx7jIqdfY2Knn6Nip+AjoqggY6KoYOOiqOFjoqjho+LpIiOi6WK"
    "BY4GmQaNBpiMjYuZjIyLmY2Ni5iOjIuYjo2LmI8FjIuYj4yLmJCMi5eQjIyXkIyMjIuWkIyMl5GMjJaSjIuVkoyMlpOMi5WTjIyVkwWLjJWUi4yVlFmdgoKC"
    "goGDgoOBhIGFgYWAhYGGgIaBh4CHgIeAiH+JgImAiX+KBX+Lf4p1jHWOdo93kHeSeJSKi3mVeZd6mHyZfJt+nIqLf52BnoGgg6CFoYeiiKMFkAeMrxWKjAaR"
    "B46jj6KRoZOglaCVnpedjIuYnJqbmpmcmJ2XnZWMi56Un5KfkKCPoY6hjJ+KBZ6JnYeMi5yHnIWLipuEmoKagZh/jIuXfYyLlnyMi5Z7lHmUd5N3kXWQc49y"
    "jXEFggeKigf8jAb3avk8FYgHjIiMiIyIjImNiIyJjYiNiY2JjomNio6JjYqOio6KjoqXi46MjoyOjI2Mjo2NjAWOjY2NjY2NjoyNjY6MjYyOjI6LjoyOi5GK"
    "jouOio6KjoqNiY6KjYmOiY2JjYiNBYmMiI2JjIiMiIyIjH+LiIqIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIiogFhQcO+Y346fhTFfxgjIr4HoyM+Lqt"
    "/NgGhwaHBoiKiIqKi4iKiYmKi4mKiYmJiYqJi4qKiQWJB/3wB4kHjImLioyJjYmNiY2KjIuNiY6KjIuOio6KBY8Gjwb3vgaHhYuKhX+FfoZ/h38Fh3+If4qL"
    "iYCLiomAioCKgIuAjIGMgYyKjYKLio6Ci4qOgoyKj4KMipGDi4qShAWMipGEjYqSho2KkoaNipOIjoqTiI6LlImOipSKmouTjI6LlIyOjJONjouTjo2MBZSO"
    "jIyTj42Mk4+MjJKQjIySkWqZhIaFh4SHhYiEiISJhYmEioCLhYyFjIaNhY0FhY6GkIWQhpGGkoeTiJSIk4mUipWKlIufjZWMlo6VjpaOl5CWkJeQl5GXkpiM"
    "jAX3gq37bQaNjmSVg34F+7SMivgejIz4YK0GDvmm9yH3rhWMjAf4qwaPBo8GjoyPjI6MBY6NjY2NjIuMjY2MjYyNi42Kp4uMiaWLjIelhqOKjIWii4yDoIqM"
    "gp+LjICei4wFf52KjH6biox9moqMfJiJjHqXioyKi3qWiYx4lIiMeJKHjHeRh4t2j4eMdY2HjAV0BocGcYqIi3KIh4tzhoiKc4WIinWDiIp2gYiKd4CJinh+"
    "iYp5fYqKenuKint7i4oFiop9eoqKfneLin93i4qBdouKg3WKioV0hnOLiohyinGMcY5yi4qQc5F0jIqTdQWLipV2i4qXd4uKmHeMipl6jIqLipt7jIqce4yK"
    "nX2Nip5+jYqfgI6KoIGOiqGDBY6Ko4WOiqOGj4ukiI6LpYqNi4iFi4qFf4qLhn6Gf4Z/h3+If4iAi4qKgIqLioAFgAd2jAeMgYuKjYKLio6Ci4qPgouKkIKM"
    "ipCDjIqRhIyKkoSMipKGjYqTho2Kk4iOigWTiI6Lk4mOipSKmouUjI6LlIyNjJSNjYuUjo2Mk46NjJOPjIyTj4yMk5CMjJKRBWmZhYaEh4WHhIiEiIWJhImF"
    "in+LhYyGjIWNho2FjoWQhpCFkYeShpOIlIiTipQFiZWKlIuVjJWMlY2WjZWOlo+Xj5aQl5GXkZeSmI2OjIuZjY2LmI6Mi5iOjYuYjwWMi5iPjIuYkIyLl5CM"
    "jJeQjIyMi5aQjIyXkYyMlpKMi5WSjIyWk4yLlZOMjJWTBYuMlZSLjJWUWZ2CgoKCgYOCg4GEgYWBhYCFgYaAhoGHgIeAh4CIf4mKi2+Sg38Fgot/inWMdY52"
    "j3eQd5J4lIqLeZV5l3qYfJl8m36ciot/nYGegaCDoIWhh6KIowWQB4yvFYqMBpEHjqOPopGhk6CVoJWel52Mi5icmpuamZyYnZedlYyLnpSfkp+QoI+hjqGM"
    "n4oFnomdh4yLnIechYuKm4SagpqBmH+Mi5d9jIuWfIyLlnuUeZR3k3eRdZBzj3KNcQWCB4qKB/yMBg75jfjp+FMV/GCMivgejIz4uq382AaHBocGiIqIioqL"
    "iIqJiYqLiYqJiYmJiomLioqJBYkH/fAHiQeMiYuKjImNiY2JjYqMi42JjoqMi46KjooFjwaPBvjYrfy6jIr4HoyM+GCtBvt7+H4VjYqOio2KjoqYi46MjYyO"
    "jI2MjIz3G/cWaJn7CfsF+wr3BWh99xv7FgUO+ab3IfeuFYyMB/irBo8GjwaOjI+MjowFjo2NjY2Mi4yNjYyNjI2LjYqni4yJpYuMh6WGo4qMhaKLjIOgioyC"
    "n4uMgJ6LjAV/nYqMfpuKjH2aiox8mImMepeKjIqLepaJjHiUiIx4koeMd5GHi3aPh4x1jYeMBXQGhwZxioiLcoiHi3OGiIpzhQWIinWDiIp2gYiKd4CJinh+"
    "iYp5fYqKenuKint7i4qKin16iop+d4uKf3eLioF2BYuKg3WKioV0hnOLiohyinGMcY5yi4qQc5F0jIqTdYuKlXaLipd3i4qYd4yKmXoFjIqLipt7jIqce4yK"
    "nX2Nip5+jYqfgI6KoIGOiqGDjoqjhY6Ko4aPi6SIjouligWOBpkGjQaYjI2LmYyMi5mNjYuYjoyLmI6Ni5iPBYyLmI+Mi5iQjIuXkIyMl5CMjIyLlpCMjJeR"
    "jIyWkoyLlZKMjJaTjIuVk4yMlZMFi4yVlIuMlZRZnYKCgoKBg4KDgYSBhYGFgIWBhoCGgYeAh4CHgIh/iYCJgIl/igV/i3+KdYx1jnaPd5B3kniUiot5lXmX"
    "eph8mXybfpyKi3+dgZ6BoIOghaGHooijBZAHjK8ViowGkQeOo4+ikaGToJWglZ6XnYyLmJyam5qZnJidl52VjIuelJ+Sn5Cgj6GOoYyfigWeiZ2HjIuch5yF"
    "i4qbhJqCmoGYf4yLl32Mi5Z8jIuWe5R5lHeTd5F1kHOPco1xBYIHiooH/IwG94D5ABWNio2KjYqOipmLjoyNjI2MjYyNjPcb9xZnmfsJ+wX7CfcFZ333G/sW"
    "BQ759/h6+BwVafeEioz7aQeDgoB/f4B/gH+BfoF+gn6DfoQFfYR9hXyFfIeLinyHe4h7iHuIeop5inqKiotwjHGPc5BzknWUdZZ3mXeaeZ16nwV7oX2jfqWL"
    "jICngamDq4Wth66IsYqyjLKOsY+uka2Tq5WplqeLjJilmaOboZyfBZ2dn5qfmaGWoZSjkqOQpY+mjIyLmoqbipqKm4maiJqImYeZhoyLmYaZhZiFmYQFmIOY"
    "gpiCl4GXgZeAln+Wfr+bf5iLjH+Xiox/l4qLfpaLjH6Wiot9lYqMfZSKiwV9lIqLfJOKjHySiYx8kYqMe5CJjHuQiot6kIqLeo+Ji3qOiox5jYqLeY2Ji3qM"
    "BYkGeQaIBmyKh4tuh4eLb4WHinCDBYiKcoCIinN+iYp0fImKdnqKind4i4qKinl2iop7dIqKfHKLin1wi4p/bouKgWwFg2qLioRpi4qHZ4uKiGWKY4xjjmWL"
    "io9ni4qSaYuKk2qVbIuKl26Liplwi4qacgWMipt0jIqddoyKi4qfeIyKoHqNiqJ8jYqjfo6KpICOiqaDj4qnhY+LqIePi6qKBY4GnwaNBp6MjYuejY2LnY6N"
    "i52PjYudj4yLnJCNjJyQjIybkY2LBZuSjIyakoyMmpOMjJmTjIyZlYyLmZWLjJiWjIuYl5eXjIyXl4uMl5iMjYyNjI0FjQf3fweNB4qNi4yKjYmNiY2JjIqL"
    "iY2IjIqLiIyIjAWHBocG+6IGzPi2Fa6Z+xv3FoqMiYyIjImMiIx+i4iKiYqIiomKior7G/sWrn33CfcFBQ75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoG"
    "lXaWdoyKmHiMipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qiho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6Y"
    "jIydmo2Mm5qMjJucjIyZnYyMBYwGjIoGWQeKbIluh2+FcYRzgnSBdoB4fnl+fIuKfH17fwV7gIqLeoF5g4qLeIV3hnaHdIl0inyMfYt9jIuMfYx+jX6Of41/"
    "j3+Pf4+Aj4CQBYCRgZCBkoGRgpKBk4KTg5NZeZWCi4qVg4uKlYOMi5WDjIuWg4yLloSMipaFjIoFl4WMi5iFjIuXho2KmIaMi5mHjIqZiIyKmYiNi5mIjYua"
    "iIyLm4mMi5uKjIubigWNBpsGjQamBo6MpY2OjKOPjosFo5GOjKGSjoyglI2Nn5WNjJ6YjYycmY2Mm5uMjJmcjIyZnouMl5+MjJWhjIyTowWMjAaSpJGmi4yP"
    "p4yMjamMqwX4q1H7BweKigeKBoqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOI"
    "inaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIFxqMVjqKPoYuMkqCToIuMlJ+WnpedmZyam5qajIub"
    "mJ2XnZWelJ6Sn5Gfj5+OBaCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50jHMFinOIdId1i4qEdoN2i4qCd4B4f3l9enx7e3x7"
    "fnl/eYF4gniEd4V3h3eIdop2jAV3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIehiKKKo4yjBfgX+PoVr5n7G/cWiYyJjImMiYyIjH2L"
    "iIqJiomKiYqJivsb+xavffcJ9wUFDvn3+Hr4HBVp94SKjPtpB4OCgH9/gH+Af4F+gX6CfoN+hAV9hH2FfIV8h4uKfId7iHuIe4h6inmKeoqKi3CMcY9zkHOS"
    "dZR1lneZd5p5nXqfBXuhfaN+pYuMgKeBqYOrha2HroixirKMso6xj66RrZOrlamWp4uMmKWZo5uhnJ8FnZ2fmp+ZoZahlKOSo5Clj6aMjIuaipuKmoqbiZqI"
    "moiZh5mGjIuZhpmFmIWZhAWYg5iCmIKXgZeBl4CWf5Z+v5t/mIuMf5eKjH+Xiot+louMfpaKi32Viox9lIqLBX2Uiot8k4qMfJKJjHyRiox7kImMe5CKi3qQ"
    "iot6j4mLeo6KjHmNiot5jYmLeowFiQZ5BogGbIqHi26Hh4tvhYeKcIMFiIpygIiKc36JinR8iYp2eoqKd3iLioqKeXaKint0iop8couKfXCLin9ui4qBbAWD"
    "aouKhGmLiodni4qIZYpjjGOOZYuKj2eLipJpi4qTapVsi4qXbouKmXCLippyBYyKm3SMip12jIqLip94jIqgeo2KonyNiqN+joqkgI6KpoOPiqeFj4uoh4+L"
    "qooFjgafBo0GnoyNi56NjYudjo2LnY+Ni52PjIuckI2MnJCMjJuRjYsFm5KMjJqSjIyak4yMmZOMjJmVjIuZlYuMmJaMi5iXl5eMjJeXi4yXmIyNjI2MjQWN"
    "B/d/B40Hio2LjIqNiY2JjYmMiouJjYiMiouIjIiMBYcGhwb7ogb7a/kWFZR8jIqXfoyKmX+NipqCjombhI6KnYWOip6GjosFnoiOi5+KjoufjI6Lno6Oi56Q"
    "joydkY6Mm5KOjZqUjYyZl4yMl5iMjJSajI2SmwVikYV7g36AgH+BfoN8hHyGe4d6iXqKe4x6jXuPfJB8kn6Tf5WAloOYhZtihZJ7BQ75zt/3wBWMcouKjnOL"
    "ipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiMipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qiho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKO"
    "jKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qMjJucjIyZnYyMBYwGjIoGWQeKbIluh2+FcYRzgnSBdoB4fnl+fIuKfH17fwV7gIqLeoF5g4qLeIV3hnaHdIl0inyM"
    "fYt9jIuMfYx+jX6Of41/j3+Pf4+Aj4CQBYCRgZCBkoGRgpKBk4KTg5NZeZWCi4qVg4uKlYOMi5WDjIuWg4yLloSMipaFjIoFl4WMi5iFjIuXho2KmIaMi5mH"
    "jIqZiIyKmYiNi5mIjYuaiIyLm4mMi5uKjIubigWNBpsGjQamBo6MpY2OjKOPjosFo5GOjKGSjoyglI2Nn5WNjJ6YjYycmY2Mm5uMjJmcjIyZnouMl5+MjJWh"
    "jIyTowWMjAaSpJGmi4yPp4yMjamMqwX4q1H7BweKigeKBoqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KKiItz"
    "iIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIFxqMVjqKPoYuMkqCToIuM"
    "lJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OBaCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50jHMFinOIdId1i4qEdoN2"
    "i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdop2jAV3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIehiKKKo4yjBfb5WhWUfI2K"
    "ln6Niph/jYqbgo2JnISNip6FjYqeho6LBZ+Ijoueio+LnoyOi5+OjouekI2MnpGNjJySjY2blI2MmJeNjJaYjYyUmoyNkZsFY5GEe4N+gYB/gX2DfYR7hnuH"
    "e4l6inqMe417j3uQfZJ9k3+VgZaDmISbY4WRewUO+ff4evgcFWn3hIqM+2kHg4KAf3+Af4B/gX6BfoJ+g36EBX2EfYV8hXyHi4p8h3uIe4h7iHqKeYp6ioqL"
    "cIxxj3OQc5J1lHWWd5l3mnmdep8Fe6F9o36li4yAp4Gpg6uFrYeuiLGKsoyyjrGPrpGtk6uVqZani4yYpZmjm6GcnwWdnZ+an5mhlqGUo5KjkKWPpoyMi5qK"
    "m4qaipuJmoiaiJmHmYaMi5mGmYWYhZmEBZiDmIKYgpeBl4GXgJZ/ln6/m3+Yi4x/l4qMf5eKi36Wi4x+loqLfZWKjH2UiosFfZSKi3yTiox8komMfJGKjHuQ"
    "iYx7kIqLepCKi3qPiYt6joqMeY2Ki3mNiYt6jAWJBnkGiAZsioeLboeHi2+Fh4pwgwWIinKAiIpzfomKdHyJinZ6iop3eIuKiop5doqKe3SKinxyi4p9cIuK"
    "f26LioFsBYNqi4qEaYuKh2eLiohlimOMY45li4qPZ4uKkmmLipNqlWyLipdui4qZcIuKmnIFjIqbdIyKnXaMiouKn3iMiqB6jYqifI2Ko36OiqSAjoqmg4+K"
    "p4WPi6iHj4uqigWOBp8GjQaejI2Lno2Ni52OjYudj42LnY+Mi5yQjYyckIyMm5GNiwWbkoyMmpKMjJqTjIyZk4yMmZWMi5mVi4yYloyLmJeXl4yMl5eLjJeY"
    "jI2MjYyNBY0H938HjQeKjYuMio2JjYmNiYyKi4mNiIyKi4iMiIwFhwaHBvuiBjH48RWIB4yIi4iMiI2JjIiNiY2IjYmNiY2JjYqOiY6KjYqOio6KmIuOjI6M"
    "jYyOjI6NjYwFjY2NjY2NjY6NjYyOjY2MjouOjI6Ll4qOi46KjomNio6JjYmOiY2JjYmNiYyIjQWIjImMiIyIjH6LiIqIiomKiIqIiYmKiYmJiYmJiYiJiYqI"
    "iYmKiIuIioiLiIqIBQ75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKmHiMipl5jIoFm3qMipt8jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qi"
    "ho+Lo4iOi6SKj4ukjAWOi6OOj4uikI+MoZKOjKGTjoyglY2Mn5aNjJ6YjIydmo2Mm5qMjJucjIyZnYyMBYwGjIoGWQeKbIluh2+FcYRzgnSBdoB4fnl+fIuK"
    "fH17fwV7gIqLeoF5g4qLeIV3hnaHdIl0inyMfYt9jIuMfYx+jX6Of41/j3+Pf4+Aj4CQBYCRgZCBkoGRgpKBk4KTg5NZeZWCi4qVg4uKlYOMi5WDjIuWg4yL"
    "loSMipaFjIoFl4WMi5iFjIuXho2KmIaMi5mHjIqZiIyKmYiNi5mIjYuaiIyLm4mMi5uKjIubigWNBpsGjQamBo6MpY2OjKOPjosFo5GOjKGSjoyglI2Nn5WN"
    "jJ6YjYycmY2Mm5uMjJmcjIyZnouMl5+MjJWhjIyTowWMjAaSpJGmi4yPp4yMjamMqwX4q1H7BweKigeKBoqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaV"
    "iIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KKiItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSL"
    "ioZ0i4qIc4uKinIFxqMVjqKPoYuMkqCToIuMlJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OBaCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3"
    "i4qTdpJ2i4qPdY50jHMFinOIdId1i4qEdoN2i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdop2jAV3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCe"
    "gp+LjIOghKCLjIehiKKKo4yjBfd8+TUViAeMiIyIjIiMiY2IjImNiI2JjYmOiY2KjomNio6KjoqOipeLjoyOjI6MjYyOjY2MBY6NjY2NjY2OjI2NjoyNjI6M"
    "jouOjI6LkYqOi46KjoqOio2JjoqNiY6JjYmNiI0FiYyIjYmMiIyIjIiMf4uIioiKiIqJioiJiYqIiYmJiYmJiIqJiYiKiYqIioiKiAWFBw759/h6+BwVafeE"
    "ioz7aQeDgoB/f4B/gH+BfoF+gn6DfoQFfYR9hXyFfIeLinyHe4h7iHuIeop5inqKiotwjHGPc5BzknWUdZZ3mXeaeZ16nwV7oX2jfqWLjICngamDq4Wth66I"
    "sYqyjLKOsY+uka2Tq5WplqeLjJilmaOboZyfBZ2dn5qfmaGWoZSjkqOQpY+mjIyLmoqbipqKm4maiJqImYeZhoyLmYaZhZiFmYQFmIOYgpiCl4GXgZeAln+W"
    "fr+bf5iLjH+Xiox/l4qLfpaLjH6Wiot9lYqMfZSKiwV9lIqLfJOKjHySiYx8kYqMe5CJjHuQiot6kIqLeo+Ji3qOiox5jYqLeY2Ji3qMBYkGeQaIBmyKh4tu"
    "h4eLb4WHinCDiIpygIiKc36JinR8iYp2eoqKd3iLioqKeXaKigV7dIqKfHKLin1wi4p/bouKgWyDaouKhGmLiodni4qIZYpjjGOOZYuKj2eLipJpBYuKk2qV"
    "bIuKl26Liplwi4qacoyKm3SMip12jIqLip94jIqgeo2KonyNiqN+jooFpICOiqaDj4qnhY+LqIeNi4l/h36HgIaBhYKFgoSDhIODg4SDg4KDg4SCi4qEggWF"
    "gYqKhoGLioeAioqIf4uKiX2LioqDi4qMg4uKjYOLio2EjIqOhYyKj4WMipCGBYyKkYaMipGHjoqRiI6KkomOipOKjoqTiqWLlI2Oi5SNjYuVjo2MlY6Ni5WQ"
    "dJ8FgYiDiIKJhImEin6LhoyGi4eNh42HjYeOiI+Ij4iQiZGKkYqSi5ONmI6Xj5WQlQWRlJGTkZOSk5OTkpOMi5KTi4yTk5KUjIySlIuMkZWMjJCVi4yQl4uM"
    "j5iLjI6ZBZkGjQaejI2Lno2Ni52OjYudj42LnY+Mi5yQjYyckIyMm5GNiwWbkoyMmpKMjJqTjIyZk4yMmZWMi5mVi4yYloyLmJeXl4yMl5eLjJeYjI2MjYyN"
    "BY0H938HjQeKjYuMio2JjYmNiYyKi4mNiIyKi4iMiIwFhwaHBvuiBg75zt/3wBWMcouKjnOLipB0i4qSdIuKk3UFjIoGlXaWdoyKBZh4jIqZeYyKm3qMipt8"
    "jYqdfIyKnn6Nip+AjYqggY6KoYOOiqGEj4qiho+Lo4gFjouPiomAiH6GgIaBhoKEgoSDhIOEg4ODg4KEg4qLhIKLioWCiouFgYuKhoGKigWHgIuKh3+Liol9"
    "i4KFjYCQgJGBkIGSgZGCkoGTgpODk1l5lYKLipWDi4qVg4yLBZWDjIuWg4yLloSMipaFjIqXhYyLmIWMi5eGjYqYhoyLmYeMipmIjIqNipCGjYoFkYeNipKI"
    "joqSiY6KkoqPipOKpYuUjY2LlY2Ni5WOjIyPjKSLjoyljY6Mo4+OiwWjkY6MoZKOjKCUjY2flY2MnpiNjJyZjYybm4yMmZyMjJmei4yXn4yMlaGMjJOjBYyM"
    "BpKkkaaLjI+njIyNqYyrBfirUfsHB4qKB4oGiox9nYqMe5yKjHuaiYwFeZqKjHiYiYx3lomMdpWIjHWTiIx1koeMdJCHi3OOiItyjIeLcoqIi3OIh4t0hgWH"
    "inWEiIp1g4iKdoGJineAiYp4foqKeXyJint8iop7eoqKfXmKin54ioqAdoF2BYqKB4N1i4qEdIuKhnSLiohzi4qKcgX3+PvQFY4Go46Pi6KQj4yhko6MoZOO"
    "jKCVjYyflo2MnpiMjJ2ajYybmoyMm5yMjJmdjIwFjAaMigZZB4psiW6Hb4VxhHOCdIF2gHh+eX58i4p8fXt/e4AFiot6gXmDiot4hXeGdod0iXSKfIx9i32M"
    "i4x9jH6Nfo6CjYmRi5qNmI6Xj5WQlQWQlJGTkpOSk5KTk5OTk4uMkpOMi5KUi4ySlIyMkZWLjJGVi4yQl4uMj5iLjI2ZBfu19+kVjqKPoYuMkqCToIuMlJ+W"
    "npedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OBaCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50jHMFinOIdId1i4qEdoN2i4qC"
    "d4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdop2jAV3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIehiKKKo4yjBQ753t0WxfgwjIz4"
    "xIqM/DDF+fBR/DCKivzEjIr4MFH98Ab4XPo+Fa+Z+xv3FomMiYyJjImMiIx9i4iKiYqJiomKiYr7G/sWr333CfcFBQ75ut8WxffOBpKkk6STo4yLlKKWoJaf"
    "l56YnJiamZmamJqWm5WclJySnJCdkJ6Ono0FjIufjKGKn4mfh52HnYSchJuCmoGZf5h+mH2LipZ8lXmUeZJ3knaQdI50jXKMcQX7ysX3ygeKpomlh6SGo4Sh"
    "i4yDoIuMgZ6LjICeBYuMf5yKjH6aio18mYqMe5iJjHqXiYx5lYmMd5SIjHeSiIx1kYiLdI+IjHONiIwFcgaHBnQGiIp0iQWIi3WHiIp2hoiKd4WKi4mKd4OJ"
    "inmBiYp5gImKen+Kint9iop8fIqKfXqKin56BYqKB4B6BYoGiowG+BxR/fAH+Ej6PhWvmfsb9xaJjImMiYyJjIiMfYuIiomKiYqJiomK+xv7Fq999wn3BQUO"
    "+jDd+SoVabOKjP0HxfgwjIz4xIqM/DDF+QeMjLOtY4yK91lR+1mKivzEjIr3WVH7WYqKYwfuZxWMjIz4xIqM+0eKivzEjIqMBw753935OhWzioz9OcX3zgaS"
    "pJOkk6OMi5SilqCWn5eemJyYmpmZmpialpuVnJSckpyQnZCejp6NBYyLn4yhip+Jn4edh52EnISbgpqBmX+Yfph9i4qWfJV5lHmSd5J2kHSOdI1yjHEF+8rF"
    "98oHiqaJpYekhqOEoYuMg6CLjIGei4yAngWLjH+ciox+moqNfJmKjHuYiYx6l4mMeZWJjHeUiIx3koiMdZGIi3SPiIxzjYiMBXIGhwZ0BoiKdIkFiIt1h4iK"
    "doaIineFiouJineDiYp5gYmKeYCJinp/iop7fYqKfHyKin16iop+egWKigeAegWKBoqMBvdljIz3SK37SIyK9ydR+yeKimNpBw74aPcE+oIVkpKSkJGQkY6R"
    "jo+MkIySi4+KkIqQiJGIkoYFkoaShJKDmnmTgpODjIuShI2LkoWNipOHjYqSiI6Kk4iOipOKjYqji42Mk4yPjAWSjo6Mk46NjJOPjIyTkYyLk5KMi5KTjIuT"
    "lJOVZZWEg4SDhISEhoSGhYiGiIaKBYeKhIuGjIeMhY6FjoWQhJCEkoSTfJ2DlIqLhJOKi4OSiouDkYqMg4+JjIOOiIwFhI6HjIOMiYxzi4mKg4qIioOIiIqE"
    "iImKg4eJioSFiYuEhIqLg4ODgoOBsYGSkwXl/noVxfnwUf3wBg74bPcG+oIVkpKSkJGQkY6Rjo+MkIySi4+KkIqQiJGIkoYFkoaShJKDmnmTgpODjIuShI2L"
    "koWNipOHjYqSiI6Kk4iOipOKjYqji42Mk4yPjAWSjo6Mk46NjJOPjIyTkYyLk5KMi5KTjIuTlJOVZZWEg4SDhISEhoSGhYiGiIaKBYeKhIuGjIeMhY6FjoWQ"
    "hJCEkoSTfJ2DlIqLhJOKi4OSiouDkYqMg4+JjIOOiIwFhI6HjIOMiYxzi4mKg4qIioOIiIqEiImKg4eJioSFiYuEhIqLg4ODgoOBsYGSkwXb+0IVjIiMiIyI"
    "jIiNiIyJjYmNiI2JjoqNiY6JjoqNio6KjouPigWOBo4GjoyOi46MjoyOjI2Njo2NjI2NjY6NjY2NjI6NjoyOjI6LjoyOBY4HjgeKjouOio6KjomOio6JjYmN"
    "iY6JjYmMiI2JjYiMiIyIjIiLiIwFiAaIBoeKiIuIiomKiIqIiYmJiIqJiYmIiYmKiYmIioiKiIqIiogFiAeIB4gHz/3PFfjsUfzsBw74OPf8+o4V+8Bx98AG"
    "+0f+dBXF+fBR/fAGDvg89/76jhX7wHH3wAb7UPs8FYgHjIiMiIyIjYiNiYyJjYiOiY2KjYmOiY6KjoqOio6LjooFjgaOBo6MjouOjI6MjoyOjY2NjYyOjY2O"
    "jI2NjY2OjI6MjoyOi46MjgWOB44Hio6LjoqOio6KjomOiY2KjYmOiI2JjImNiI2IjIiMiIyIi4iMBYgGiAaIioiLiIqIioiKiImJiYmKiImJiIqJiYmJiIqI"
    "ioiKiIuIiogFiAeIB8/9zxX47FH87AcO+GHO+p4VlXyMipd+jIqZf42KmoKNiZyEjoqdhY6KnoaNiwWfiI6Ln4qOi5+Mjouejo6LnpCOjJ2RjYycko6NmpSN"
    "jJmXjIyXmIyMlJqMjZKbBWKRhXuCfoGAf4F+g3yEfIZ7h3qJeop6jHuNe497kH2SfpN/lYCWg5iFm2KFknsF9xr+oBXF+fBR/fAGDvhl0PqeFZV8jIqXfoyK"
    "mX+NipqCjYmchI6KnYWOip6GjYsFn4iOi5+KjoufjI6Lno6Oi56QjoydkY2MnJKOjZqUjYyZl4yMl5iMjJSajI2SmwVikYV7gn6BgH+BfoN8hHyGe4d6iXqK"
    "eox7jXuPe5B9kn6Tf5WAloOYhZtihZJ7BfcR+2gViAeMiIyIjYiMiI2JjYmNiI2JjYqOiY2JjoqOio6KjouOigWOBo8GjoyOi46MjYyOjI6NjY2OjI2NjY6N"
    "jYyNjY6MjoyOjI6MjgWOB44HjgeOB4qOio6KjoqOiY6KjYmNiY6JjYiMiY2IjYiMiYyIjIiLiIwFhwaIBoiKiIuIioiKiIqJiYiJiYqJiYmIiYmJiYqIiYiK"
    "iIqIi4iKiAWIB4gHz/3PFfjsUfzsBw735PcXmxWDfoqLhH6LioR/i4qFf4qLhn6Gf4Z/h3+If4iAi4qKgIqLioAFgAeAB4yBjIGLio2Ci4qOgouKj4KLipCC"
    "jIqQg4yKkYSMipKEjIqSho2KBZOGjYqTiI6Kk4iOi5OJjoqUipqLlIyOi5SMjYyUjY2LlI6NjJOOjYyTj4yMk48FjIyTkIyMkpFpmYWGhIeFh4SIhIiFiYSJ"
    "hYp/i4WMhoyFjYaNhY6FkIaQhZGHkgWGk4iUiJOKlImVipSLlYyVjJWNlo2VjpaPl4+WkJeRl5GXkpiSl5OYk5hllYJ9BUR7FcX58FH98AYO9/L3G44VigaE"
    "fouKhH+LioV/iouGfoZ/hn+Hf4h/iICLioqAiouKgAWAB4AHjIGMgYuKjYKLio6Ci4qPgouKkIKMipCDjIqRhIyKkoSMipKGjYoFk4aNipOIjoqTiI6Lk4mO"
    "ipSKmouUjI6LlIyNjJSNjYuUjo2Mk46NjJOPjIyTjwWMjJOQjIySkWmZhYaEh4WHhIiEiIWJhImFin+LhYyGjIWNho2FjoWQhpCFkYeSBYaTiJSIk4qUiZWK"
    "lIuVjJWMlY2WjZWOlo+Xj5aQl5GXkZeSmJKXk5iTmGWVgn0FOvm/FYyIi4iMiIyIjIiNiI2JjImNiI6JjYqNiY6JjoqOio6KjouOigWOBo4GjoyOi46MjoyO"
    "jI6NjY2NjI6NjY6MjY2NjY6MjoyOjI6LjoyOBY4HjgeKjouOio6KjoqOiY6JjYqNiY6IjYmMiY2IjYiMiIyIjIiLiIwFiAaIBoiKiIuIioiKiIqIiYmJiYqI"
    "iYmIiomJiYmIioiKiIqIi4iKiAWIB4gHz/3PFfjsUfzsBw73WMf6eRWIB4yIjIiMiIyJjYiMiY2IjYmNiY6JjYqOiY2KjoqOio6Kl4uOjI6MjoyNjAWOjY2M"
    "jo2NjY2NjY6MjY2OjI2MjoyOjI6Ll4qOio6KjoqNiY6KjYmOiY2JjYiNBYmMiI2JjIiMiIyIjH+LiIqIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIiogF"
    "hQeU/nwVxfnwUf3wBg73ct0WxfjsUQYO+cr4VpwVjWkFowaOBqKNjouhjo6MoI+OjJ6Qjoyeko2MnZONjJyUjY2alQWNjJiXjIuMjJiYjI2WmYyMlZuMjJSc"
    "i4yTnYuMkZ6MjI+gjIyOoYuMjaKLjIykBfj2Ufz2B4pziXSIdoZ3hXiEeoN7gX2LioF+gICKi3+AfoJ9g3yEe4Z6hnmHd4l3iXSKBfwEehXF+fBR/fAGDviA"
    "9yn7hxWPaaCLjoyfjY+Lno6OjJ6QjoydkY2MnJKNjJuUjYwFmpWMjZmWjIyXmIyMl5mMjJWbi4yUm4yMk52LjJKei4yRn4uMj6GLjI6ijaSMpQX451H85weK"
    "copzh3WHdoZ3hXmKi4R6g3uKi4J9gX6AgH+BfoJ+hH2EfYZ7h3uJeol4igVH94cVxfjsUfzsBg75hdH3PhWTepR7i4qUfIyKlX2LipZ+i4qXfouKl4CMipiA"
    "BYyKmIGMi4yKmYKMipqCjYqahI2Km4SOi5uFjoqcho6LnYeNip6IjoufiY2Kn4oFjgagBo0GowaOBqKNjouhjo6MoI+NjJ+Qjoyeko2MnZONjJyUjY2alQWM"
    "jIyLmJeNjJiYjI2WmYyMlZuMjJSci4yTnYuMkZ6LjJCgi4yPoYuMjaKLjIykBfj2Ufz2B4pzBYl0iHaGd4V4hHqDe4F9i4qBfn+Af4B+gn2DfIR7hnqGeYeK"
    "i3iJd4l1iniMeIwFeo2Ki3uOe46Ki3yQfJB9kH2SfpN+k3+UgJWKi4CWgZeBmIKYiouDmoOag5xTgQX4OfmUFa6Z+xv3FoqMiYyIjImMiIx+i4iKiYqIiomK"
    "ior7G/sWrn33CfcFBQ74OffY+j4Vrpn7G/cWioyJjImMiIyJjH2LiIqJiomKiYqJivsb+xauffcK9wUFxPt3FYyIjIiMiIyIjYiMiYyLjImNiI6JjYqNiY6J"
    "joqOio6KjouOigWOBo4GjoyOi46MjoyOjI6NjY2NjI6NjY6MjY2NjY6MjoyOjI6LjoyOBY4HjgeKjouOio6KjoqOiY6JjYqNiY6IjYmMiY2IjYiMiIyIjIiL"
    "iIwFiAaIBoiKiIuIioiKiIqIiYmJiYqIiYmIiomKi4qJiYiKiIqIioiKiAWIB4gHiAf7Uf7kFaAGjoyfjY+Lno6OjJ6QjoydkY2MnJKNjJuUjYyZlQWMi4yN"
    "mZaMjJeYjIyXmYyMlZuLjJSbjIyTnYuMkp6LjJGfi4yPoYuMjqKNpIylBfjnUfznB4pyinOHdYd2hneEeYR6g3uKi4J9gX6AgH+BfoJ+hH2EfYZ7h3uJeol4"
    "igWKBg757tkWxff2BowH9yz3DgWMBowG+E78eb2b/Fr4iIuM+Dr36V2f/Mr8XAWKBoqMBvhRUf3wB/fcbxWHfoeAhoGFgoSChYODg4SDg4OEgoqLhIOEgoqK"
    "hYKKiwWFgYuKhoGLioaAi4qIf4uKiH2LgYyDi4qMg4yKjYSMio6FjIqPhYyKj4aNipCGBY2KkYeOipGIjoqSiY6Kk4qOipOKpYuUjY2LlY2Ni5WOjYyVjoyL"
    "lpB0n4GIgogFg4mDiYSKf4uGjIaLh42HjYeNh46Ij4ePiZCJkYmRi5qNmI6Xj5WQlZCUkZOSkwWSk5OTkpOTk4uMk5OSlIyMkZSMjJGVi4yRlYuMkJeLjI+Y"
    "i4yOmYuMjJphjYp7BQ75q9cWxfeHBoyM9xziBYwGjAb4Ivvpu5/8KffvBYqMjIwG9/n3eWGj/IH70YqLiowF+MhR/fAH971vFYd+h4CGgYWChYKEg4SDg4OD"
    "g4SCg4OEgoqKhYKKi4aBiooFhoGLioaAi4qIf4uKiX2KiouCjIOLio2Di4qNhIyKjoWMio+FjIqQhoyKkIaNigWRh46KkYiOipKJjoqTio6Kk4qli5SNjouU"
    "jY2LlY6NjJWOjIuWkHSfgYiCiIOJBYSJg4p/i4aMhouHjYeNh42HjoiPiI+IkImRipGKkouTjZiOl4+VkJWQlJKTkZMFkpOTk5KTk5OLjJOTkpSMjJGUjIyR"
    "lYuMkZWLjJCXi4yPmIuMjpmLjIyaYo2JewUO+aLdFsX3hwaMjAf3GeQFjIyKB/gN++u7n/wU9/IFioyMjAb35fd2X6P8Z/vPBYoGiowG98JR/OwHDvmt2RaJ"
    "B4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+QCt/OKMivneUf3wBvgS+tgV+zn7IK599zn3IAUO99P3dfrYFfs5+yCtffc59yAF+w3+yhXF+fBR/fAG"
    "Dvmt2RaJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G96UGiYCHfoeAhoGFgoWChIOEg4ODhIODgoODhIKLioSCBYWBioqGgYuKh4CKioh/i4qJfYuK"
    "ioOLioyDi4qNg4uKjYSMio6FjIqPhYyKkIYFjIqRhoyKkYeOipGIjoqSiY6Kk4qOipOKpYuUjY6LlI2Ni5WOjYyVjo2LlZB0nwWBiIOIgomEiYSKfouGjIaL"
    "h42HjYeNh46Ij4iPiJCJkYqRipKLk42YjpePlZCVBZGUkZORk5KTk5OSk4yLkpOLjJOTkpSMjJKUi4yRlYyMkJWLjJCXi4yPmIuMjpkF98Wt/OKMivneUf3w"
    "Bg73rPc2FpMGinyIfoh+h4CFgYaChIKFg4ODhIODg4SCiouEg4SCioqFggWKi4WBi4qGgYqKh4CLioh/ioqJfYuBjIOLioyDjIqNhIyKjoWMio+FjIqPho2K"
    "BZCGjYqRh42KkoiOipKJjoqTio6Kk4qli5SNjYuVjY2LlY6NjJWOjIuWkHSfgYgFgoiDiYOJhIp/i4aMhouHjYeNh42HjoePiI+JkImRiZGLmo2YjpePlZCV"
    "kJSRkwWSk5KTkpOTk5OTi4yTk5KUi4ySlIyMkZWLjJGVi4yQl4uMj5iLjI2ZjIyMmnaMBaj58FH98AYO+a3ZFokHjImLioyJjYmNiY2KjIuNiY6KjIuOio6K"
    "BY8Gjwb5AK384oyK+d5R/fAG98j6PRWNio6KjYqOipiLjoyNjI6MjYyMjPcb9xZomfsJ+wX7CvcFaH33G/sWBQ74PfdZ+j0VjYqNio2KjoqZi42MjoyNjI2M"
    "jIz3G/cWaJn7CfsF+wr3BWh99xv7FgWA/j4VxfnwUf3wBg76BfmN+GcVgQeKiomKiYqJiomKiomJioqJiomKiYqJiomKiQWJB4gHiQeJB4gHiQeMiYyJjImM"
    "iYyJjImNioyJjYqNio2KjYqNio2KBY0GjgaNBo0GjgaNBo2MjYyNjI2MjYyNjIyNjYyMjYyNjI2MjYyNjI0FjQeOB40HjQeOB40Hio2KjYqNio2KjYqNiYyK"
    "jYmMiYyJjImMiowFlXcH/Tv4HRX98AeJB4yJi4qMiY2JjYmNioyLjYmOioyLjoqOigWPBo8G+QCt/OKMivneUQYO+C/3t/hnFYEHioqJiomKiYqJioqJiYqK"
    "iYqJiomKiYqJiokFiQeIB4kHiQeIB4kHjImMiYyJjImMiYyJjYqMiY2KjYqNio2KjYqNigWNBo4GjQaNBo4GjQaNjI2MjYyNjI2MjYyMjY2MjI2MjYyNjI2M"
    "jYyNBY0HjgeNB40HjgeNB4qNio2KjYqNio2KjYmMio2JjImMiYyJjIqMBZV3B/tl/GcVxfnwUf3wBg756t33uRWycYyLnpYFjAaMigb7qQeJB4yJi4qMiY2J"
    "jYmNio6JjoqPio6KBY8Gjwb47K38zgaKjAX3uQeMB/eY9yWLi2Sl+2/7DwWKBoqMBviHUfypB4qKT2oFDvhZ3fejFbd1oZoFjAaMigb7m8X3xQeMB/c39wVf"
    "ofsJOgWKBoqMBviHUfyxB4qKB0hcBQ753t0WxfmYBoyMB4wG+Mb9oI2JjYmOio6JjoqOio+KBZIGjwaPBo6Mj4yOjI6NjYyNjY2NjI2LjIyNBY0H+fBR/ZgH"
    "iooHigb8xvmgiY2JjYiMiI2IjIiMh4wFhAaHBocGiIqHioiKiImJiomJiYmKiYuKiokFiQf98Af4IfrYFfs5+yCtffc59yAFDvm63xbF984GkqSTpJOjjIuU"
    "opaglp+XnpicmJqZmZqYmpablZyUnJKckJ2Qno6ejQWMi5+MoYqfiZ+HnYedhJyEm4KagZl/mH6YfYuKlnyVeZR5kneSdpB0jnSNcoxxBfvKxffKB4qmiaWH"
    "pIajhKGLjIOgi4yBnouMgJ4Fi4x/nIqMfpqKjXyZiox7mImMepeJjHmViYx3lIiMd5KIjHWRiIt0j4iMc42IjAVyBocGdAaIinSJBYiLdYeIinaGiIp3hYqL"
    "iYp3g4mKeYGJinmAiYp6f4qKe32Kinx8iop9eoqKfnoFiooHf3eKi392i4qAdouKgXSLioJzi4qDcYNwBYgH+88H+A362BX7OfsgrX33OfcgBQ753t0WxfmY"
    "BoyMB4wG+Mb9oI2JjYmOio6JjoqOio+KBZIGjwaPBo6Mj4yOjI6NjYyNjY2NjI2LjIyNBY0H+fBR/ZgHiooHigb8xvmgiY2JjYiMiI2IjIiMh4wFhAaHBocG"
    "iIqHioiKiImJiomJiYmKiYuKiokFiQf98Af3zm8ViH6GgIaBhoKEgoSDhIOEg4ODg4KEg4qLhIKLioWCiouFgQWLioaBioqHgIuKh3+Liol9i4GMg4uKjIOM"
    "io2Ei4qPhYuKj4WMipCGjYqQho2KBZGHjYqSiI6KkomOipKKj4qTiqWLlI2Ni5WNjYuVjoyMlo6Mi5aQc5+CiIKIg4kFg4mEin+LhoyGi4eNho2HjYiOh4+I"
    "j4mQiZGJkYuajZiOl4+VkJWQlJGTkpOSkwWSk5OTk5OLjJKTjIuSlIuMkpSMjJGVi4yRlYuMkJeLjI+Yi4yNmYuMjZphjYp7BQ75ut8WxffOBpKkk6STo4yL"
    "lKKWoJafl56YnJiamZmamJqWm5WclJySnJCdkJ6Ono0FjIufjKGKn4mfh52HnYSchJuCmoGZf5h+mH2LipZ8lXmUeZJ3knaQdI50jXKMcQX7ysX3ygeKpoml"
    "h6SGo4Shi4yDoIuMgZ6LjICeBYuMf5yKjH6aio18mYqMe5iJjHqXiYx5lYmMd5SIjHeSiIx1kYiLdI+IjHONiIwFcgaHBnQGiIp0iQWIi3WHiIp2hoiKd4WK"
    "i4mKd4OJinmBiYp5gImKen+Kint9iop8fIqKfXqKin56BYqKB393iot/douKgHaLioF0i4qCc4uKg3GDcAWIB/vPB/e6bxWIfoaAhoGGgoSChIOEg4SDg4OD"
    "goSDiouEgouKhYKKi4WBBYuKhoGKioeAi4qHf4uKiX2LgYyDi4qMg4yKjYSLio+Fi4qPhYyKkIaNipCGjYoFkYeNipKIjoqSiY6KkoqPipOKpYuUjY2LlY2N"
    "i5WOjIyWjoyLlpBzn4KIgoiDiQWDiYSKf4uGjIaLh42GjYeNiI6Hj4iPiZCJkYmRi5qNmI6Xj5WQlZCUkZOSk5KTBZKTk5OTk4uMkpOMi5KUi4ySlIyMkZWL"
    "jJGVi4yQl4uMj5iLjI2Zi4yNmmGNinsFDvne3RbF+ZgGjIwHjAb4xv2gjYmNiY6KjomOio6Kj4oFkgaPBo8GjoyPjI6Mjo2NjI2NjY2MjYuMjI0FjQf58FH9"
    "mAeKigeKBvzG+aCJjYmNiIyIjYiMiIyHjAWEBocGhwaIioeKiIqIiYmKiYmJiYqJi4qKiQWJB/3wB/fX+j0VjYqNio2KjoqZi46MjYyNjI2MjYz3G/cWZ5n7"
    "CfsF+wn3BWd99xv7FgUO+brfFsX3zgaSpJOkk6OMi5SilqCWn5eemJyYmpmZmpialpuVnJSckpyQnZCejp6NBYyLn4yhip+Jn4edh52EnISbgpqBmX+Yfph9"
    "i4qWfJV5lHmSd5J2kHSOdI1yjHEF+8rF98oHiqaJpYekhqOEoYuMg6CLjIGei4yAngWLjH+ciox+moqNfJmKjHuYiYx6l4mMeZWJjHeUiIx3koiMdZGIi3SP"
    "iIxzjYiMBXIGhwZ0BoiKdIkFiIt1h4iKdoaIineFiouJineDiYp5gYmKeYCJinp/iop7fYqKfHyKin16iop+egWKigd/d4qLf3aLioB2i4qBdIuKgnOLioNx"
    "g3AFiAf7zwf3w/o9FY2KjYqNio6KmYuOjI2MjYyNjI2M9xv3FmeZ+wn7BfsJ9wVnffcb+xYFDvm63xbF984GkqSTpJOjjIuUopaglp+XnpicmJqZmZqYmpab"
    "lZyUnJKckJ2Qno6ejQWMi5+MoYqfiZ+HnYedhJyEm4KagZl/mH6YfYuKlnyVeZR5kneSdpB0jnSNcoxxBfvKxffKB4qmiaWHpIajhKGLjIOgi4yBnouMgJ4F"
    "i4x/nIqMfpqKjXyZiox7mImMepeJjHmViYx3lIiMd5KIjHWRiIt0j4iMc42IjAVyBocGdAaIinSJBYiLdYeIinaGiIp3hYqLiYp3g4mKeYGJinmAiYp6f4qK"
    "e32Kinx8iop9eoqKfnoFiooHf3eKi392i4qAdouKgXSLioJzi4qDcYNwBYgH+88HDvqK3RbF+ZgGjIwHjAb4xv2gjYmNiY6KjomOio6Kj4oFkgaPBo8GjoyP"
    "jI6Mjo2NjI2NjY2MjYuMjI0FjQf5RgeMnwWMno6cj5yQmpCZjIyRmJOXk5aUlIuMlJOMi5WTlZKWkJeQl4+YjpiNmYybjImtBXkGiAZ6iYiLeoiIinyIiIp8"
    "hoiKfYaIiX6FiYl+g4mKf4KKioCBioqBgAWKigaBf4uJg36KioR9ioqFfIuKhnuLiod6ioqIeYuKineKdgX87geKigeKBvzG+aCJjYmNiIyIjYiMiIyHjAWE"
    "BocGhwaIioeKiIqIiYmKiYmJiYqJi4qKiQWJB/3wBw75tt0WxffOBpKkk6STo4yLlKKWoJafl56YnJiamZmamJqWm5WclJySnJCdkJ6Ono2Mi5+MBaGKn4mf"
    "h52HnYSchJuCmoGZf5h+mH2LipZ8lXmUeZJ3knaQdI50jXKMcYv7yopyBYpziHWHdoZ4i4qGeYR7g3yLioJ+gn6AgICCf4OLin+EfoZ9hnyHe4l6iXiKjWkF"
    "oQaOBp+NBY6Mno6PjJ2PjoyckY6Mm5KNjJqUjYyZlY2MmJaMjZeXjIyVmYyMlJqMjJObjIwFkpyLjJGejIyQn4uMj6CLjI6ijKSMpYv3yoqmiaWHpIajhKGL"
    "jIOgi4yBnouMgJ4Fi4x/nIqMfpqKjXyZiox7mImMepeJjHmViYx3lIiMd5KIjHWRiIt0j4iMc42IjAVyBocGdAaIinSJBYiLdYeIinaGiIp3hYqLiYp3g4mK"
    "eYGJinmAiYp6f4qKe32Kinx8iop9eoqKfnoFiooHf3eKi392i4qAdouKgXSLioJzi4qDcYNwBYgH+88HDvoa3fhCFYxnjmiMi5Bpi4qTa4uKlGyLipdti4qY"
    "bouKmnCLipxxi4qdcwWMip51jIqfdoyKoXiNiqF7jYqjfI6Ko3+OiqWCj4qlhJCKpoeQiqeKj4unjJCMBaaPkIylko+MpZSNjIyLo5eOjKOajIyMi6GbjYyh"
    "noyMn6CMjJ6hjIydo4uMnKUFi4yapouMmKiLjJepi4yUqouMk6uLjJCtj66Mr4qvh66GrYuMg6uLjIKqi4x/qQWLjH6oi4x8pouMeqWLjHmjiox4oYqMd6CK"
    "jHWeiYx1m4qLioxzmoiMc5eKi4mMBXGUh4xxkoaMcI+GjG+Mh4tvioaKcIeGinGEh4pxgoiKc3+IinN8iYp1e4mKdXgFiop3doqKeHWKinlzi4p6cYuKfHCL"
    "in5ui4p/bYuKgmyLioNri4qGaYqLiGiKZwXFFoyujqyRrJKrlKqWqJiomaWbpJyjnaAFn5+fnKCaoZihlqKToZGijqKNoomiiKGFooOhgKF+oHyfep93nXac"
    "c5tymXGYbgWWbpRskmuRao5qjGiKaIhqhWqEa4JsgG5+bn1xe3J6c3l2d3d3enZ8dX51gHSDBXWFdIh0iXSNdI51kXSTdZZ1mHaad5x3n3mgeqN7pH2lfqiA"
    "qIKqhKuFrIisiq4F+GH44BX7wHH3wAYO+eLh98AVjHKLio5zi4qQdIuKknSLipR1i4qWdouKl3eLipl4i4qaeYyKm3qMigWcfI2KnXyNip5+jYqggI2KoYGN"
    "iqGCjoujhI6Ko4aOi6SIj4ukio+LpIyPi6SOBY6Lo5COjKOSjYuilI2MoZWNjKCWjYyemI2MnZqNjJyajIybnIyMmp2LjJmei4wFl5+LjJagi4yUoYuMkqKL"
    "jJCii4yOo4uMjKSKpIuMiKOLjIaii4yEoouMgqGLjAWAoIuMf5+LjH2ei4x8nYqMe5yKjHqaiYx5momMeJiJjHaWiYx1lYmMdJSJi3OSBYiMc5CIi3KOh4ty"
    "jIeLcoqHi3KIiItzhoiKc4SIi3WCiYp1gYmKdoCJinh+iYoFeXyJinp8iop7eoqKfHmLin14i4p/d4uKgHaLioJ1i4qEdIuKhnSLiohzi4qKcgXFFoyjjqIF"
    "kKGLjJGgk6CVoJaemJ2ZnJqbm5qcmJ2XnpWelIyLnpKgkZ+PoI6Mi6CMoIqMiwWfiIyLn4eghZ6EjIuegp6BnX+cfpt8mnuZeph5lniVdpN2kXaLipB1jnSM"
    "c4pzBYh0hnWLioV2g3aBdoB4fnl9enx7e3x6fnl/eIF4goqLeIR2hXeHiot3iIqLdooFdoyKi3aOd492kXiSiot4lHiVeZd6mHuafJt9nH6dgJ6BoIOghaCL"
    "jIahiKKKowX4QfliFfvAcffABg76Gt34QhWMZ45ojIuQaYuKk2uLipRsi4qXbYuKmG6Lippwi4qccYuKnXMFjIqedYyKn3aMiqF4jYqhe42Ko3yOiqN/joql"
    "go+KpYSQiqaHkIqnio+Lp4yQjAWmj5CMpZKPjKWUjYyMi6OXjoyjmoyMjIuhm42MoZ6MjJ+gjIyeoYyMnaOLjJylBYuMmqaLjJioi4yXqYuMlKqLjJOri4yQ"
    "rY+ujK+Kr4euhq2LjIOri4yCqouMf6kFi4x+qIuMfKaLjHqli4x5o4qMeKGKjHegiox1nomMdZuKi4qMc5qIjHOXiouJjAVxlIeMcZKGjHCPhoxvjIeLb4qG"
    "inCHhopxhIeKcYKIinN/iIpzfImKdXuJinV4BYqKd3aKinh1iop5c4uKenGLinxwi4p+bouKf22LioJsi4qDa4uKhmmKi4hoimcFxRaMro6skaySq5SqlqiY"
    "qJmlm6Sco52gBZ+fn5ygmqGYoZaik6GRoo6ijaKJooihhaKDoYChfqB8n3qfd512nHObcplxmG4Flm6UbJJrkWqOaoxoimiIaoVqhGuCbIBufm59cXtyenN5"
    "dnd3d3p2fHV+dYB0gwV1hXSIdIl0jXSOdZF0k3WWdZh2mnecd595oHqje6R9pX6ogKiCqoSrhayIrIquBfco+PAVlHyNipZ+jYqYf42Km4KNiZyEjYqehY2K"
    "noaOiwWfiI6LnoqPi56Mjoufjo6LnpCNjJ6RjYycko2Nm5SNjJiXjYyWmI2MlJqMjZGbBWORhHuDfoGAf4F9g32Ee4Z7h3uJeop6jHuNe497kH2SfZN/lYGW"
    "g5iEm2OFkXsFDvni4ffAFYxyi4qOc4uKkHSLipJ0i4qUdYuKlnaLipd3i4qZeIuKmnmMipt6jIoFnHyNip18jYqefo2KoICNiqGBjYqhgo6Lo4SOiqOGjouk"
    "iI+LpIqPi6SMj4ukjgWOi6OQjoyjko2LopSNjKGVjYyglo2MnpiNjJ2ajYycmoyMm5yMjJqdi4yZnouMBZefi4yWoIuMlKGLjJKii4yQoouMjqOLjIykiqSL"
    "jIiji4yGoouMhKKLjIKhi4wFgKCLjH+fi4x9nouMfJ2KjHuciox6momMeZqJjHiYiYx2lomMdZWJjHSUiYtzkgWIjHOQiItyjoeLcoyHi3KKh4tyiIiLc4aI"
    "inOEiIt1gomKdYGJinaAiYp4fomKBXl8iYp6fIqKe3qKinx5i4p9eIuKf3eLioB2i4qCdYuKhHSLioZ0i4qIc4uKinIFxRaMo46iBZChi4yRoJOglaCWnpid"
    "mZyam5uanJidl56VnpSMi56SoJGfj6COjIugjKCKjIsFn4iMi5+HoIWehIyLnoKegZ1/nH6bfJp7mXqYeZZ4lXaTdpF2i4qQdY50jHOKcwWIdIZ1i4qFdoN2"
    "gXaAeH55fXp8e3t8en55f3iBeIKKi3iEdoV3h4qLd4iKi3aKBXaMiot2jnePdpF4koqLeJR4lXmXeph7mnybfZx+nYCegaCDoIWgi4yGoYiiiqMF9wj5chWU"
    "fI2Kln6Niph/jYqbgo2JnISNip6FjYqeho6LBZ+Ijoueio6Ln4yOi5+OjouekI2MnpGNjJySjY2blI2MmJeNjJaYjYyUmoyNkZsFY5GEe4N+gYB/gX2DfYR7"
    "hnuHe4l6inqMe417j3uQfZJ9k3+VgZaDmISbY4WRewUO+hrd+EIVjGeOaIyLkGmLipNri4qUbIuKl22Liphui4qacIuKnHGLip1zBYyKnnWMip92jIqheI2K"
    "oXuNiqN8joqjf46KpYKPiqWEkIqmh5CKp4qPi6eMkIwFpo+QjKWSj4yllI2MjIujl46Mo5qMjIyLoZuNjKGejIyfoIyMnqGMjJ2ji4ycpQWLjJqmi4yYqIuM"
    "l6mLjJSqi4yTq4uMkK2Proyviq+Hroati4yDq4uMgqqLjH+pBYuMfqiLjHymi4x6pYuMeaOKjHihiox3oIqMdZ6JjHWbiouKjHOaiIxzl4qLiYwFcZSHjHGS"
    "hoxwj4aMb4yHi2+Khopwh4aKcYSHinGCiIpzf4iKc3yJinV7iYp1eAWKind2iop4dYqKeXOLinpxi4p8cIuKfm6Lin9ti4qCbIuKg2uLioZpiouIaIpnBcau"
    "FY6skaySq5SqlqiYqJmlm6Sco52gBZ+fn5ygmqGYoZaik6GRoo6ijaKJooihhaKDoYChfqB8n3qfd512nHObcplxmG4Flm6UbJJrkWqOaoxoimiIaoVqhGuC"
    "bIBufm59cXtyenN5dnd3d3p2fHV+dYB0gwV1hXSIdIl0jXSOdZF0k3WWdZh2mnecd595oHqje6R9pX6ogKiCqoSrhayIrIquBff3+JIV6vcgZZUs+yAF+y8W"
    "sYHq9yBllQUO+eLh98AVjHKLio5zi4qQdIuKknSLipR1i4qWdouKl3eLipl4i4qaeYyKm3qMigWcfI2KnXyNip5+jYqggI2KoYGNiqGCjoujhI6Ko4aOi6SI"
    "j4ukio+LpIyPi6SOBY6Lo5COjKOSjYuilI2MoZWNjKCWjYyemI2MnZqNjJyajIybnIyMmp2LjJmei4wFl5+LjJagi4yUoYuMkqKLjJCii4yOo4uMjKSKpIuM"
    "iKOLjIaii4yEoouMgqGLjAWAoIuMf5+LjH2ei4x8nYqMe5yKjHqaiYx5momMeJiJjHaWiYx1lYmMdJSJi3OSBYiMc5CIi3KOh4tyjIeLcoqHi3KIiItzhoiK"
    "c4SIi3WCiYp1gYmKdoCJinh+iYoFeXyJinp8iop7eoqKfHmLin14i4p/d4uKgHaLioJ1i4qEdIuKhnSLiohzi4qKcgXGoxWOogWQoYuMkaCToJWglp6YnZmc"
    "mpubmpyYnZeelZ6UjIuekqCRn4+gjoyLoIygioyLBZ+IjIufh6CFnoSMi56CnoGdf5x+m3yae5l6mHmWeJV2k3aRdouKkHWOdIxzinMFiHSGdYuKhXaDdoF2"
    "gHh+eX16fHt7fHp+eX94gXiCiot4hHaFd4eKi3eIiot2igV2jIqLdo53j3aReJKKi3iUeJV5l3qYe5p8m32cfp2AnoGgg6CFoIuMhqGIooqjBffX+RQV6vcg"
    "ZZUs+yAF+y8WsYHq9yBllQUOHAUD+av53xX4Lq38Lgb98ARp+C6t/C4H/Vn4MRWMZ45ojIuQaYuKk2uLipRsBYuKl22Liphui4qacIuKnHGLip1zjIqedYyK"
    "n3aMiqF4jYqhe42Ko3yOiqN/jooFpYKPiqWEkIqmh5CKp4qPi6eMkIymj5CMpZKPjKWUjYyMi6OXjoyjmoyMjIuhmwWNjKGejIyfoIyMnqGMjJ2ji4ycpYuM"
    "mqaLjJioi4yXqYuMlKqLjJOri4yQrY+uBZ2MjPe3rfu3jIqdB4euhq2LjIOri4yCqouMf6kFi4x+qIuMfKaLjHqli4x5o4qMeKGKjHegiox1nomMdZuKi4qM"
    "c5qIjHOXiouJjAVxlIeMcZKGjHCPhoxvjIeLb4qGinCHhopxhIeKcYKIinN/iIpzfImKdXuJinV4BYqKd3aKinh1iop5c4uKenGLinxwi4p+bouKf22LioJs"
    "i4qDa4uKhmmKi4hoimcF1vcZFZSqlqiYqJmlm6Sco52gn5+fnKCaBaGYoZaik6GRoo6ijaKJooihhaKDoYChfqB8n3qfd512nHObcplxmG6WbpRskmsFkWqO"
    "aoxoimiIaoVqhGuCbIBufm59cXtyenN5dnd3d3p2fHV+dYB0g3WFdIh0iQV0jXSOdZF0k3WWdZh2mnecd595oHqje6R9pX6ogKiCqoSrhayIrIqujK6OrJGs"
    "BQ4cBVfd98AVjHKLio5zi4qQdIuKknSLipN1BYyKBpV2lnaMiph4jIqZeYyKm3qMipt8jYqdfAWMip5+jYqfgI2KoIGOiqGDjoqhhI+KooaPi6OIjoukio+L"
    "pIyOi6OOj4uikI+MBaGSjoyhk46MoJWNjJ+WjYyemIyMnZqNjJuajIybnIyMmZ2MjJiei4yXoI6RjIwFjIoGjoSMipV3jIqXeIyKmHkFjIqaeoyKm3yMiouK"
    "nH2Nip1/jYmegI2KoIGOiqCDjoqihY6KooaPi6OIj4ukigWOBpoGjQaajI2LmoyNi5mNjYuZjo2LmY6NjJiOjYyYj42LBZiQjIuYkIyMmJGMi5eSjIuXkoyM"
    "l5KMjJaTjIuWlIyMlpSMi5aVi4yWlYyLlZYFWZ2BgICBgYKBgoCDgISBhICFgIWAhn+GgId/iICIf4iKi3+Jfop+iX6LfoqKiwV2jHaOd494kIqLeZJ4lHqV"
    "epaLjHuXi4x8mX6bfpyAnYqLgZ6Dn4OhhqGGoomjBZEHjIwH+JYGjwaPBo6Mj4yOjI6NjY2NjIuMjY2MjYyNBYuNiqaJpYuMh6OLjIajiouFooqMg6CLjIGf"
    "i4yAnouMf52KjH6biox8moqMe5kFiox6mImMeZWLjImMeJSJjHeTiIx2kYeMdY+IjHSNh4t0jIeLc4qIi3SJh4p1hwWIinWFiIp3g4iKd4KJiniAiYp5f4qK"
    "en2Kint8iop9eoqKfnmKin94gXaKioeCBYoGigaJj3+gi4x+noqMfZ2KjHuciox7momMBXmaiox4mImMd5aJjHaViIx1k4iMdZKHjHSQh4tzjoiLcoyHi3KK"
    "iItziIeLdIYFh4p1hIiKdYOIinaBiYp3gImKeH6Kinl8iYp7fIqKe3qKin15iop+eIqKgHaBdgWKigeDdYuKhHSLioZ0i4qIc4uKinIF+SelFY2kkKKLjJCh"
    "k6GTn4uMlZ6WnZibi4yYmpqZmpeblpyVnJOdkp6QBZ2PjIuejZ+Mn4qeiZ2HnYachJuDm4KagJl+mX2XfJd6lXmUeJN2kXaQdI9zjXIFgweKigf8dwaKjAaM"
    "B/zppxWPoYuMkqCToIuMlJ+WnpedmZyam5qajIubmJ2XnZWelJ6Sn5Gfj5+OBaCMoIqfiJ+Hn4WehJ6CnYGdf5t+m3yae5l6l3mWeJR3i4qTdpJ2i4qPdY50"
    "jHMFinOIdId1i4qEdoN2i4qCd4B4f3l9enx7e3x7fnl/eYF4gniEd4V3h3eIdop2jAV3jnePd5F4kniUeZV5l3uYiot8mnybfZx/nYCegp+LjIOghKCLjIeh"
    "iKKKo4yjBQ75+t0WxffHjIz3mAaMBpcGjAb33/vRvZ370PfCBYwHjIwHl40FjoulkI2MpJGNjKKTjYyhlI2Mn5aNjJ2XjYybmI2MmpqMjJmajI2Xm4yMlZ2M"
    "jAWUnouMkp+MjJCgi4yQoo2jjKSKpImji4yGoYuMhqCKjISfi4yCn4qMgZ2KjH+cBYqMfZuKjHyaiox6mYmMeZeJjHeWiYx2lYiMdZOIjHOSiItxkYiLcI+J"
    "i26NiIwFbQaKBvu2ewaKigdv/fAG9+753hWliaSHooeghZ+DnoOdgZuAmn+Zfph8lnuVepN5kneRd491jXWMc4p0i4oFiXWHdoV2hHiDeYF7i4qAfH59fn58"
    "f4qLe4B6goqLeYN3hIqLdoV0hnOIcYluigX7l4yK+IeMjPeXBqeKBdD3jhX7OfsgrX33OfcgBQ74ss8WxffZBoyjjaOPoZCgkZ+TnpSclZyWmpeZjIuYl5iW"
    "mpWak5uSm5GMi5uQnY6djZ+MBfcIrfsJBokGdQaIinWJiIt2h4iKiot3hoiKeISJiniDiYp6gYmKeoCKint+BYqKfX6LioqKfn2LioqKf3uKioB6i4qBeIN3"
    "i4qEdouKhXaLiod0i4qJc4uKinIF+9kH95j62BX7OfsgrX33OfcgBQ75+t0WxffHjIz3mAaMBpcGjAb33/vRvZ370PfCBYwHjIwHl40FjoulkI2MpJGNjKKT"
    "jYyhlI2Mn5aNjJ2XjYybmI2MmpqMjJmajI2Xm4yMlZ2MjAWUnouMkp+MjJCgi4yQoo2jjKSKpImji4yGoYuMhqCKjISfi4yCn4qMgZ2KjH+cBYqMfZuKjHya"
    "iox6mYmMeZeJjHeWiYx2lYiMdZOIjHOSiItxkYiLcI+Ji26NiIwFbQaKBvu2ewaKigdv/fAG9+753hWliaSHooeghZ+DnoOdgZuAmn+Zfph8lnuVepN5kneR"
    "d491jXWMc4p0i4oFiXWHdoV2hHiDeYF7i4qAfH59fn58f4qLe4B6goqLeYN3hIqLdoV0hnOIcYluigX7l4yK+IeMjPeXBqeKBX39+hWIfoaAhoGFgoWChIOE"
    "g4SDg4ODgoSDiouEgouKhIIFhYGLioWBi4qHgIuKh3+Liol9i3iNg4uKjoSLio6FjIqPhYyKkIaMipGGjYqRhwWNipKIjYqTiY6KkoqOipSKpIuVjY2LlY2N"
    "i5WOjIyVjo2LlZB0n4KIgoiDiYOJBYSKf4uFjIeLho2HjYeNiI6Hj4iPiJCJkYqRipKMk42YjpePlY+VkZSRk5KTkpMFkpOTk5KTjIySk4yLkpSLjJKUi4yR"
    "lYyMkJWMjI+XjIyOmIyMjZmLjI2aYY2KewUO+LLPFsX32QaMo42jj6GQoJGfk56UnJWclpqXmYyLmJeYlpqVmpObkpuRjIubkJ2OnY2fjAX3CK37CQaJBnUG"
    "iIp1iYiLdoeIioqLd4aIiniEiYp4g4mKeoGJinqAiop7fgWKin1+i4qKin59i4qKin97ioqAeouKgXiDd4uKhHaLioV2i4qHdIuKiXOLiopyBfvZB/dFbxWI"
    "foaAhoGGgoSChIOEg4SDg4ODgoSDiouEgouKhYKKi4WBBYuKhoGKioeAi4qHf4uKiX2LgYyDi4qMg4yKjYSLio+Fi4qPhYyKkIaNipCGjYoFkYeNipKIjoqS"
    "iY6KkoqPipOKpYuUjY2LlY2Ni5WOjIyWjoyLlpBzn4KIgoiDiQWDiYSKf4uGjIaLh42GjYeNiI6Hj4iPiZCJkYmRi5qNmI6Xj5WQlZCUkZOSk5KTBZKTk5OT"
    "k4uMkpOMi5KUi4ySlIyMkZWLjJGVi4yQl4uMj5iLjI2Zi4yNmmGNinsFDvn63RbF98eMjPeYBowGlwaMBvff+9G9nfvQ98IFjAeMjAeXjQWOi6WQjYykkY2M"
    "opONjKGUjYyflo2MnZeNjJuYjYyamoyMmZqMjZebjIyVnYyMBZSei4ySn4yMkKCLjJCijaOMpIqkiaOLjIahi4yGoIqMhJ+LjIKfioyBnYqMf5wFiox9m4qM"
    "fJqKjHqZiYx5l4mMd5aJjHaViIx1k4iMc5KIi3GRiItwj4mLbo2IjAVtBooG+7Z7BoqKB2/98Ab37vneFaWJpIeih6CFn4Oeg52Bm4Caf5l+mHyWe5V6k3mS"
    "d5F3j3WNdYxzinSLigWJdYd2hXaEeIN5gXuLioB8fn1+fnx/iot7gHqCiot5g3eEiot2hXSGc4hxiW6KBfuXjIr4h4yM95cGp4oFhuoVjYqNio2KjoqZi42M"
    "joyNjI2MjIz3G/cWaJn7CfsF+wr3BWh99xv7FgUO+LLPFsX32QaMo42jj6GQoJGfk56UnJWclpqXmYyLmJeYlpqVmpObkpuRjIubkJ2OnY2fjAX3CK37CQaJ"
    "BnUGiIp1iYiLdoeIioqLd4aIiniEiYp4g4mKeoGJinqAiop7fgWKin1+i4qKin59i4qKin97ioqAeouKgXiDd4uKhHaLioV2i4qHdIuKiXOLiopyBfvZB/dO"
    "+j0VjYqNio2KjoqZi46MjYyNjI2MjYz3G/cWZ5n7CfsF+wn3BWd99xv7FgUO+enb9wIVl4CMi5iAmIGMipmBjIuZgouKmoOMipqDjIoFm4SMipuEjIubhI2L"
    "nIWMipyGjYudhoyLnoaMi56IjYqeiI2Ln4mMi6CJjIugigWNBqAGjQanBo0GpY2Ni6WNjYujj42Lo4+NjKGQjYyfkY6MnpKNjAWdk42Mm5SNjJqVjYyZloyM"
    "l5aMjZaXjI2UmIyMkpmMjZGai4yPm4yMjZyLjIycBYwHjAeKnYuMBYmci4yKjIebi42Fm4qMhJqKjIuMgpmKjIGZio1/mImMfpeJjHyXiox6loqMeZYFiYwG"
    "d5SKjHaUiYx1kwWKi3OTiYtzkomMcZGJi3CRiotukHCPcZBzkHSQdpB3kXiRepF8k4qLfZJ+lH+UBYGVgpaKi4SYiouFmYWah5yJnYqfjJ+LjI2ej56RnJKc"
    "k5uVmYuMlpiXmJmWmZYFm5Sck4uMnZKekZ+QoI+ijqKNpIybipuLm4maiZqJmoiZh5mHmYeZhpmFmISYhQWLipiEmIOXgpiCl4GXgJaAjIu8nX+Xiot/l4qL"
    "fpZ9lYuMfZSKi36Uiox9k4qLBX2Tiot8koqMfJGKjHuQiox8kImMe4+Ki3uPiYx7jomLeo6Ki3qNiYx6jImLeYwFigZ5BokGcAaIBnGJiYpyiIiLc4aJigWK"
    "i3WGiIp2hIiKd4OJiniBiYp6gYmKe3+Jinx+iop9fYqKf3yLioB7ioqCeouKBYN5i4qFeYuKh3eLioh3i4qKdYx2i4qOd4uKj3mLipF7jImSfIyKlH2MipV+"
    "jIsFjIqLipeAjIqZgI2KmYGNipyCjYqcg42KnoSNi5+EjIqhhYyLooWMi6OGjIukhgWmhqaGp4elhaSFooWhhKCDnoOdgpyCmoCZgZd/ln+Vf4uKlH6RfYyL"
    "kHyPe416BYx6inuLiol8h3yGfYR+g3+BgIGBf4KLin2DfYN7hHqFeYV3hnaIdYdziXKKcYoFd4x4jHiMeY15jnmOe4+Ki3uPe5B8kYqLfJF9kXySfZN+k32T"
    "f5R+lX+Vf5ZbdwX4KfpqFfs5+yCtffc59yAFDvmL2cIVloGMi5eCjIqXgpiCjIuYgwWMigaYhIyKmYSMiwWZhIyLmYWMipqFjIubhoyKm4eMipuHjIuch4yL"
    "nIiMipyJjYuciY2KnYqMi56KBYwGngaMBqEGjAagjIyLn4yNi56NjIuejo2LnI4FjYucj42Lm4+NjJqQjYyZkI2Ml5GNjIyLlpGNjIuMlpKMjJWTjIyTlIyM"
    "kpSMjQWQlYyMj5aLjI6Wi42MlouNipiLjImXi42Hl4uMh5aKjYWVi42ElYqMg5WKi4qNBYKUioyAlIqMf5SJjH6TiYx9komMfJKJjHqSiot5kYqMd5GKi3aR"
    "iot2kYmLdZAFiotykHKPdJB2kHaQeJB5kHuRfJF9kYuMf5F/koGTg5OKi4SThJWGlYaViJeJlwWZB5oHjpqOmZCYkpiSl5SWlZaVlIyLlpSYlJmSmZKbkQWb"
    "kJyPnI6ejp6Mn4yWipeLloqWiYyLlYqMi5WIlomViJaHlYiLipWHlIaMi5SGBZSFlYWThIyLk4SUg5ODk4K+m4OViouClIuMgZOLjIGTioyCkoqMgJKKi4GS"
    "iosFgJGKjICQiox/kIqMf4+KjH+OiYx/joqMfo6Ki36Oiot+jYmLfo2Ji36Miot9jAWKBn0GiQZ1BokGdYmJi3aIiIt3h4mKeIeIinmFiYt5hImKe4MFiYp7"
    "g4mKfYKKioqKfoGKin+AioqAf4qKgX+KioN9ioqEfYuKhX2Liod8i4mIfAWKB3sHeweKB459i4qOfoyKkH+LiZKAi4qTgIyKlIGMiQWWgoyKl4KMipiDjYqZ"
    "g42KmoSNi5uEjYqdhYyLnoWMiqCGjIqghoyLooaMi6KGBYyLpIaih6GGoIaehp2FnIWahZmFmISXhJWDjIuUhIuKk4OSg5GCi4qQgo+Bi4oFjoGNf4x/ioGJ"
    "gYiDi4qGhIuKhoSEhISFgoWChoCGf4Z/h32HfIh7iXqIeYp4iQV4i3eKeox6i3uNe417jXyOfI59j32PfZB9kH6RfpF+kn+Sf5J/k3+TgJSAlFx3Bff7+qEV"
    "+zn7IK199zn3IAUO+enb9wIVl4CMi5iAmIGMipmBjIuZgouKmoOMipqDjIoFm4SMipuEjIubhI2LnIWMipyGjYudhoyLnoaMi56IjYqeiI2Ln4mMi6CJjIug"
    "igWNBqAGjQanBo0GpY2Ni6WNjYujj42Lo4+NjKGQjYyfkY6MnpKNjAWdk42Mm5SNjJqVjYyZloyMl5aMjZaXjI2UmIyMkpmMjZGai4yPm4yMjZyLjIycBYwH"
    "jAeKnYuMBYmci4yKjIebi42Fm4qMhJqKjIuMgpmKjIGZio1/mImMfpeJjHyXiox6loqMeZYFiYwGd5SKjHaUiYx1kwWKi3OTiYtzkomMcZGJi3CRiotukHCP"
    "cZBzkHSQdpB3kXiRepF8k4qLfZJ+lH+UBYGVgpaKi4SYiouFmYWah5yJnYqfjJ+LjI2ej56RnJKck5uVmYuMlpiXmJmWmZYFm5Sck4uMnZKekZ+QoI+ijqKN"
    "pIybipuLm4maiZqJmoiZh5mHmYeZhpmFmISYhQWLipiEmIOXgpiCl4GXgJaAjIu8nX+Xiot/l4qLfpZ9lYuMfZSKi36Uiox9k4qLBX2Tiot8koqMfJGKjHuQ"
    "iox8kImMe4+Ki3uPiYx7jomLeo6Ki3qNiYx6jImLeYwFigZ5BokGcAaIBnGJiYpyiIiLc4aJigWKi3WGiIp2hIiKd4OJiniBiYp6gYmKe3+Jinx+iop9fYqK"
    "f3yLioB7ioqCeouKBYN5i4qFeYuKh3eLioh3i4qKdYx2i4qOd4uKj3mLipF7jImSfIyKlH2MipV+jIsFjIqLipeAjIqZgI2KmYGNipyCjYqcg42KnoSNi5+E"
    "jIqhhYyLooWMi6OGjIukhgWmhqaGp4elhaSFooWhhKCDnoOdgpyCmoCZgZd/ln+Vf4uKlH6RfYyLkHyPe416BYx6inuLiol8h3yGfYR+g3+BgIGBf4KLin2D"
    "fYN7hHqFeYV3hnaIdYdziXKKcYoFd4x4jHiMeY15jnmOe4+Ki3uPe5B8kYqLfJF9kXySfZN+k32Tf5R+lX+Vf5ZbdwX4ZPnQFa6Z+xv3FoqMiYyIjImMiIx+"
    "i4iKiYqIiomKior7G/sWrn33CvcFBQ75i9nCFZaBjIuXgoyKl4KYgoyLmIMFjIoGmISMipmEjIsFmYSMi5mFjIqahYyLm4aMipuHjIqbh4yLnIeMi5yIjIqc"
    "iY2LnImNip2KjIueigWMBp4GjAahBowGoIyMi5+MjYuejYyLno6Ni5yOBY2LnI+Ni5uPjYyakI2MmZCNjJeRjYyMi5aRjYyLjJaSjIyVk4yMk5SMjJKUjI0F"
    "kJWMjI+Wi4yOlouNjJaLjYqYi4yJl4uNh5eLjIeWio2FlYuNhJWKjIOViouKjQWClIqMgJSKjH+UiYx+k4mMfZKJjHySiYx6koqLeZGKjHeRiot2kYqLdpGJ"
    "i3WQBYqLcpByj3SQdpB2kHiQeZB7kXyRfZGLjH+Rf5KBk4OTiouEk4SVhpWGlYiXiZcFmQeaB46ajpmQmJKYkpeUlpWWlZSMi5aUmJSZkpmSm5EFm5Ccj5yO"
    "no6ejJ+MloqXi5aKlomMi5WKjIuViJaJlYiWh5WIi4qVh5SGjIuUhgWUhZWFk4SMi5OElIOTg5OCvpuDlYqLgpSLjIGTi4yBk4qMgpKKjICSiouBkoqLBYCR"
    "ioyAkIqMf5CKjH+Piox/jomMf46KjH6Oiot+joqLfo2Ji36NiYt+jIqLfYwFigZ9BokGdQaJBnWJiYt2iIiLd4eJiniHiIp5hYmLeYSJinuDBYmKe4OJin2C"
    "ioqKin6Biop/gIqKgH+KioF/ioqDfYqKhH2LioV9i4qHfIuJiHwFigd7B3sHigeOfYuKjn6MipB/i4mSgIuKk4CMipSBjIkFloKMipeCjIqYg42KmYONipqE"
    "jYubhI2KnYWMi56FjIqghoyKoIaMi6KGjIuihgWMi6SGooehhqCGnoadhZyFmoWZhZiEl4SVg4yLlISLipODkoORgouKkIKPgYuKBY6BjX+Mf4qBiYGIg4uK"
    "hoSLioaEhISEhYKFgoaAhn+Gf4d9h3yIe4l6iHmKeIkFeIt3inqMeot7jXuNe418jnyOfY99j32QfZB+kX6RfpJ/kn+Sf5N/k4CUgJRcdwX4N/oHFa6Z+xv3"
    "FomMiYyJjImMiIx9i4mKiIqJiomKior7G/sWrn33CfcFBQ756dv3AhWXgIyLmICYgYyKmYGMi5mCi4qag4yKBZqDjIqbhIyKm4SMi5uEjYuchYyKnIaNi52G"
    "jIuehoyLnoiNip6IjYufiYyLoIkFjAaVBol/h36HgIaBhYKFgoSDhIODg4SDg4KDg4SCi4oFhIKFgYqKhoGLioeAioqIf4uKiX2LioqDi4qMg4uKjYOLio2E"
    "jIqOhYyKj4WMigWQhoyKkYaMipKHjYqRiI6KkomOipOKjoqTiqWLlY2Ni5SNjYuVjo2MlY6Ni5WQBXSfgYiDiIKJhImEin6LhoyGi4eNh42HjYeOiI+Ij4iQ"
    "iZGKkYqSi5ONmI6Xj5UFkJWRlJGTkZOSk5OTk5OSk4yMkpOSlIyMkpSLjJGVjIyQlYuMkJeLjI+Yi4yOmQWhBo0GpY2Ni6WNjYujj42Lo4+NjKGQjYyfkY6M"
    "npKNjAWdk42Mm5SNjJqVjYyZloyMl5aMjZaXjI2UmIyMkpmMjZGai4yPm4yMjZyLjIycBYwHjAeKnYuMBYmci4yKjIebi42Fm4qMhJqKjIuMgpmKjIGZio1/"
    "mImMfpeJjHyXiox6loqMeZYFiYwGd5SKjHaUiYx1kwWKi3OTiYtzkomMcZGJi3CRiotukHCPcZBzkHSQdpB3kXiRepF8k4qLfZJ+lH+UBYGVgpaKi4SYiouF"
    "mYWah5yJnYqfjJ+LjI2ej56RnJKck5uVmYuMlpiXmJmWmZYFm5Sck4uMnZKekZ+QoI+ijqKNpIybipuLm4maiZqJmoiZh5mHmYeZhpmFmISYhQWLipiEmIOX"
    "gpiCl4GXgJaAjIu8nX+Xiot/l4qLfpZ9lYuMfZSKi36Uiox9k4qLBX2Tiot8koqMfJGKjHuQiox8kImMe4+Ki3uPiYx7jomLeo6Ki3qNiYx6jImLeYwFigZ5"
    "BokGcAaIBnGJiYpyiIiLc4aJigWKi3WGiIp2hIiKd4OJiniBiYp6gYmKe3+Jinx+iop9fYqKf3yLioB7ioqCeouKBYN5i4qFeYuKh3eLioh3i4qKdYx2i4qO"
    "d4uKj3mLipF7jImSfIyKlH2MipV+jIsFjIqLipeAjIqZgI2KmYGNipyCjYqcg42KnoSNi5+EjIqhhYyLooWMi6OGjIukhgWmhqaGp4elhaSFooWhhKCDnoOd"
    "gpyCmoCZgZd/ln+Vf4uKlH6RfYyLkHyPe416BYx6inuLiol8h3yGfYR+g3+BgIGBf4KLin2DfYN7hHqFeYV3hnaIdYdziXKKcYoFd4x4jHiMeY15jnmOe4+K"
    "i3uPe5B8kYqLfJF9kXySfZN+k32Tf5R+lX+Vf5ZbdwUO+YvZwhWWgYyLl4KMipeCmIKMi5iDBYyKBpiEjIqZhIyLmYSMiwWZhYyKmoWMi5uGjIqbh4yKm4eM"
    "i5yHjIuciIyKnImNi5yJjYqNi4iEhYKEgoWDBYODhIODg4SCiouEg4SCioqFgoqLhYGLioaBi4qGgIuKiH+Lioh9i4GMg4uKjYMFi4qNhIyKjoWMio+FjIqP"
    "ho2KkIaNipGHjoqRiI6KkomOipOKjoqTiqWLlI2NiwWVjY2LlY6NjJWOjIuWkHSfgYiCiIOJg4mEin+LhoyGi4eNh42HjYeOiI+Hj4mQBYmRiZGLmo2YjpeP"
    "lZCVkJSRk5KTkpOTk5KTk5OLjJOTkpSMjJGUjIyRlYuMkZUFjZkHjAahBowGoIyMi5+MjYuejYyLno6Ni5yOBY2LnI+Ni5uPjYyakI2MmZCNjJeRjYyMi5aR"
    "jYyLjJaSjIyVk4yMk5SMjJKUjI0FkJWMjI+Wi4yOlouNjJaLjYqYi4yJl4uNh5eLjIeWio2FlYuNhJWKjIOViouKjQWClIqMgJSKjH+UiYx+k4mMfZKJjHyS"
    "iYx6koqLeZGKjHeRiot2kYqLdpGJi3WQBYqLcpByj3SQdpB2kHiQeZB7kXyRfZGLjH+Rf5KBk4OTiouEk4SVhpWGlYiXiZcFmQeaB46ajpmQmJKYkpeUlpWW"
    "lZSMi5aUmJSZkpmSm5EFm5Ccj5yOno6ejJ+MloqXi5aKlomMi5WKjIuViJaJlYiWh5WIi4qVh5SGjIuUhgWUhZWFk4SMi5OElIOTg5OCvpuDlYqLgpSLjIGT"
    "i4yBk4qMgpKKjICSiouBkoqLBYCRioyAkIqMf5CKjH+Piox/jomMf46KjH6Oiot+joqLfo2Ji36NiYt+jIqLfYwFigZ9BokGdQaJBnWJiYt2iIiLd4eJiniH"
    "iIp5hYmLeYSJinuDBYmKe4OJin2CioqKin6Biop/gIqKgH+KioF/ioqDfYqKhH2LioV9i4qHfIuJiHwFigd7B3sHigeOfYuKjn6MipB/i4mSgIuKk4CMipSB"
    "jImWgoyKl4KMipiDBY2KmYONipqEjYubhI2KnYWMi56FjIqghoyKoIaMi6KGjIuihoyLpIaih6GGoIYFnoadhZyFmoWZhZiEl4SVg4yLlISLipODkoORgouK"
    "kIKPgYuKjoGNf4x/ioGJgQWIg4uKhoSLioaEhISEhYKFgoaAhn+Gf4d9h3yIe4l6iHmKeIl4i3eKh4uMkYuMBYyaYY2Ke4qFh4x7jXuNfI58jn2PfY99kH2Q"
    "fpF+kX6Sf5J/kn+Tf5OAlICUXHcFDvnp2/cCFZeAjIuYgJiBjIqZgYyLmYKLipqDjIqag4yKBZuEjIqbhIyLm4SNi5yFjIqcho2LnYaMi56GjIueiI2KnoiN"
    "i5+JjIugiYyLoIoFjQagBo0GpwaNBqWNjYuljY2Lo4+Ni6OPjYyhkI2Mn5GOjJ6SjYwFnZONjJuUjYyalY2MmZaMjJeWjI2Wl4yNlJiMjJKZjI2RmouMj5uM"
    "jI2ci4yMnAWMB4wHip2LjAWJnIuMioyHm4uNhZuKjISaioyLjIKZioyBmYqNf5iJjH6XiYx8l4qMepaKjHmWBYmMBneUiox2lImMdZMFiotzk4mLc5KJjHGR"
    "iYtwkYqLbpBwj3GQc5B0kHaQd5F4kXqRfJOKi32SfpR/lAWBlYKWiouEmIqLhZmFmoeciZ2Kn4yfi4yNno+ekZySnJOblZmLjJaYl5iZlpmWBZuUnJOLjJ2S"
    "npGfkKCPoo6ijaSMm4qbi5uJmomaiZqImYeZh5mHmYaZhZiEmIUFi4qYhJiDl4KYgpeBl4CWgIyLvJ1/l4qLf5eKi36WfZWLjH2Uiot+lIqMfZOKiwV9k4qL"
    "fJKKjHyRiox7kIqMfJCJjHuPiot7j4mMe46Ji3qOiot6jYmMeoyJi3mMBYoGeQaJBnAGiAZxiYmKcoiIi3OGiYoFiot1hoiKdoSIineDiYp4gYmKeoGJint/"
    "iYp8foqKfX2Kin98i4qAe4qKgnqLigWDeYuKhXmLiod3i4qId4uKinWMdouKjneLio95i4qRe4yJknyMipR9jIqVfoyLBYyKi4qXgIyKmYCNipmBjYqcgo2K"
    "nIONip6EjYufhIyKoYWMi6KFjIujhoyLpIYFpoamhqeHpYWkhaKFoYSgg56DnYKcgpqAmYGXf5Z/lX+LipR+kX2Mi5B8j3uNegWMeop7i4qJfId8hn2EfoN/"
    "gYCBgX+Ci4p9g32De4R6hXmFd4Z2iHWHc4lyinGKBXeMeIx4jHmNeY55jnuPiot7j3uQfJGKi3yRfZF8kn2TfpN9k3+UfpV/lX+WW3cF9975zxWNio6KjYqO"
    "ipiLjoyNjI6MjYyMjPcb9xZomfsJ+wX7CvcFaH33G/sWBQ75i9nCFZaBjIuXgoyKl4KYgoyLmIMFjIoGmISMipmEjIsFmYSMi5mFjIqahYyLm4aMipuHjIqb"
    "h4yLnIeMi5yIjIqciY2LnImNip2KjIueigWMBp4GjAahBowGoIyMi5+MjYuejYyLno6Ni5yOBY2LnI+Ni5uPjYyakI2MmZCNjJeRjYyMi5aRjYyLjJaSjIyV"
    "k4yMk5SMjJKUjI0FkJWMjI+Wi4yOlouNjJaLjYqYi4yJl4uNh5eLjIeWio2FlYuNhJWKjIOViouKjQWClIqMgJSKjH+UiYx+k4mMfZKJjHySiYx6koqLeZGK"
    "jHeRiot2kYqLdpGJi3WQBYqLcpByj3SQdpB2kHiQeZB7kXyRfZGLjH+Rf5KBk4OTiouEk4SVhpWGlYiXiZcFmQeaB46ajpmQmJKYkpeUlpWWlZSMi5aUmJSZ"
    "kpmSm5EFm5Ccj5yOno6ejJ+MloqXi5aKlomMi5WKjIuViJaJlYiWh5WIi4qVh5SGjIuUhgWUhZWFk4SMi5OElIOTg5OCvpuDlYqLgpSLjIGTi4yBk4qMgpKK"
    "jICSiouBkoqLBYCRioyAkIqMf5CKjH+Piox/jomMf46KjH6Oiot+joqLfo2Ji36NiYt+jIqLfYwFigZ9BokGdQaJBnWJiYt2iIiLd4eJiniHiIp5hYmLeYSJ"
    "inuDBYmKe4OJin2CioqKin6Biop/gIqKgH+KioF/ioqDfYqKhH2LioV9i4qHfIuJiHwFigd7B3sHigeOfYuKjn6MipB/i4mSgIuKk4CMipSBjIkFloKMipeC"
    "jIqYg42KmYONipqEjYubhI2KnYWMi56FjIqghoyKoIaMi6KGjIuihgWMi6SGooehhqCGnoadhZyFmoWZhZiEl4SVg4yLlISLipODkoORgouKkIKPgYuKBY6B"
    "jX+Mf4qBiYGIg4uKhoSLioaEhISEhYKFgoaAhn+Gf4d9h3yIe4l6iHmKeIkFeIt3inqMeot7jXuNe418jnyOfY99j32QfZB+kX6RfpJ/kn+Sf5N/k4CUgJRc"
    "dwX3sfoGFY2KjYqOio2KmYuOjI2MjYyNjI2M9xv3FmiZ+wr7BfsJ9wVoffcb+xYFDvnI0fnfFffKioz93pMGinyIfoh+hoCGgYaChIKEg4SDhIODg4OChIOK"
    "i4SCi4qFgoqLBYWBi4qGgYqKh4CLiod/i4qJfYuBjIOLioyDjIqNhIuKj4WLio+FjIqQho2KkIYFjYqRh42KkoiOipKJjoqSio+Kk4qli5SNjYuVjY2LlY6M"
    "jJaOjIuWkHOfgoiCiAWDiYOJhIp/i4aMhouHjYaNh42IjoePiI+JkImRiZGLmo2YjpePlZCVkJSRk5KTBZKTkpOTk5OTi4ySk4yLkpSLjJKUjIyRlYuMkZWL"
    "jJCXi4yPmIuMjZmLjI2adowFqPnejIz3yq39PGkGDviGx/j9FWn3EYqM/D8HeweNfIuKjX6Lio5+jIqPf4uKkICLipGAjIqRgoyJBZOCjIqTg4yKlIOMi4yK"
    "lYWNiZaGjYmXh42KmIaNi5iHjIuNipmJjoqZiY6LmooFjgabBowGwq1VBnyMBX6Mf4yAjYGOgo6Ki4OPiouDkIOQhJGEkoWShZSGlIaViJaLjIiWi4yJl4uM"
    "iZgFmgf4P4yM9zmt+zmMivc2Ufs2ior7EQf3Of0ZFYh+hoCGgYaChIKEg4SDhIODg4OChIOKi4SCi4qFgoqLhYEFi4qGgYqKh4CLiod/i4qJfYuBjIOLioyD"
    "jIqNhIuKj4WLio+FjIqQho2KkIaNigWRh42KkoiOipKJjoqSio+Kk4qli5SNjYuVjY2LlY6MjJaOjIuWkHOfgoiCiIOJBYOJhIp/i4aMhouHjYaNh42IjoeP"
    "iI+JkImRiZGLmo2YjpePlZCVkJSRk5KTkpMFkpOTk5OTi4ySk4yLkpSLjJKUjIyRlYuMkZWLjJCXi4yPmIuMjZmLjI2aYY2KewUO+cjR+d8V98qKjP3exfne"
    "jIz3yq39PGkG99jpFY2KjYqNio6KmYuOjI2MjYyNjI2M9xv3FmeZ+wn7BfsJ9wVnffcb+xYFDviGx/j9FWn3EYqM/D8HeweNfIuKjX6Lio5+jIqPf4uKkICL"
    "ipGAjIqRgoyJBZOCjIqTg4yKlIOMi4yKlYWNiZaGjYmXh42KmIaNi5iHjIuNipmJjoqZiY6LmooFjgabBowGwq1VBnyMBX6Mf4yAjYGOgo6Ki4OPiouDkIOQ"
    "hJGEkoWShZSGlIaViJaLjIiWi4yJl4uMiZgFmgf4P4yM9zmt+zmMivc2Ufs2ior7EQf3QvfUFY2KjYqNio6KmYuOjI2MjYyNjI2M9xv3FmeZ+wn7BfsJ9wVn"
    "ffcb+xYFDvng3fnfFffKioz8HoqK+3pp93qKjPwwxfgwjIz3eq37eoyK+B6MjPfKrf08aQYO+MPd+A4V9yCKjPtyBnsHjXyLio1+i4qOfoyKj3+LipCAi4qR"
    "gIyKkYKMiQWTgoyKk4OMipSDjIuMipWFjYmWho2Jl4eNipiGjYuYh4yLjYqZiY6KmYmOi5qKBY4GmwaMBsKtVQZ8jAV+jH+MgI2BjoKOiouDj4qLg5CDkISR"
    "hJKFkoWUhpSGlYiWi4yIlouMiZeLjImYBZoH93KMjPcMrfsMjIr3PYyM9zmt+zmMivc2Ufs2ior7EWn3EYqM+z2KivsgaQcO+d7d99kVjGuNbYuKj2+MipBw"
    "k3KLipRzi4qWdYuKl3cFjIqYeIyKmnqNipt7jYqdfYyJn3+NiqCAjoqhgo6Ko4SOiqSFjoulho+LpomOigWpBo0GqQaOjKaNj4ulkI6LpJGOjKOSjoyhlI6M"
    "oJaNjAWfl4yNnZmNjJubjYyanIyMmJ6MjJefi4yWoYuMlKOLjJOkkKaMjI+ni4yNqYyrBfirUfyrB4psiW4Fh2+FcYRzg3SBdn94f3qLin18fHx7f4qLeoB5"
    "gXeEdoR1hnSHiotziYqLcYpxjAWKi3ONiot0j3WQdpJ3knmVepaKi3uXfJp9mouMf5x/noGgg6KEo4Wlh6eJqIqqBfirUfyrB/d3+T0VkpKSkJGQkY6RjpCM"
    "j4ySi4+KkIqRiAWRiJGGkoaShJKDmnmTgpODjIuThIyLk4WMipOHjYqTiI2Kk4iPipKKjYqki42MBZKMj4yTjo2Mk46NjJOPjIyTkYyLk5KMi5OTk5STlWWV"
    "hIOEg4SEhIaFhoWIhYgFhoqHioSLh4yGjIWOhY6FkISQhJKEk3ydg5SDk4qLg5KKi4ORioyDj4mMg46JjAWDjoeMhIyJjHKLiYqEioeKg4iJioOIiYqDh4qK"
    "g4WKi4OEiouDg4OCg4GxgZKTBQ75sN/3thWMbo1wi4qPcZFzi4qSdIuKk3aMipV4i4qWeQWMiouKmHuMipl7jIqafY2Km3+Nip2AjoqegY2KoIOOiqCFjoqi"
    "ho6Lo4eOi6WIBY0GpgaNBqYGjQaljo6Lo4+Oi6KQjoygkY6MoJONjJ6VjoydlgWNjJuXjYyamYyMmZuMjJibi4yMjJadi4yVnoyMk6CLjJKii4yRo4+li4yN"
    "poyoBffKUfvKB4pviXGHcoZ0hHWDd4J4gXp/fH59fX+Lin2Biot8gXqDeYR4hXeHdoh1iQVzinOMdY12jnePeJF5knqTfJWKi32Vi4x9l36Zf5qBnIKeg5+E"
    "oYaih6SJpYqnBffKUfvKB/de+WAVkpKSkJGQkY6RjpCMj4ySi4+KkIqRiAWRiJGGkoaShJKDmnmTgpODjIuThIyLk4WMipOHjYqTiI2Kk4iPipKKjYqki42M"
    "BZKMj4yTjo2Mk46NjJOPjIyTkYyLk5KMi5OTk5STlWWVhIOEg4SEhIaFhoWIhYgFhoqHioSLh4yGjIWOhY6FkISQhJKEk3ydg5SDk4qLg5KKi4ORioyDj4mM"
    "g46JjAWDjoeMhIyJjHKLiYqEioeKg4iJioOIiYqDh4qKg4WKi4OEiouDg4OCg4GxgZKTBQ753t332RWMa41ti4qPb4yKkHCTcouKlHOLipZ1i4qXdwWMiph4"
    "jIqaeo2Km3uNip19jImff42KoICOiqGCjoqjhI6KpIWOi6WGj4umiY6KBakGjQapBo6Mpo2Pi6WQjoukkY6Mo5KOjKGUjoyglo2MBZ+XjI2dmY2Mm5uNjJqc"
    "jIyYnoyMl5+LjJahi4yUo4uMk6SQpoyMj6eLjI2pjKsF+KtR/KsHimyJbgWHb4VxhHODdIF2f3h/eouKfXx8fHt/iot6gHmBd4R2hHWGdIeKi3OJiotxinGM"
    "BYqLc42Ki3SPdZB2kneSeZV6loqLe5d8mn2ai4x/nH+egaCDooSjhaWHp4moiqoF+KtR/KsH+H35SRX7wHH3wAYO+bDf97YVjG6NcIuKj3GRc4uKknSLipN2"
    "jIqVeIuKlnkFjIqLiph7jIqZe4yKmn2Nipt/jYqdgI6KnoGNiqCDjoqghY6KooaOi6OHjouliAWNBqYGjQamBo0GpY6Oi6OPjouikI6MoJGOjKCTjYyelY6M"
    "nZYFjYybl42MmpmMjJmbjIyYm4uMjIyWnYuMlZ6MjJOgi4ySoouMkaOPpYuMjaaMqAX3ylH7ygeKb4lxh3KGdIR1g3eCeIF6f3x+fX1/i4p9gYqLfIF6g3mE"
    "eIV3h3aIdYkFc4pzjHWNdo53j3iReZJ6k3yViot9lYuMfZd+mX+agZyCnoOfhKGGooekiaWKpwX3ylH7ygf4ZPlsFfvAcffABg753t332RWMa41ti4qPb4yK"
    "kHCTcouKlHOLipZ1i4qXdwWMiph4jIqaeo2Km3uNip19jImff42KoICOiqGCjoqjhI6KpIWOi6WGj4umiY6KBakGjQapBo6Mpo2Pi6WQjoukkY6Mo5KOjKGU"
    "joyglo2MBZ+XjI2dmY2Mm5uNjJqcjIyYnoyMl5+LjJahi4yUo4uMk6SQpoyMj6eLjI2pjKsF+KtR/KsHimyJbgWHb4VxhHODdIF2f3h/eouKfXx8fHt/iot6"
    "gHmBd4R2hHWGdIeKi3OJiotxinGMBYqLc42Ki3SPdZB2kneSeZV6loqLe5d8mn2ai4x/nH+egaCDooSjhaWHp4moiqoF+KtR/KsH90T5WRWUfI2Kln6Niph/"
    "jYqbgo2JnISNip6FjYqeho6LBZ+Ijoueio+LnoyOi5+OjouekI2MnpGNjJySjY2blI2MmJeNjJaYjYyUmoyNkZsFY5GEe4N+gYB/gX2DfYR7hnuHe4l6inqM"
    "e417j3uQfZJ9k3+VgZaDmISbY4WRewUO+bDf97YVjG6NcIuKj3GRc4uKknSLipN2jIqVeIuKlnkFjIqLiph7jIqZe4yKmn2Nipt/jYqdgI6KnoGNiqCDjoqg"
    "hY6KooaOi6OHjouliAWNBqYGjQamBo0GpY6Oi6OPjouikI6MoJGOjKCTjYyelY6MnZYFjYybl42MmpmMjJmbjIyYm4uMjIyWnYuMlZ6MjJOgi4ySoouMkaOP"
    "pYuMjaaMqAX3ylH7ygeKb4lxh3KGdIR1g3eCeIF6f3x+fX1/i4p9gYqLfIF6g3mEeIV3h3aIdYkFc4pzjHWNdo53j3iReZJ6k3yViot9lYuMfZd+mX+agZyC"
    "noOfhKGGooekiaWKpwX3ylH7ygf3K/l8FZR8jYqWfo2KmH+NipuCjYmchI2KnoWNip6GjosFn4iOi56Kj4uejI6Ln46Oi56QjYyekY2MnJKNjZuUjYyYl42M"
    "lpiNjJSajI2RmwVjkYR7g36BgH+BfYN9hHuGe4d7iXqKeox7jXuPe5B9kn2Tf5WBloOYhJtjhZF7BQ753t332RWMa41ti4qPb4yKkHCTcouKlHOLipZ1i4qX"
    "dwWMiph4jIqaeo2Km3uNip19jImff42KoICOiqGCjoqjhI6KpIWOi6WGj4umiY6KBakGjQapBo6Mpo2Pi6WQjoukkY6Mo5KOjKGUjoyglo2MBZ+XjI2dmY2M"
    "m5uNjJqcjIyYnoyMl5+LjJahi4yUo4uMk6SQpoyMj6eLjI2pjKsF+KtR/KsHimyJbgWHb4VxhHODdIF2f3h/eouKfXx8fHt/iot6gHmBd4R2hHWGdIeKi3OJ"
    "iotxinGMBYqLc42Ki3SPdZB2kneSeZV6loqLe5d8mn2ai4x/nH+egaCDooSjhaWHp4moiqoF+KtR/KsH9335NxWDB4yEjYSLio2Fi4qNhYyLjYWMio6Fj4WM"
    "io+GkIaLipCHjIqQh4yKBZGHjIuRh4yLkYiNipGJjYqSiY2LkYmOi5KKoYuSjI6LkY2Ni5KNjYyRjY2MkY4FjIuRj4yLkY+MjJCPjIyQj4uMkJCPkIyMj5GO"
    "kYyMjZGMi42Ri4yNkYuMjZKMkgWTB5MHipKJkouMiZGLjImRiouJkYqMiJGHkYqMh5CGkIuMho+KjIaPiowFhY+Ki4WPiouFjomMhY2JjISNiYuFjYiLhIx1"
    "i4SKiIuFiYmLhImJioWJiYqFiAWKi4WHiouFh4qKhoeKioaHi4qGhoeGioqHhYiFioqJhYqLiYWLiomFi4qJhIqEBYMHtZEVjJKMkQWNkI2RjZCOkI+QjpCP"
    "j4+Oj4+PjpCNj42PjZCMj4ybi4+KkIqPiY+JkImPiI+HBY+Ij4eOho+GjoaNho2FjYaMhYyEi3+KhIqFiYaJhYmGiIaHhoiGh4eHiIeHh4gFhomHiYeJhoqH"
    "inuLh4yGjIeNh42GjYeOh4+HjoePiJCHkIiQiZCJkYmQipGKkgWRBw75sN/3thWMbo1wi4qPcZFzi4qSdIuKk3aMipV4i4qWeQWMiouKmHuMipl7jIqafY2K"
    "m3+Nip2AjoqegY2KoIOOiqCFjoqiho6Lo4eOi6WIBY0GpgaNBqYGjQaljo6Lo4+Oi6KQjoygkY6MoJONjJ6VjoydlgWNjJuXjYyamYyMmZuMjJibi4yMjJad"
    "i4yVnoyMk6CLjJKii4yRo4+li4yNpoyoBffKUfvKB4pviXGHcoZ0hHWDd4J4gXp/fH59fX+Lin2Biot8gXqDeYR4hXeHdoh1iQVzinOMdY12jnePeJF5knqT"
    "fJWKi32Vi4x9l36Zf5qBnIKeg5+EoYaih6SJpYqnBffKUfvKB/dk+VoVgweMhI2Ei4qNhYuKjYWMi42FjIqOhY+FjIqPhpCGi4qQh4yKkIeMigWRh4yLkYeM"
    "i5GIjYqRiY2KkomNi5GJjouSiqGLkoyOi5GNjYuSjY2MkY2NjJGOBYyLkY+Mi5GPjIyQj4yMkI+LjJCQj5CMjI+RjpGMjI2RjIuNkYuMjZGLjI2SjJIFkweT"
    "B4qSiZKLjImRi4yJkYqLiZGKjIiRh5GKjIeQhpCLjIaPioyGj4qMBYWPiouFj4qLhY6JjIWNiYyEjYmLhY2Ii4SMdYuEioiLhYmJi4SJiYqFiYmKhYgFiouF"
    "h4qLhYeKioaHioqGh4uKhoaHhoqKh4WIhYqKiYWKi4mFi4qJhYuKiYSKhAWDB7WRFYySjJEFjZCNkY2QjpCPkI6Qj4+Pjo+Pj46QjY+Nj42QjI+Mm4uPipCK"
    "j4mPiZCJj4iPhwWPiI+HjoaPho6GjYaNhY2GjIWMhIt/ioSKhYmGiYWJhoiGh4aIhoeHh4iHh4eIBYaJh4mHiYaKh4p7i4eMhoyHjYeNho2HjoePh46Hj4iQ"
    "h5CIkImQiZGJkIqRipIFkQcO+d7d99kVjGuNbYuKj2+MipBwk3KLipRzi4qWdYuKl3cFjIqYeIyKmnqNipt7jYqdfYyJn3+NiqCAjoqhgo6Ko4SOiqSFjoul"
    "ho+LpomOigWpBo0GqQaOjKaNj4ulkI6LpJGOjKOSjoyhlI6MoJaNjAWfl4yNnZmNjJubjYyanIyMmJ6MjJefi4yWoYuMlKOLjJOkkKaMjI+ni4yNqYyrBfir"
    "UfyrB4psiW4Fh2+FcYRzg3SBdn94f3qLin18fHx7f4qLeoB5gXeEdoR1hnSHiotziYqLcYpxjAWKi3ONiot0j3WQdpJ3knmVepaKi3uXfJp9mouMf5x/noGg"
    "g6KEo4Wlh6eJqIqqBfirUfyrB/gT+PsV6vcgZZUs+yAF+y8WsYHq9yBllQUO+bDf97YVjG6NcIuKj3GRc4uKknSLipN2jIqVeIuKlnkFjIqLiph7jIqZe4yK"
    "mn2Nipt/jYqdgI6KnoGNiqCDjoqghY6KooaOi6OHjouliAWNBqYGjQamBo0GpY6Oi6OPjouikI6MoJGOjKCTjYyelY6MnZYFjYybl42MmpmMjJmbjIyYm4uM"
    "jIyWnYuMlZ6MjJOgi4ySoouMkaOPpYuMjaaMqAX3ylH7ygeKb4lxh3KGdIR1g3eCeIF6f3x+fX1/i4p9gYqLfIF6g3mEeIV3h3aIdYkFc4pzjHWNdo53j3iR"
    "eZJ6k3yViot9lYuMfZd+mX+agZyCnoOfhKGGooekiaWKpwX3ylH7ygf3+vkeFer3IGWVLPsgBfsvFrGB6vcgZZUFDvne3ffZFYxrjW2Lio9vjIqQcJNyi4qU"
    "c4uKlnWLipd3BYyKmHiMipp6jYqbe42KnX2MiZ9/jYqggI6KoYKOiqOEjoqkhY6LpYaPi6aJjooFqQaNBqEGh4WLioV/iouGfoZ/hn+Hf4h/iICLioqAiouK"
    "gAWAB4AHjIGMgQWLio2Ci4qOgouKj4KLipCCjIqQg4yKkYSMipKEjIqSho2Kk4aNipOIjoqTiI6LBZOJjoqUipqLlIyOi5SMjYyUjY2LlI6NjJOOjYyTj4yM"
    "k4+MjJOQjIySkWmZhYYFhIeFh4SIhIiFiYSJhYp/i4WMhoyFjYaNhY6FkIaQhZGHkoaTiJSIk4qUiZWKlAWLlYyVjJWNlo2VjpaPl4+WkJeRl5GXkpiOkKGP"
    "joukkY6Mo5KOjKGUjoyglo2MBZ+XjI2dmY2Mm5uNjJqcjIyYnoyMl5+LjJahi4yUo4uMk6SQpoyMj6eLjI2pjKsF+KtR/KsHimyJbgWHb4VxhHODdIF2f3h/"
    "eouKfXx8fHt/iot6gHmBd4R2hHWGc5GEgXqJiotxinGMBYqLc42Ki3SPdZB2kneSeZV6loqLe5d8mn2ai4x/nH+egaCDooSjhaWHp4moiqoF+KtR/KsHDvmw"
    "3/e2FYxujXCLio9xkXOLipJ0i4qTdoyKlXiLipZ5BYyKi4qYe4yKmXuMipp9jYqbf42KnYCOip6BjYqgg46KoIWOiqKGjoujh46LpYgFjQamBo0GoQaHhYuK"
    "hX+Ki4Z+hn+Gf4d/iH+IgIuKioCKi4qABYAHgAeMgYyBi4oFjYKLio6Ci4qPgouKkIKMipCDjIqRhIyKkoSMipKGjYqTho2Kk4iOipOIjouTiQWOipSKmouU"
    "jI6LlIyNjJSNjYuUjo2Mk46NjJOPjIyTj4yMk5CMjJKRaZmFhoSHBYWHhIiEiIWJhImFin+LhYyGjIWNho2FjoWQhpCFkYeShpOIlIiTipSJlYqUi5UFjJWM"
    "lY2WjZWOlo+Xj5aQl5GXkZeSmI6QmY6Oi6KQjoygkY6MoJONjJ6VjoydlgWNjJuXjYyamYyMmZuMjJibi4yMjJadi4yVnoyMk6CLjJKii4yRo4+li4yNpoyo"
    "BffKUfvKB4pviXGHcoZ0hHWDd4J4gXp/fH59fX+Lin2Biot8gXqDeYR4hX2IdZGEgXeJBXOKc4x1jXaOd494kXmSepN8lYqLfZWLjH2Xfpl/moGcgp6Dn4Sh"
    "hqKHpImliqcF98pR+8oHDhwEh8357RX3Zv3vi4qMiYyJjYmNiY2KjIuNiY+KjoqPigWOBo8GjwaPBo6MjoyMi46MjY2OjY2Mi4yNjYyN94r42QWMBowG94r8"
    "2YyJjImLio2KjomOiY6KjoqPigWSBo8GjwaOjI+MjoyOjY2Mjo2MjY2Ni473ZvnvUpGLivtQ/ZYFiooGioz7gvjGio2LjImNiYyIjYmNiouIjIiMh4wFhAaH"
    "BocGiIqKi4iKiIqIiYmJiouJioqJi4qKifuC/MYFioqKjIoG+1D5louMUoUF+QvcFa6Z+xv3FoqMiYyJjIiMiYx9i4iKiYqJiomKiYr7G/sWrn33CvcFBQ4c"
    "BHPN+OgV92v87IyJjYmMiYyLjYmOio6JjoqOigWPBo8GjwaPBo6Mj4yOjI6Mjo2NjY2NjI33e/hMBYyMjIoG93z8TIyJjYmNiY6JjoqOio6KjIuOigWPBo8G"
    "kgaPjI6Mjo2OjI2NjY2NjYyN92v47FKT+1P8qQWKigeKjAb7dvhBio2JjYiNiY2IjIeNiIuHjAWHBocGiAaHioeLiImIioiJiYmJiYqJ+3b8QQWKigeKjAb7"
    "U/ipUoMF+QH36hWumfsb9xaJjImMiYyJjIiMfYuJioiKiYqJioqK+xv7Fq599wn3BQUO+f3R+ekV9+X8SAWKB/w0xfg0B4yM9+X4SFaZ+838KAWKigeKjAb7"
    "zPgoVn0F+HjgFa6Z+xv3FomMiYyJjImMiIx9i4mKiIqJiomKior7G/sWrn33CfcFBQ75w9H45xX3xfzsk4wFjAaMigZP+wyFgIWABYWAhIKLioSChIOEg4uK"
    "g4SEhIOFg4aLioOHgoaCiIOIgYiCiYGKgIp/ioqLjmkFmQaOBpiMjoyYjI2MmI2NjJeOjYyXj42Mlo+NjJWQjYyUkY2MBZSSjIuUkoyMlJOMjJOTjIyTk4uM"
    "k5SLjJKVjIuSlpGWjIuRl5GXjIv4APluVJX7q/zCBYoGigb7sPjCVIEF+Fv36xWumfsb9xaKjImMiYyIjImMfYuIiomKiYqJiomK+xv7Fq599wr3BQUO+f34"
    "K/g1FYoH/DTF+DQHjIz35fhIVpn7zfwoBYqKB4qMBvvM+ChWfQX4P/cfFYyIjIiMiY2JjIiNiY2JjYmNio2JjYqOio2KjoqOipaLjoyNjI6MjYyOjI2NjYwF"
    "jY2NjY2NjI6MjY2NjI6LjoyNi5eKjYuOio6JjYqNio6JjYmNiY2JjImNiIyJjAWIjImMiIyAi4iKiIqJioiKiYqJiYmKiYmJiYmJioiJiYqJioiKiIuJioiL"
    "hYyIBftSFokHjIiMiIyJjYmMiI2JjYmNiY2KjYmNio6KjYqOio6KlouOjI2MjoyNjI6MBY2NjYyNjY2NjY2MjoyNjY2MjouOjI2Ll4qNi46KjomNio2KjomN"
    "iY2JjYmMiY0FiIyJjIiMiYyIjICLiIqIiomKiIqJiomJiYqJiYmJiYmKiImJiomKiIqIi4mKiAWIB4gHDvn39wj53xX47gaMiouK/Rj91omJiokFiQeIB4yJ"
    "jImNiY2JjYmOiY6Kj4qOigWPBo8G+RSt/OQGioyLjPkY+daNjYyNBY0HjgeKjYqNiY2JjYmNiI2IjIeMiIwFhwaHBv0eaQb4C/eNFfs5+yCuffc59yAFDvmT"
    "9yv4/RVp+GOKjIqKigf8rvzPiYmLioqJiokFiQeJB4yIjImNiY2JjoqNiY+KjoqOigWPBo8G+MSt/JCMioyMjAb4rvjPjY2LjIyNjI0FjQeNB4qOio2JjYmN"
    "iIyJjYeMiIyHjAWIBocG/JcG97b4bxX7Ofsgrn33OfcgBQ759/cI+d8V+O4GjIqLiv0Y/daJiYqJBYkHiAeMiYyJjYmNiY2JjomOio+KjooFjwaPBvkUrfzk"
    "BoqMi4z5GPnWjY2MjQWNB44Hio2KjYmNiY2JjYiNiIyHjIiMBYcGhwb9HmkG96z3LhWIB4yIi4iMiI2JjIiNiY2IjYmNiY2JjYqOiY6KjYqOio6KmIuOjI2M"
    "joyOjI2NjowFjY2NjY2NjY6NjYyOjI2NjouOjI6Ll4qOi46JjoqNio6JjYmOiY2JjYmNiIyJjQWIjIiMiYyIjH6LiIqIiomKiIqIiYmKiYmJiYmJiYiJiYqI"
    "iYmKiIuIioiLiIqIBQ75k/cr+P0VafhjioyKiooH/K78z4mJi4qKiYqJBYkHiQeMiIyJjYmNiY6KjYmPio6KjooFjwaPBvjErfyQjIqMjIwG+K74z42Ni4yM"
    "jYyNBY0HjQeKjoqNiY2JjYiMiY2HjIiMh4wFiAaHBvyXBvdX+BAViAeMiIuIjIiNiYyIjYmNiI2JjYmNiY2KjomOio2KjoqOipiLjoyOjI2MjoyOjY2MBY2N"
    "jY2NjY2OjY2Mjo2NjI6LjoyOi5eKjouOio6JjYqOiY2JjomNiY2JjYmMiI0FiIyJjIiMiIx+i4iKiIqJioiKiImJiomJiYmJiYmIiYmKiImJioiLiIqIi4iK"
    "iAUO+ff3CPnfFfjuBoyKi4r9GP3WiYmKiQWJB4gHjImMiY2JjYmNiY6JjoqPio6KBY8Gjwb5FK385AaKjIuM+Rj51o2NjI0FjQeOB4qNio2JjYmNiY2IjYiM"
    "h4yIjAWHBocG/R5pBvfB6RWNio6KjYqOipiLjoyNjI6MjYyMjPcb9xZomfsK+wX7CfcFaH33G/sWBQ75k/cr+P0VafhjioyKiooH/K78z4mJi4qKiYqJBYkH"
    "iQeMiIyJjYmNiY6KjYmPio6KjooFjwaPBvjErfyQjIqMjIwG+K74z42Ni4yMjYyNBY0HjQeKjoqNiY2JjYiMiY2HjIiMh4wFiAaHBvyXBvds99QVjYqOio2K"
    "joqYi46MjYyOjI2MjIz3G/cWaJn7CvsF+wn3BWh99xv7FgUO+C7dFsX40waMpQWMo46ij6CQn5GekpyTmpSZi4yVl5WWlpWXk4uMl5KYkJiQmY+ajZqNnYyM"
    "i4etBXgGhwZ4iYiKeYiHinqHiIp7hYiKe4SJiXyDBYmJfoKJiX6AioqAfoqKgH2KioJ8ioqDe4qKhHmKioV4i4qGdoZ1i4qIc4pyinAF/NMHDviq7PfHFWn3"
    "6K376AcO+VTd98cVafiwrfywBw75VN33xxVp+LCt/LAHDvpd3ffHFWn5ua39uQcO+l3d98cVafm5rf25Bw73ofcO+e0Vg4CLioSAiouFf4Z/i4qGgIuKiH+L"
    "ioiAi4qKf4uKioAFigd/B41/i4qNgIuKjoGLio+AjIqPgYyKkYKLiZKCjIqSg4yLjIqLipOEjYmUhLmhBYKRhJKFkoWThpOHlIeViJWJlYqVi5aMloyWjZaO"
    "lo+Vi4yQlZCWjIuRlZKVVpkFDveh3fnbFZOFjIuRhIyLkYSRg5CDj4KPgY2BjYGMgYyAioCKgIiAiIAFh4GLioeBhYCFgYOBwH2TloyMkpaRl4yLkJeLjI+W"
    "i4yPl4uMjZaLjI2Xi4yMlgWLjIqXipeLjImWi4yIlYqMh5aLjIaVi4yFlIqNhZSKjIOTioyLjIKSio2Ckl11BQ74c/fg+e0Vg4CLioSAiouFf4Z/i4qGgIuK"
    "iH+LioiAi4qKf4uKioAFigd/B41/i4qNgIuKjoGLio+AjIqPgYyKkYKLiZKCjIqSg4yLjIqLipOEjYmUhLmhBYKRhJKFkoWThpOHlIeViJWJlYqVi5aMloyW"
    "jZaOlo+Vi4yQlZCWjIuRlZKVVpkF+2YWg4CLioSAiouFf4Z/i4qGgIuKiH+LioiAi4qKf4uKioAFigd/B41/i4qNgIuKjoGLio+AjIqPgYyKkYKLiZKCjIqS"
    "g4yLjIqLipOEjYmUhLmhBYKRhJKFkoWThpOHlIeViJWJlYqVi5aMloyWjZaOlo+Vi4yQlZCWjIuRlZKVVpkFDvhz97j52xWThYyLkYSMi5GEkYOQg4+Cj4GN"
    "gY2BjIGMgIqAioCIgIiABYeBi4qHgYWAhYGDgcB9k5aMjJKWkZeMi5CXi4yPlouMj5eLjI2Wi4yNl4uMjJYFi4yKl4qXi4yJlouMiJWKjIeWi4yGlYuMhZSK"
    "jYWUioyDk4qMi4yCkoqNgpJddQX7ZhaThYyLkYSMi5GEkYOQg4+Cj4GNgY2BjIGMgIqAioCIgIiABYeBi4qHgYWAhYGDgcB9k5aMjJKWkZeMi5CXi4yPlouM"
    "j5eLjI2Wi4yNl4uMjJYFi4yKl4qXi4yJlouMiJWKjIeWi4yGlYuMhZSKjYWUioyDk4qMi4yCkoqNgpJddQUO+PDd+N8VafdSioz8vMX4vIyM91Kt+1KMivek"
    "Ufukior7UgcO+PDd958VafdSioz7fMX3fIyM91Kt+1KMivfOjIz3Uq37UoyK94ZR+4aKivtSafdSioz7zoqK+1IHDvem3ffAFYYHjIeMh4yHjIeNh42HjoeN"
    "iI6IjoiPiY6Ij4mPiY+Kj4oFio8HkIoFjwaQBo+MBY+MBo+MjIuPjI6Nj42Pjo6Njo6Ojo6OjY+Nj42PjI+Nj4uPjI8FkAeQB4qPi4+Jj4qPiY+Jj4mPiI6I"
    "joiOiI2HjoeNiI2HjIqLh4wFjIcHh4wFhgaHBoaKBYeKBoeKh4qHiYeJiIiHiYiIiIiJiIiHiYeJh4qHioeKh4qHBYYHDvjY+DyiFYyIjImNiIyJjYiNiY2J"
    "jomNio6JjYqOio6KBY6KjgaOBo4GjoyOBo6MjoyNjI6NjYyOjY2NjY2NjoyNjY6MjYyOjI4FjoyOB44HjgeOio4Hio6KjoqNiY6KjYmOiY2JjYiNiYyIjYmM"
    "iIyIjAWIjIgGiAaIBoiKiAaIioiKiYqIiYmKiImJiYmJiYiKiYmIiomKiIqIBYiKiAeIB4gHiIyIB/s9iBWMiIyJjYiMiY2IjYmNiY6JjYqOiY2KjoqOigWO"
    "io4GjgaOBo6MjgaOjI6MjYyOjY2Mjo2NjY2NjY6MjY2OjI2MjoyOBY6MjgeOB44HjoqOB4qOio6KjYmOio2JjomNiY2IjYmMiI2JjIiMiIwFiIyIBogGiAaI"
    "iogGiIqIiomKiImJioiJiYmJiYmIiomJiIqJioiKiAWIiogHiAeIB4iMiAf7P44VjIgGjIiMiIyJjYiMiY2IjYmNiY6JjYqOiY2KjoqOigWOio4GjgaOBo6M"
    "jgaOjI6MjYyOjY2Mjo2NjY2NjY6MjY2OjI2MjoyOBY6MjgeOB44HjoqOB4qOio6KjYmOio2JjomNiY2IjYmMiI2JjIiMiIwFiIyIBogGiAaIiogGiIqIiomK"
    "iImJioiJiYmJiYmIiomJiIqJioiKiAWIiogHiAeIBw75wPhE2xWBB4oHjYGLio2Ci4qOgouKjoKMi46CjIqQg4uKBZCEjIqRg4yKkYWMipKEjIqShoyKk4WN"
    "ipOHjYqTh46KlIeNipWJjoqUiY+LlIkFjwaVBo8GlQaPBpSNj4uUjY6MlY2NjJSPjoyTj42Mk4+NjAWTkYyMkpCMjJKSjIyRkYyMkZOMjJCSi4yQk4yMjpSM"
    "i46Ui4yOlIuMjZSLjI2VBYwHlQeVB4wHiZWLjImUi4yIlIuMiJSKi4iUioyGk4uMBYaSioyFk4qMhZGKjISSioyEkIqMg5GJjIOPiYyDj4iMgo+JjIGNiIyC"
    "jYeLgo0FhwaBBocGgQaHBoKJh4uCiYiKgYmJioKHiIqDh4mKg4eJigWDhYqKhIaKioSEioqFhYqKhYOKioaEi4qGg4qKiIKKi4iCi4qIgouKiYKLiomBBYoH"
    "gQfLrRWMB46SjpOPkpCSkJGQkZGQkZCRj5GOkY6RjpGMBZGNkYuRjJGKkYuRiZGKkYiRiJGIkYeRhpGGkIWQhZCEj4SOg46Ei4qOg42DjIIFggeCB4qCiYOI"
    "g4uKiISIg4eEhoSGhYaFhYaFhoWHhYiFiIWIhYoFhYmFi4WKhYyFi4WNhYyFjoWOhY6Fj4WQhZCGkYaRhpKHkoiTiJKLjIiTiZOKlAWUB5QHjJSNkwX8ACcV"
    "wX/4iPnwVZf8iP3wBVz5mhWBB4oHjYGLio2Ci4qOgouKjoKMi46CjIqQg4uKBZCEjIqRg4yKkYWMipKEjIqShoyKk4WNipOHjYqTh46KlIeNipWJjoqUiY+L"
    "lIkFjwaVBo8GlQaPBpSNj4uUjY6MlY2NjJSPjoyTj42Mk4+NjAWTkYyMkpCMjJKSjIyRkYyMkZOMjJCSi4yQk4yMjpSMi46Ui4yOlIuMjZSLjI2VBYwHlQeV"
    "B4wHiZWLjImUi4yIlIuMiJSKi4iUioyGk4uMBYaSioyFk4qMhZGKjISSioyEkIqMg5GJjIOPiYyDj4iMgo+JjIGNiIyCjYeLgo0FhwaBBocGgQaHBoKJh4uC"
    "iYiKgYmJioKHiIqDh4mKg4eJigWDhYqKhIaKioSEioqFhYqKhYOKioaEi4qGg4qKiIKKi4iCi4qIgouKiYKLiomBBYoHgQfGnRWNk46Ti4yOko6Tj5KQkpCR"
    "kJGRkJGQkY+RjpGOkY6RjAWRjZGLkYyRipGLkYmRipGIkYiRiJGHkYaRhpCFkIWQhI+EjoOOhIuKjoONg4yCBYIHggeKgomDiIOLioiEiIOHhIaEhoWGhYWG"
    "hYaFh4WIhYiFiIWKBYWJhYuFioWMhYuFjYWMhY6FjoWOhY+FkIWQhpGGkYaSh5KIk4iSi4yIk4mTipQFlAeUBw73VM75BRXF939RBg74F/ea+fAV+3rF93pR"
    "B/tXFvt6xfd6UQcO+Gng98gViomKiYqJBYkHiQeMiYyJjIn3jvusvpuMi/uG96MFiowGjIwH94b3o4qLWJsFDvhp3acVvnv3jvesjY2MjYuNjI2KjYuNio2J"
    "jfuO96xYe/eG+6MFigeKB/uG+6MFDvnb3ff5FWmrioyKB4yGkWmLipJrjIqUbIuKlm6YcIuKmXGMipp0BYyKnHaMip53jIqfeo2Ki4qgfIyLjYqifo2KjIuj"
    "gI6KpYOPiqaFj4qoh4+LqYoFjgabBo0Gm4yNi5uMjIubjY2Lmo6NiwWajo2Lmo+Ni5qPjIyaj4yMmZCNjJmQjIyZkYyMmJKMi5iTjIuXk4yMl5SMi5eUBYyM"
    "B5WVBYyMBpaVi4yVllmbgYGBgYGCgIOAg4CDf4V/hH+Gf4aLin6Hfod+h32IfYh9iX2KfYkFfIt9inCNco50kHWSiot2lHaXeJl4mnqdi4x7n3yhfaN/pYuM"
    "gaeCqYOrhqyKjQWMjIz4Ca38DYyKjAeJq4uMirGMsYuMjKEFjIyM+Cyt/CiMiowHjZeQrJOrlKmVp4uMl6WZo5qhm5+LjJydnpqemaCXoJSMi6GSopCkjqaN"
    "mYqaiwWZiZmKmYmZiJmImIeYh5iHi4qXhpeGl4SXhZaDloOWg5WClYGVgb2bgZaLjICVBYyKB4GVBYqMBn+Uiot/lIqMf5OKi36TiosFfpKKjH2Riox9kImM"
    "fZCKjHyPiox8j4mLfI+Ji3yOiYt8jomLe42Ki3uMiYt7jAWJBnsGiAZtioeLboeHinCFh4pxg4iKc4CKi4mKdH6JigWKi3Z8i4qJind6iop4d4qKenaKinx0"
    "iop9cYuKfnCAbouKgmyKioRri4qFaYl8BYqKimxppgeMiouKinaLiopkjGSLio1sBYqKim8HDvn9+Cf4khXF914GjIwHjIoG9vspjYmNiY6KjYmMi46KjoqP"
    "igWOBo8GjwaOBo+MjoyOjIyLjY2OjI2NjY329ykFjIwHjIoG+17F97YHjQeKjYuMio2JjYmNiYyIjYiMh4yIjAWHBocGhwaIBoeKiIqIioiJiIqJiYmJ+x37"
    "VAWKBooG+x33VImNiY2IjIiNiIyIjIeMBYgGhwaHBocGiIqHioiKiImJiomJiYmKiYuKiokFiQf7tgf71ffHFWn3B4qM+6TF96SMjPcHrfu2Bw75VN33xxVp"
    "+LCt/LAHDvk23fdtFWn3AIqMigcx+wC/e/P3EYyMBffirfvDBoqMBowH9zj3WgWMjPcfrfsAjIqMBuX3AFebI/sRiooF++Jp98MGjIoGigf7OPtaBYqK+x8G"
    "DvlJ4vfsFYmKi4qJiYqJBYkHiQeJB4yJjYmLio2KjYmNifh0+52zpfxb948FiowGjIwH+Fv3j2Ol/HT7nYmJBaH7+xVp+HSt/HQHDvlJ3d4VsnH4dPedjo2N"
    "jY2Mi4yMjYyNjI2LjYqNio2KjYuMiYyJjYiN/HT3nWRx+Fv7jwWKB4oH/Fv7jwWfKxVp+HSt/HQHDhwTiBwS5fnHFYyIjIeNiAWMiI2IjomNiI2JjomOiY6J"
    "joqOiY+KjouPipKLj4yPi46Mjo2OjI6Njo2OjY6NBY2OjY2Njo2OjI6Mj4yOjI+MjouTio6Kj4qOio+KjomOiY6JjYmOiI2IjYiNiI0FiIyIjYiMh4uHjISL"
    "h4qIi4eKiImIioiJiImIiYmJiYiIiYmIioiJiIqHioiKhwWEB4QHHO1m/LQVjHiMio14i4qPeJF4i4qReYyLknkFi4qUeoyKlHuMipZ7l3uMi5d8jIqZfoyK"
    "mn6Mipp/jYqbgIyKnYGMip2CjYqehAWNip6EjoqehY6Ln4aOi5+Ijoqfio6KooufjI6Ln42Ni5+OjouekI6LnpGNi52SBY2MnpKMjJ2UjYuclYyMnJaMjJuX"
    "jIuamIyMmZmMjJiajIyYmouMl5yLjJWcjIsFk52MjJKcjIyRnYuMkJ2MjI6ejIyNnYuMjZ6Ln4qei4yInoeei4yGnYuMhZ2LjAWEnIqMg52Ki4KcioyAnIqM"
    "f5t+moqMfpiKjH2Yiox8lomMfJaKjHuViYt6lIqMBXqTiYx5kYmMeZGJjHmPiIx5j4iLeI6Ii3iMiIx2i3eKiIt4iYmLd4iJiniHiIsFeIWJi3iEiYp5hIqK"
    "eYOKinqBiot7gIqKe4CKin1/iot+foqKfn6KioB9ioqGhAX4GU/8xIt3B9LlFZGckpuSm5SalJqVmZeYBZeYl5eYlpmVmpWak5qTm5GckZuQnI6bjpuMnIyb"
    "i5uJm4mbiJuHm4aahZuEmoMFmYKagZiAmX+Xf5d9l3yVe5R7lHqSepB6kHmOeY55jHmLeYl5iXmHeYZ5hXqEegWCeoJ7gHt/fX99fX59f32AfIF7gnuDe4V7"
    "hXqGe4h6iHqKeop6i3qNeo56jnmQBXqRe5N7k3uUfJV9ln2Xfph/mICagZqBmoObhJyFnYach52Inoqei56MnY2dj5wFHAgvtRWQoJGfk56UnJWblpqXmZiY"
    "mZaZlJuUmpKckZuPnY6djZ+MBfcIr/sLBnWKh4t2iYeKdoeIi3eFiIp4hYiJeIOJinqBiYp6gIqKBXt/iol9foqKfXyLin97ioqAeYuKgXmLioN3ioqFdoqK"
    "hnaLiod0i4qJc4uKinIF+9nH99kHjKONowX+5vsQFZB0i4qSdIuKlHWLipV2jIqXd4uKmHiMipp5jIqbeoyKnHuNip19jYoFnn6NiqB/jYqhgY2KooOOiqKE"
    "joujho+KpIiPi6SKj4ukjI+LpI6OjKOQjoujkgWOjKGTjoyglY6Mn5eNjJ+YjIyemYyMnJuNjJucjIyZnYyMmJ6MjJefi4yVoIyMBZOhjIyRooyMj6KMjI6j"
    "i4yMpIqki4yIo4qMh6KKjIWiioyDoYqMgaCLjH+fiowFfp6KjH2diox7nImMepuKjHiZiox3mImMd5eIjHaViIx1k4iMc5KIi3OQiIxyjgWHi3KMh4tyioeL"
    "coiHinOGiIt0hIiKdIOJinWBiYp2f4mKeH6Jinl9iYp6e4qKBXt6iop8eYqKfniLin93ioqBdouKgnWLioR0i4qGdIuKiHOLiopyi4uMcouKjnMFy/cLFZKh"
    "k6CVn5afmJ2YnJqbnJmcmJ2XBZ2Wn5Ofk5+Qn5CgjaGMoIqgiaCGn4afg56DnoCdf5x+m32ae5l6mHmWd5V3k3YFkXWQdY50jHOKc4h0hnWFdYN2gXeAd355"
    "fXp8e3t9en55f3iAeIN3g3eGdoZ2iQV2inWMdo13kHeQd5N3k3mWeZd6mHqZfJt+nH6dgJ+Bn4OghKGHoYiiiqOMo46iBRwGavvqFYqIi4eMiIyJjYmNiY6J"
    "joqOiY6Kj4uPigX4yK/8kQb4sPjRjY2MjYyOi42KjYuOiY2KjYiNiY2IjIiNh4yHi4iMBfybZ/hjBvyv/NGJiQX6NPeGFZB0i4qSdIuKlHWLipV2jIqXd4uK"
    "mHiMipp5jIqbeoyKnHuNip19jYoFnn6NiqB/jYqhgY2KooOOiqKEjoujho+KpIiPi6SKj4ukjI+Lo46PjKOQjoujkgWOjKGTjoyglY6Mn5eNjJ+YjIyemYyM"
    "nJuNjJucjIyZnYyMmJ6MjJefi4yVoIyMBZOhjIyRooyMj6KMjI6ji4yMpIqki4yIo4qMh6KKjIWiioyDoYqMgaCLjH+fiowFfp6KjH2diox7nImMepuKjHiZ"
    "iox3mImMd5eIjHaViIx1k4iMc5KIi3OQh4xzjgWHi3KMh4tyioeLcoiHinOGiIt0hIiKdIOJinWBiYp2f4mKeH6Jinl9iYp6e4qKBXt6iop8eYqKfniLin93"
    "ioqBdouKgnWLioR0i4qGdIuKiHOLiopyi4uMcouKjnMFy/cLFZKhk6CVn5afl52ZnJqbnJmcmJ2XBZ2Wn5Ofk5+Qn5CgjaGMoIqgiaCGn4afg56DnoCdf5x+"
    "m32ae5l6mHmWd5R3lHYFkXWQdY50jHOKc4h0hnWFdYJ2gneAd355fXp8e3t9en55f3iAeIN3g3eGdoZ2iQV2inWMdo13kHeQd5N3k3mWeZd6mHqZfJt9nH+d"
    "gJ+Bn4OghKGHoYiiiqOMo46iBfqe++8Vx/jsTwYOBgAAAQAAAAoAJAAyAAJERkxUAA5sYXRuAA4ABAAAAAD//wABAAAAAWtlcm4ACAAAAAEAAAABAAQAAgAA"
    "AAEACAABACoABAAAABAATgBUAGYAbAB2AHwAkgCkALYAyADOAOQA9gESAQQBEgABABAADwAiACcALQAxADUANwA4ADoARgBHAFMAVQBXAFgAWgABAA//9wAE"
    "ADX/3wA3/8oAOP/WADr/xwABACL/5AACADX/6gA6/+IAAQAi/+sABQAi/98AQv/kAEb/4ABQ/+AAWv/rAAQAIv/KAEL/5ABG/+QAUP/kAAQAIv/WAEL/6gBG"
    "/+oAUP/qAAQAIv/HAEL/3wBG/98AUP/eAAEARwAMAAUAQv/yAEb/8QBKAAYATQAEAFD/8QAEAEL/+ABG//gAUP/3AFUABAADAEL/9gBG//QAUP/0AAMAQv/w"
    "AEb/8QBQ//EAAwBC/+sARv/uAFD/7AAAAAEAAAAKACYAOgACREZMVAAObGF0bgAOAAQAAAAA//8AAgAAAAEAAmxpZ2EADnJsaWcADgAAAAEAAAABAAQABAAA"
    "AAEACAABABoAAQAIAAEABAFYAAYAUABTAFsAUABKAAEAAQBDAAACWAAAAUoAAADSAEMBgwBDAt4AUgNAAFIDLABSAx0AUgDAAEMBhQBDAYUAQwJdAFICrABS"
    "AO8AQwIWAGEA0gBDAuMAUgM6AFQBzwB6AyoAVAM6AFQDMwBQAzYAVAMcAFQDOABQAzoAVAMcAFQA0gBDAO8AQwK2AFICogBSArYAUgMbAFIDxwBSA08AUANU"
    "AFIDUgBSA2MAUgL5AE4C8QBOA2MAUgNKAFIAsgA8AvEARgNaAE4DGQBOA3IAUgNKAFIDhgBSAzMAUgOqAFIDZgBSA1UAUAM0AEYDSgBSA2wARgSHAEIDXwBG"
    "A2kARgNjAEgBXABDAuMAUgFcAEMC8wBSAugAUgEwAEMDOgBUA14AVgMLAFIDOgBUAxIAUgHGADoDOgBUAyYAVADKAD4BgwA2AxcATACyADwEqABSAyYAVANO"
    "AFYDOgBUAzoAVAIeAEQC9wBOAfIAPAMcAFQDLwBGBHMAQgMWAEYDLwBGAv8ASAHCAEMAwABDAcIAQwLyAFIBSgAAAMAAQwMPAFIDBQBSAwYAUgMnAFIA3gBS"
    "AtcAUgGaAEMDmgBSAfYASAKdAEgCyQBSAhYAYQOaAFIBvABIAeIAUgKsAFIB9gA5AgEAOQFGAEMDAwBUAvsAUgDSAEMBMABDAPAAOQH2AEgCnQBIA6QAQwOW"
    "AEMDpABDAxsAUgNPAFADTwBQA08AUANPAFADTwBQA08AUAOfAFIDUgBSAvkATgL5AE4C+QBOAvkATgE/ADwBPwA8AakAPAF8ADwDeABSA0oAUgOGAFIDhgBS"
    "A4YAUgOGAFIDhgBSAsoAUgOGAFIDSgBSA0oAUgNKAFIDSgBSA2kARgM7AFIDHQBSAzoAVAM6AFQDOgBUAzoAVAM6AFQDOgBUBTQAUgMLAFIDEgBSAxIAUgMS"
    "AFIDEgBSAUMAPgFDAD4BrQA+AYAAPgM2AFIDJgBUA04AVgNOAFYDTgBWA04AVgNOAFYCogBSA0YAUgMcAFQDHABUAxwAVAMcAFQDLwBGAzYAUgMvAEYDTwBQ"
    "AzoAVANPAFADOgBUA08AUAM6AFQDUgBSAwsAUgNSAFIDCwBSA1IAUgMLAFIDUgBSAwsAUgNjAFIDOgBUA3gAUgOWAFIC+QBOAxIAUgL5AE4DEgBSAvkATgMS"
    "AFIC+QBOAxIAUgL5AE4DEgBSA2MAUgM6AFQDYwBSAzoAVANjAFIDOgBUA2MAUgM6AFQDSgBSAyYAVAOcAFIDSwBSAdQAPAHYAD4BpAA8AagAPgHNADwB0QA+"
    "AVAAPAFeAD4AxAA8AN4AUgM2AFIB7ABSAvEARgGlADYDWgBOAxcATAMOAFIDGQBOAT8APAMZAE4BGAA8AxkATgGpADwDcQBSAZsAUgNWAFIBxQBSA0oAUgMm"
    "AFQDSgBSAyYAVANKAFIDJgBUAyYAVAP2AFIDIgBSA4YAUgNOAFYDhgBSA04AVgOGAFIDTgBWBQMAUgVXAFIDZgBSAh4ARANmAFICHgBEA2YAUgIeAEQDVQBQ"
    "AvcATgNVAFAC9wBOA1UAUAL3AE4DVQBQAvcATgM0AEYB8gA8AzQARgHyADwDTABSAi8AUgNKAFIDHABUA0oAUgMcAFQDSgBSAxwAVANKAFIDHABUA0oAUgMc"
    "AFQDSgBSAxwAVASHAEIEcwBCA2kARgMvAEYDaQBGA2MASAL/AEgDYwBIAv8ASANjAEgC/wBIAZoAUgIWAGECwABSAsAAUgPJAFIDyQBSAQ0AUgENAFIB3wBS"
    "Ad8AUgJcAFICXABSARIAUgJEAFIDLABSAMAAQwGDAEMB1QBSAdUAUgNHAFIDaQBSAsAAUgKiAFICtQBSArUAUhOIAEo="
)
OVERLAY_JS = OVERLAY_JS.replace("__BORZOI__", BORZOI_OTF_B64)  # both copies: page scope + shadow scope

DEFAULT_PROFILE_DIR = str(Path.home() / ".whippet" / "profile")
DEFAULT_ALIASES_PATH = str(Path.home() / ".whippet" / "aliases.json")
DEFAULT_SLIDES_PATH = str(Path.home() / ".whippet" / "slides.pdf")

# --------------------------------------------------------------------------------------
# Slides: a PDF rendered page-by-page (PyMuPDF) and served from memory at a private origin the
# browser never reaches over the network. One image per slide, so the page costs nothing to show.
# --------------------------------------------------------------------------------------
SLIDES_ORIGIN = "https://whippet.local"
SLIDES_URL = SLIDES_ORIGIN + "/slides/"
SLIDES_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#0b0d12;overflow:hidden;font-family:-apple-system,system-ui,sans-serif}
#stage{position:relative;box-sizing:border-box;width:100%;height:100vh;display:flex;align-items:center;justify-content:center;padding:12px}
main{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden}
#img{max-width:100%;max-height:100%;object-fit:contain;box-shadow:0 20px 60px rgba(0,0,0,.6);background:#fff}
#num{position:fixed;left:18px;bottom:14px;color:rgba(255,255,255,.55);font-size:14px;letter-spacing:.04em;font-variant-numeric:tabular-nums}
#txt{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden}
</style></head><body>
<div id="stage"><img id="img" alt="slide"></div><div id="num"></div>
<main><h1 id="h">__TITLE__</h1><p id="txt"></p></main>
<script>
(() => {
  const N = __COUNT__, texts = __TEXTS__;
  const img = document.getElementById('img'), num = document.getElementById('num'), txt = document.getElementById('txt');
  const cache = {};
  const pre = n => { if (n >= 1 && n <= N && !cache[n]) { const i = new Image(); i.src = n + '.png'; cache[n] = i; } };
  let cur = 0;
  function go(n) {
    n = Math.max(1, Math.min(N, n | 0));
    if (n !== cur) {
      cur = n;
      img.src = n + '.png';
      num.textContent = n + ' / ' + N;
      txt.textContent = texts[n - 1] || '';
      document.title = '__TITLE__ - slide ' + n + ' of ' + N;
      history.replaceState(null, '', '#' + n);
    }
    pre(n + 1); pre(n - 1);
    return {n: cur, count: N};
  }
  window.__slides = {go, next: () => go(cur + 1), prev: () => go(cur - 1), get n() { return cur; }, count: N};
  addEventListener('keydown', e => {
    if (e.target && /input|textarea/i.test(e.target.tagName)) return;
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(e.key)) { go(cur + 1); e.preventDefault(); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(e.key)) { go(cur - 1); e.preventDefault(); }
    else if (e.key === 'Home') go(1); else if (e.key === 'End') go(N);
  });
  addEventListener('click', e => { if (e.target === img || e.target.id === 'stage') go(cur + (e.clientX > innerWidth / 2 ? 1 : -1)); });
  go(parseInt(location.hash.slice(1), 10) || 1);
})();
</script></body></html>"""


class SlideDeck:
    def __init__(self, path: str):
        self.path = str(Path(path).expanduser())
        self.title = Path(self.path).stem.replace("_", " ").replace("-", " ")
        self.pngs: list[bytes] = []
        self.texts: list[str] = []
        self.ready = False
        self.error: str | None = None
        self.load_ms = 0
        self.current = 1  # last slide shown, so reopening the deck resumes there
        self.loaded = threading.Event()  # set once `load` has finished (ready or error)

    @property
    def count(self) -> int:
        return len(self.pngs)

    async def wait(self, timeout: float = 60.0) -> bool:
        """Await the (possibly still running) background load; True when the deck is usable."""
        if not self.loaded.is_set():
            await asyncio.get_running_loop().run_in_executor(None, self.loaded.wait, timeout)
        return self.ready

    def load(self, scale: float = 1.6) -> None:
        t0 = time.time()
        try:
            import pymupdf
            doc = pymupdf.open(self.path)
            pngs, texts = [], []
            for page in doc:
                pngs.append(page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png"))
                texts.append(re.sub(r"[ \t]+", " ", page.get_text()).strip())
            if not pngs:
                raise ValueError("the PDF has no pages")
            meta_title = (doc.metadata or {}).get("title", "").strip()
            if meta_title and len(meta_title) <= 60:
                self.title = meta_title
            self.pngs, self.texts, self.ready = pngs, texts, True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        self.load_ms = int((time.time() - t0) * 1000)
        self.loaded.set()
        log.info("slides: %s pages=%d in %dms %s", self.path, self.count, self.load_ms, self.error or "")

    def html(self) -> str:
        return (SLIDES_HTML.replace("__TITLE__", html_mod.escape(self.title)).replace("__COUNT__", str(self.count))
                .replace("__TEXTS__", json.dumps(self.texts).replace("</", "<\\/")))

    async def route(self, route) -> None:
        """Playwright route handler for SLIDES_ORIGIN."""
        path = urlparse(route.request.url).path
        m = re.fullmatch(r"/slides/(\d+)\.png", path)
        if m and 1 <= int(m.group(1)) <= self.count:
            await route.fulfill(status=200, content_type="image/png", body=self.pngs[int(m.group(1)) - 1],
                                headers={"Cache-Control": "max-age=86400"})
        elif path in ("/slides", "/slides/"):
            await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=self.html())
        else:
            await route.fulfill(status=404, content_type="text/plain", body="not found")


class Browser:
    def __init__(self) -> None:
        self.pw = None
        self.browser = None
        self.context = None
        self.page = None
        self.pages: list = []
        self.on_tabs: Callable[[], None] | None = None
        self.on_message: Callable[[dict, Any], None] | None = None  # messages from the in-page assistant
        self.deck: SlideDeck | None = None

    async def launch(self, headless: bool = False, cdp: str | None = None, profile_dir: str = DEFAULT_PROFILE_DIR,
                     start_url: str = "about:blank") -> None:
        from playwright.async_api import async_playwright
        self.pw = await async_playwright().start()
        if cdp:
            self.browser = await self.pw.chromium.connect_over_cdp(cdp)
            self.context = self.browser.contexts[0] if self.browser.contexts else await self.browser.new_context()
        else:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            args = ["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream",
                    "--hide-crash-restore-bubble"]
            if headless:  # no real microphone in headless runs: Chromium's fake device (tone) keeps getUserMedia working
                args += ["--use-fake-device-for-media-stream"]
            if not headless:
                args += ["--window-size=1280,900", "--window-position=40,40"]
            self.context = await self.pw.chromium.launch_persistent_context(
                profile_dir, headless=headless, viewport={"width": 1280, "height": 900} if headless else None,
                no_viewport=not headless,
                args=args, ignore_default_args=["--enable-automation"],
                bypass_csp=True,  # strict-CSP sites would otherwise strip the overlay's styles (invisible assistant)
            )
            try:
                await self.context.grant_permissions(["microphone"])
            except Exception as e:
                log.debug("microphone permission: %s", e)
        try:
            await self.context.expose_binding("__vbSend", self._from_page)
        except Exception as e:  # already exposed on a re-attached CDP context
            log.debug("expose_binding: %s", e)
        await self.context.add_init_script(OVERLAY_JS)
        await self.context.route(SLIDES_ORIGIN + "/**", self._slides_route)
        self.context.on("page", lambda p: asyncio.ensure_future(self._on_page(p)))
        for p in self.context.pages:
            self._track(p)
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        self._track(self.page)
        if start_url != "about:blank":
            await self.page.goto(start_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        for p in self.pages:
            await self.mount(p)

    async def _slides_route(self, route) -> None:
        if self.deck and self.deck.ready:
            await self.deck.route(route)
        else:
            await route.fulfill(status=200, content_type="text/html; charset=utf-8",
                                body="<title>Whippet slides</title><body style='background:#0b0d12;color:#ccc;font:16px system-ui;"
                                     "padding:40px'>No slides loaded. Start Whippet with --slides your-deck.pdf, or say "
                                     "\"open slides\" followed by the file path.</body>")

    def on_slides(self, page=None) -> bool:
        p = page or self.page
        return bool(p is not None and not p.is_closed() and p.url.startswith(SLIDES_URL))

    async def mount(self, page) -> None:
        """Pages that were open before we attached (CDP) never ran the init script: inject it."""
        try:
            await page.evaluate(OVERLAY_JS)
        except Exception as e:
            log.debug("mount: %s", e)

    def _from_page(self, source: dict, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if self.on_message:
            self.on_message(msg, source.get("page"))

    async def push(self, msg: dict, page=None, all_pages: bool = False) -> None:
        """Deliver an event to the assistant overlay of the active page (or every page)."""
        targets = list(self.pages) if all_pages else [page or self.page]
        data = json.dumps(msg, default=str)
        for p in targets:
            if p is None or p.is_closed():
                continue
            try:
                await p.evaluate("(m) => window.__vb && window.__vb.event(JSON.parse(m))", data)
            except Exception:
                pass

    def _track(self, page) -> None:
        if page in self.pages:
            return
        self.pages.append(page)

        def closed(_p=page):
            if _p in self.pages:
                self.pages.remove(_p)
            if self.page is _p:
                self.page = self.pages[-1] if self.pages else None
            if self.on_tabs:
                self.on_tabs()
        page.on("close", lambda: closed())

    async def _on_page(self, page) -> None:
        self._track(page)
        if self.on_tabs:
            self.on_tabs()

    async def set_active(self, page) -> None:
        self.page = page
        try:
            await page.bring_to_front()
        except Exception:
            pass
        if self.on_tabs:
            self.on_tabs()

    async def ensure_page(self):
        if self.page is None or self.page.is_closed():
            self.page = await self.context.new_page()
            self._track(self.page)
        return self.page

    async def snapshot(self) -> dict:
        page = await self.ensure_page()
        try:
            data = await page.evaluate(COLLECT_JS)
        except Exception as e:  # navigating / detached frame
            log.debug("snapshot failed: %s", e)
            data = {"url": page.url, "title": "", "elements": []}
        return build_snapshot(data)

    async def overlay(self, fn: str, *args: Any) -> None:
        page = await self.ensure_page()
        try:
            await page.evaluate("([fn, args]) => window.__vb && window.__vb[fn] && window.__vb[fn](...args)", [fn, list(args)])
        except Exception:
            pass

    def tabs(self) -> list[dict]:
        return [{"index": i, "url": p.url, "active": p is self.page} for i, p in enumerate(self.pages)]

    async def close(self) -> None:
        try:
            if self.context:
                await self.context.close()
        finally:
            if self.pw:
                await self.pw.stop()


_FINGERPRINT_JS = "[...document.querySelectorAll('a[href]')].slice(0, 40).map(a => a.textContent.trim()).join('|')"


async def _fingerprint(page) -> str:
    """What the page currently lists; an SPA swaps its URL long before it swaps this."""
    try:
        return await page.evaluate(_FINGERPRINT_JS)
    except Exception:
        return ""


async def _settle(page, ms: int = 1800, before: str | None = None) -> None:
    """domcontentloaded, then wait until the DOM stops growing (SPAs render results well after load).
    With `before` (the pre-action fingerprint) also wait for the content to actually change first."""
    try:
        await asyncio.wait_for(page.wait_for_load_state("domcontentloaded"), timeout=ms / 1000)
    except Exception:
        pass
    deadline = time.time() + ms / 1000
    change_by = time.time() + 1.2  # a click that legitimately changes nothing (toggle, new tab) must not stall
    last, stable, url = -1, 0, page.url
    while time.time() < deadline:
        await page.wait_for_timeout(100)
        if before is not None:
            if time.time() < change_by and await _fingerprint(page) == before:
                continue
            before = None
        if page.url != url:  # a redirect interstitial handed us on: start over on the new document
            url, last, stable = page.url, -1, 0
            deadline = time.time() + ms / 1000
            try:
                await asyncio.wait_for(page.wait_for_load_state("domcontentloaded"), timeout=ms / 1000)
            except Exception:
                pass
            continue
        try:
            n = await page.evaluate("document.querySelectorAll('a[href],button,input,textarea,select').length")
        except Exception:
            continue
        stable = stable + 1 if n == last else 0
        if stable >= 3 and n > 0:  # unchanged for ~300ms
            return
        last = n


async def execute(action: dict, browser: Browser) -> dict:
    page = await browser.ensure_page()
    label = describe(action)
    t = action["type"]

    if t == "navigate_url":
        await browser.overlay("toast", f"→ {label}")
        host = re.sub(r"^www\.", "", urlparse(action["url"]).hostname or "")
        try:
            await page.goto(action["url"], wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        except Exception as e:
            await asyncio.sleep(0.5)
            try:
                await page.goto(action["url"], wait_until="commit", timeout=NAV_TIMEOUT)
            except Exception:
                pass
            if host not in page.url:
                raise e
        await _settle(page)
        return {"ok": True, "detail": page.url}

    if t == "click_element":
        before = list(browser.pages)
        await browser.overlay("clearCandidates")
        await browser.overlay("highlight", action["targetId"], HIGHLIGHT_MS)
        await browser.overlay("toast", label)
        loc = page.locator(f'[data-vb-id="{action["targetId"]}"]').first
        fp = await _fingerprint(page)
        try:
            await loc.click(timeout=4000)
        except Exception:
            await loc.evaluate("el => el.click()")
        await _settle(page, before=fp)
        fresh = [p for p in browser.pages if p not in before]
        if not fresh:
            await asyncio.sleep(0.15)  # a target=_blank tab may still be opening
            fresh = [p for p in browser.pages if p not in before]
        if fresh:
            await browser.set_active(fresh[0])
        return {"ok": True, "detail": browser.page.url}

    if t == "type_into_field":
        await browser.overlay("clearCandidates")
        await browser.overlay("highlight", action["targetId"], HIGHLIGHT_MS + 400)
        await browser.overlay("toast", label)
        loc = page.locator(f'[data-vb-id="{action["targetId"]}"]').first
        try:
            await loc.click(timeout=4000)
        except Exception:
            await loc.focus()
        try:
            await loc.fill("")
        except Exception:
            pass
        try:  # real key events (autocomplete / React listeners), fast: ~6ms per char
            await loc.press_sequentially(action["text"], delay=6)
        except Exception:
            await loc.fill(action["text"])
        if action.get("submit"):
            fp = await _fingerprint(page)
            await page.keyboard.press("Enter")
            await _settle(page, before=fp)
        return {"ok": True, "detail": page.url}

    if t == "select_option":
        await browser.overlay("highlight", action["targetId"], HIGHLIGHT_MS)
        await browser.overlay("toast", label)
        loc = page.locator(f'[data-vb-id="{action["targetId"]}"]').first
        picked = await loc.evaluate("""(sel, wanted) => { const w = wanted.toLowerCase(); const opts = Array.from(sel.options || []);
            const hit = opts.find(o => o.label.toLowerCase() === w) || opts.find(o => o.label.toLowerCase().includes(w));
            if (!hit) return null; sel.value = hit.value; sel.dispatchEvent(new Event("change", { bubbles: true })); return hit.label; }""",
                                   action.get("text") or "")
        return {"ok": bool(picked), "detail": picked or "no matching option"}

    if t == "press_enter":
        await browser.overlay("toast", "⏎ enter")
        fp = await _fingerprint(page)
        await page.keyboard.press("Enter")
        await _settle(page, before=fp)
        return {"ok": True, "detail": page.url}

    if t in ("scroll_down", "scroll_up"):
        d = 1 if t == "scroll_down" else -1
        await browser.overlay("toast", label)
        await page.evaluate("""([dir, amount]) => { const vh = window.innerHeight;
            if (amount === "end") window.scrollTo({ top: dir > 0 ? document.documentElement.scrollHeight : 0, behavior: "smooth" });
            else window.scrollBy({ top: dir * (amount === "little" ? vh * 0.35 : vh * 0.85), behavior: "smooth" }); }""",
                            [d, action.get("amount") or "page"])
        await page.wait_for_timeout(250)
        y = await page.evaluate("() => Math.round(window.scrollY)")
        return {"ok": True, "detail": f"scrollY={y}"}

    if t == "go_back":
        await browser.overlay("toast", "← back")
        try:
            await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        except Exception:
            pass
        await _settle(page, 800)
        return {"ok": True, "detail": page.url}

    if t == "go_forward":
        await browser.overlay("toast", "→ forward")
        try:
            await page.go_forward(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        except Exception:
            pass
        await _settle(page, 800)
        return {"ok": True, "detail": page.url}

    if t == "reload":
        await browser.overlay("toast", "↻ reload")
        try:
            await page.reload(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        except Exception:
            pass
        return {"ok": True, "detail": page.url}

    if t == "open_new_tab":
        p = await browser.context.new_page()
        browser._track(p)
        await browser.set_active(p)
        await browser.overlay("toast", "new tab")
        return {"ok": True, "detail": f"tabs={len(browser.pages)}"}

    if t == "close_tab":
        await page.close()
        await asyncio.sleep(0.1)
        if not browser.pages:
            p = await browser.context.new_page()
            browser._track(p)
        await browser.set_active(browser.pages[-1])
        return {"ok": True, "detail": f"tabs={len(browser.pages)}"}

    if t == "switch_tab":
        pages = browser.pages
        if len(pages) < 2:
            return {"ok": False, "detail": "only one tab"}
        i = pages.index(browser.page) if browser.page in pages else 0
        d = action.get("direction", "next")
        nxt = pages[(i - 1) % len(pages)] if d == "previous" else pages[0] if d == "first" else pages[(i + 1) % len(pages)]
        await browser.set_active(nxt)
        await browser.overlay("toast", "switched tab")
        return {"ok": True, "detail": nxt.url}

    return {"ok": False, "detail": f"unknown action {t}"}


# --------------------------------------------------------------------------------------
# Controller: transcripts -> decisions -> actions
# --------------------------------------------------------------------------------------
@dataclass
class Utterance:
    id: str
    text: str = ""
    final: bool = False
    acted: bool = False
    done: bool = False  # final utterance resolved without an action (wait/ignore)
    queue: list[str] = field(default_factory=list)  # remaining steps of a multi-command utterance
    split: bool = False
    resnapped: bool = False  # already re-read a still-loading page once for this utterance
    changed_at: float = field(default_factory=time.time)
    consumed: int = 0  # chars of `text` already handled by a reflex ("…no wait, go back")
    gated: str = ""  # last fragment the front gate looked at
    gate: dict | None = None  # its verdict
    guided: bool = False  # "guide me through …": announce each step and wait for "next" between them
    step_no: int = 0
    step_total: int = 0

    @property
    def live(self) -> str:
        return self.text[self.consumed:].strip()


class Controller:
    def __init__(self, browser: Browser, decider: Decider, execute_fn=execute, aliases: AliasStore | None = None,
                 llm: LLM | None = None, tts: TTS | None = None, stt: STT | None = None):
        self.browser = browser
        self.decider = decider
        self.execute_fn = execute_fn
        self.aliases = aliases or AliasStore(None)
        self.gate = Gate(decider.model, self.aliases)
        self.llm = llm
        self.tts = tts
        self.stt = stt
        self.voice_on = False
        self.ptt = False
        self.fake_mic = False
        self.zoom = 1.0
        self.last_said: dict | None = None
        self.convo: list[dict] = []
        self.session = VoiceSession(stt, self._voice_text, self._voice_started, hot=self._interruptible) if stt else None
        if stt:
            stt.set_vocabulary([a.phrase for a in self.aliases.items])
        self.snapshot: dict = {"elements": [], "site": "blank", "url": "about:blank", "title": ""}
        self.pending: dict | None = None  # destructive action awaiting "confirm" (or a remap awaiting "yes")
        self.guide: Utterance | None = None  # guided mode: the next step, waiting for "next"
        self.candidates: dict | None = None  # {"items": [...], "pendingIntent": {...}}
        self.history: list[dict] = []
        self.listeners: list[Callable[[dict], None]] = []
        self.utt: Utterance | None = None
        self.stats = {"decisions": 0, "actions": 0, "reflexes": 0, "asks": 0, "decision_ms": [], "action_ms": [],
                      "gate_ms": []}
        self.running: Utterance | None = None  # utterance whose action / step queue is executing
        self._last_page_log = ""
        self._debounce: asyncio.TimerHandle | None = None
        self._busy = asyncio.Lock()
        self._gate_lock = asyncio.Lock()
        self._loop = asyncio.get_event_loop()
        browser.on_tabs = self._tabs_changed
        browser.on_message = self._from_page
        self.on(self._to_page)

    def on(self, fn: Callable[[dict], None]) -> None:
        self.listeners.append(fn)

    def emit(self, msg: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception as e:
                log.debug("listener error: %s", e)

    def log(self, text: str, level: str = "info") -> None:
        getattr(log, level if level != "warn" else "warning")(text)
        self.emit({"type": "log", "level": level, "text": text, "t": time.time()})

    async def start(self) -> None:
        await self.refresh_snapshot()
        if self.tts or self.stt:
            asyncio.ensure_future(self._watch_speech_models())

    async def _watch_speech_models(self) -> None:
        """Tell the overlays when Pocket TTS / Whisper finish loading (the mic only starts once Whisper is ready)."""
        def settled(m) -> bool:
            return m is None or m.ready or bool(m.error)
        while not (settled(self.tts) and settled(self.stt)):
            await asyncio.sleep(0.3)
        await self.push_state()
        if self.voice_on:
            self.say("Ready. I'm listening.", "info")

    # ---- the in-browser assistant ----------------------------------------------------------
    PAGE_EVENTS = {"transcript", "decision", "action", "say", "answer", "candidates", "pending", "aliases", "tabs", "voice",
                   "ptt", "suggestions", "guide"}

    def _to_page(self, msg: dict) -> None:
        if msg.get("type") == "aliases" and self.stt:
            self.stt.set_vocabulary([a.phrase for a in self.aliases.items])
        if msg.get("type") not in self.PAGE_EVENTS:
            return
        if msg["type"] == "decision":
            msg = {"type": "decision", "decision": msg.get("decision"), "summary": msg.get("summary")}
        elif msg["type"] == "action":
            msg = {"type": "action", "action": {"type": msg["action"].get("type"), "label": msg["action"].get("label")},
                   "result": {"ok": bool(msg.get("result", {}).get("ok"))}}
        asyncio.ensure_future(self.browser.push(msg, all_pages=msg["type"] in ("tabs", "aliases", "voice", "ptt")))

    def _interruptible(self) -> bool:
        return bool(self.running or (self.tts and self.tts.speaking) or self.pending or self.candidates
                    or (self.utt and self.utt.queue))

    def _tabs_changed(self) -> None:
        self.emit({"type": "tabs", "tabs": self.browser.tabs()})
        asyncio.ensure_future(self.push_state())

    async def push_state(self) -> None:
        for p in list(self.browser.pages):
            await self.browser.push(self.page_state(p), page=p)

    def page_state(self, page=None) -> dict:
        return {"type": "state", "voice": self.voice_on, "ptt": self.ptt, "active": page is None or page is self.browser.page,
                "stt": self.stt.info if self.stt else None, "tts": self.tts.info if self.tts else None,
                "llm": self.llm.info if self.llm else None, "tabs": self.browser.tabs(), "aliases": self.aliases.as_list(),
                "candidates": self.candidates["items"] if self.candidates else [], "pending": self.pending, "last": self.last_said,
                "history": self.convo[-24:], "suggestions": self.suggestions(),
                "guide": {"step": self.guide.step_no, "total": self.guide.step_total, "text": self.guide.text} if self.guide else None}

    def suggestions(self) -> list[str]:
        """Three things worth trying right now, from what is on screen (shown under the conversation)."""
        snap = self.snapshot or {}
        page = self.browser.page
        url = (page.url if page and not page.is_closed() else "") or snap.get("url") or ""
        if url != (snap.get("url") or ""):
            snap = {}
        deck = self.browser.deck
        if url.startswith(SLIDES_URL) and deck:
            n = deck.current
            pool = (["next", "last slide", f"slide {min(deck.count, max(1, n + 3))}", "go to the end", "first slide", "what slide am I on",
                     "what does this slide say", "close the slides"])
            pool = [x for x in pool if not (x == "next" and n >= deck.count) and not (x == "last slide" and n <= 1)]
        elif not url or url == "about:blank" or url.startswith("chrome://"):
            pool = ["go to wikipedia", "open hacker news", "go to bbc news", "open example dot com", "search for whippet dog",
                    "open a new tab", "open my slides" if deck else "help"]
        else:
            els = snap.get("elements") or []
            links = [e for e in els if e.get("role") == "link" and 4 <= len(e.get("text") or "") <= 28]
            pool = []
            if snap.get("video"):
                pool += ["pause the video", "turn on captions", "speed 1.5x", "skip ahead ten seconds"]
            if snap.get("searchBoxId"):
                pool.append("search for whippet")
            if links:
                pool.append(f"click {links[len(links) // 2]['text'].strip()}")
                pool.append("click the first link")
            pool += ["what's on this page", "scroll down", "read the page", "summarize this page", "go back", "open a new tab", "zoom in",
                     "when I say bounce, go back"]
        seen: list[str] = []
        for x in pool:
            if x.lower() not in [y.lower() for y in seen]:
                seen.append(x)
        return seen[:3]

    def _remember(self, role: str, text: str, kind: str = "") -> None:
        """Conversation shown in the panel; kept here so it survives page navigations."""
        if self.convo and self.convo[-1]["role"] == role and self.convo[-1]["text"] == text:
            return
        self.convo.append({"role": role, "text": text[:600], "kind": kind})
        del self.convo[:-60]

    def _from_page(self, msg: dict, page) -> None:
        t = msg.get("type")
        if t != "audio":
            log.debug("page -> %s", msg)
        if t == "audio":
            if page is self.browser.page and self.session and self.voice_on and not self.fake_mic:
                self._feed_audio(msg.get("pcm", ""), int(msg.get("rate") or STT_RATE))
        elif t == "hello":
            asyncio.ensure_future(self.browser.push(self.page_state(page), page=page))
        elif t == "command":
            self.handle_command(msg.get("text", ""))
        elif t == "transcript":  # Web Speech fallback (no local STT)
            self._voice_text(msg.get("text", ""), bool(msg.get("final")), msg.get("utteranceId") or "w")
        elif t == "voice":
            asyncio.ensure_future(self.set_voice(bool(msg.get("on"))))
        elif t == "ptt":
            asyncio.ensure_future(self.set_ptt(bool(msg.get("on"))))
        elif t == "forget":
            self.aliases.remove(msg.get("phrase", ""))
            self.emit({"type": "aliases", "aliases": self.aliases.as_list()})
        elif t == "switch_tab":
            pages = self.browser.pages
            i = int(msg.get("index", -1))
            if 0 <= i < len(pages):
                asyncio.ensure_future(self.browser.set_active(pages[i]))
        elif t == "spoken":
            if self.tts and msg.get("gen") == self.tts.gen:
                self.tts.done_speaking()
        elif t == "mic_error":
            text = str(msg.get("text", ""))
            if text != self._last_page_log:
                self._last_page_log = text
                self.log(f"page: mic: {text}", "warn")
                if not self.fake_mic:
                    self.say("I can't hear you: the microphone isn't available in this browser. You can still type to me.", "error")
        elif t == "log":
            text = str(msg.get("text", ""))
            if text != self._last_page_log:  # pages can repeat the same complaint at high rate
                self._last_page_log = text
                self.log(f"page: {text}", "warn")

    def _feed_audio(self, b64: str, rate: int) -> None:
        pcm = base64.b64decode(b64)
        if rate != STT_RATE:  # the page could not open a 16 kHz context: resample here
            import numpy as np
            x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            n = int(len(x) * STT_RATE / rate)
            pcm = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("<i2").tobytes()
        self.session.feed(pcm)

    async def set_ptt(self, on: bool) -> None:
        """Push-to-talk: the mic only reaches Whisper while Space is held (for loud rooms). Implies voice mode."""
        self.ptt = on
        self.log(f"push-to-talk {'on' if on else 'off'}")
        self.emit({"type": "ptt", "on": on})
        if on and not self.voice_on:
            await self.set_voice(True)
        self.say("Push to talk. Hold space while you speak." if on else "Push to talk off. I'm always listening again.", "info")

    async def set_voice(self, on: bool) -> None:
        if on and self.stt and not self.stt.ready and not self.stt.error:
            self.say("Hearing is still warming up. One moment.", "info")
        self.voice_on = on
        self.log(f"voice mode {'on' if on else 'off'}")
        self.emit({"type": "voice", "on": on})
        if on:
            self.say("Listening.", "info")
        else:
            self.tts and self.tts.interrupt()

    def _voice_started(self) -> None:
        pass  # the user began talking; echo of our own voice also lands here, so the interrupt waits for words

    async def play_fake_mic(self, wav: Path) -> None:
        """Voice-mode test harness: stream a 16 kHz mono WAV into the recogniser in real time, as if spoken."""
        import wave
        while not (self.stt and (self.stt.ready or self.stt.error)):
            await asyncio.sleep(0.2)
        if not self.stt.ready:
            self.log("fake mic: no STT", "warn")
            return
        self.fake_mic = True
        with wave.open(str(wav), "rb") as w:
            assert w.getframerate() == STT_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono 16-bit WAV"
            step = STT_RATE // 10
            self.log(f"fake mic: {wav.name} ({w.getnframes() / STT_RATE:.0f}s)")
            t0 = time.time()
            i = 0
            while True:
                frames = w.readframes(step)
                if not frames:
                    break
                if self.voice_on and self.session:
                    self.session.feed(frames)
                i += 1
                await asyncio.sleep(max(0.0, t0 + i * 0.1 - time.time()))
        await asyncio.sleep(3)
        self.log("fake mic: done")

    def _voice_text(self, text: str, final: bool, uid: str) -> None:
        """Transcript from the microphone: drop echoes of our own voice, barge in over it, then handle."""
        if self.tts and (self.tts.speaking or time.time() - self.tts.spoke_at < 1.2):
            spoken = self.tts.speaking or (self.last_said or {}).get("text", "")
            if spoken and self._echo_like(text, spoken):
                log.debug("echo dropped: %s", text)
                return
            if self.tts.speaking and len(_norm_words(text)) >= 1:
                self.log(f"barge-in: \"{text}\"")
                self.tts.interrupt()
                asyncio.ensure_future(self.browser.push({"type": "audio_stop"}))
        if final:
            self.log(f"heard: \"{text}\"")
        else:
            log.debug("hearing: %s", text)
        self.handle_transcript(text, final, uid)

    @staticmethod
    def _echo_like(heard: str, spoken: str) -> bool:
        hw, sw = _norm_words(heard), _norm_words(spoken)
        if not hw:
            return True
        if len(hw) <= 2 and len(sw) > 6:  # a fragment of the long sentence being read out
            return all(w in sw for w in hw)
        h = " ".join(hw)
        return any(difflib.SequenceMatcher(None, h, " ".join(_norm_words(s))).ratio() >= 0.7
                   for s in [spoken, *split_sentences(spoken)])

    def say(self, text: str, kind: str = "info", speak: bool = True, show: bool = True) -> None:
        """Show and speak a line of feedback. Errors and 'not found' go through here too."""
        self.last_said = {"text": text, "kind": kind, "t": time.time()}
        self._remember("a", text, kind)
        if show:
            self.emit({"type": "say", "text": text, "kind": kind})
        if not speak or not self.tts or not self.tts.ready:
            return

        def sink(pcm: bytes, gen: int, last: bool) -> None:
            msg = {"type": "audio", "pcm": base64.b64encode(pcm).decode(), "rate": TTS_RATE, "gen": gen, "last": last}
            self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.browser.push(msg)))
        self.tts.speak(text, sink)

    async def refresh_snapshot(self) -> dict:
        self.snapshot = await self.browser.snapshot()
        self.emit({"type": "snapshot", "snapshot": {k: v for k, v in self.snapshot.items()}})
        self.emit({"type": "suggestions", "items": self.suggestions()})
        return self.snapshot

    def handle_command(self, text: str) -> None:
        """Typed command: treated as a final utterance."""
        self.handle_transcript(text, final=True, utterance_id=f"cmd-{int(time.time() * 1000)}")

    def handle_transcript(self, text: str, final: bool = False, utterance_id: str | None = None) -> None:
        text = clean_transcript(text)
        uid = utterance_id or "live"
        if self.utt is None or self.utt.id != uid:
            self.utt = Utterance(id=uid)
        u = self.utt
        if text == u.text and not (final and not u.final):
            return
        if len(text) < u.consumed:  # recognizer rewrote history: start over
            u.consumed, u.acted = 0, False
        u.text = text
        u.final = final
        u.changed_at = time.time()
        self.emit({"type": "transcript", "text": text, "final": final, "utteranceId": uid})
        if final:
            self._remember("u", text)
        if self._debounce:
            self._debounce.cancel()
        # the front gate runs at once, even while an action is executing: this is the barge-in path
        asyncio.ensure_future(self.gate_now(u))
        delay = 0 if final else DEBOUNCE_MS / 1000
        self._debounce = self._loop.call_later(delay, lambda: asyncio.ensure_future(self.decide_now("debounce")))

    # ---- front gate -------------------------------------------------------------------
    async def gate_now(self, u: Utterance) -> dict | None:
        """Classify the newest fragment; fire reflexes (go back / stop / undo) immediately."""
        async with self._gate_lock:
            live = u.live
            frag = last_fragment(live)
            if not _norm_words(frag) or self.utt is not u or u.acted:
                return u.gate
            if frag == u.gated:
                return u.gate
            # slide control is lexical and instant; bare words ("back", "down") wait for the final transcript
            if self.browser.deck and _norm_words(frag) == _norm_words(live) and (
                    u.final or re.search(r"\b(slides?|next|previous|deck|presentation)\b", frag, re.I)):
                if not self.pending and not self.candidates and await self._step_tool(u, frag) is not None:
                    return None
            g = await self._loop.run_in_executor(None, self.gate.classify, frag)
            if self.utt is not u:
                return None
            self.stats["gate_ms"].append(g["latencyMs"])
            u.gated, u.gate = frag, g
            self.emit({"type": "gate", **g, "utteranceId": u.id})
            if last_fragment(u.live) != frag:  # more words arrived meanwhile
                asyncio.ensure_future(self.gate_now(u))
                return g
            if not u.acted and self._reflex_ok(u, g, frag, live):
                await self._reflex(u, g)
            return g

    def _reflex_ok(self, u: Utterance, g: dict, frag: str, live: str) -> bool:
        """A reflex pre-empts speech only when it is the whole request, a self-correction ("…no wait, go back")
        or a stop. Sequenced "open X, then go back" and remap definitions run through the normal path."""
        if g["family"] != "reflex" or len(_norm_words(frag)) > GATE_MAX_REFLEX_WORDS:
            return False
        if g["confidence"] < (GATE_REFLEX_CONF if u.final else GATE_REFLEX_CONF_INTERIM):
            return False
        if self.candidates and parse_candidate_pick(frag, len(self.candidates["items"])):
            return False
        pre = live[: len(live) - len(frag)]
        if REMAP_CUE_RE.search(pre):
            return False
        if g["kind"] in ("back", "forward") and re.search(r"\b(tabs?|windows?)\b", frag, re.I):
            return False  # "switch to the previous tab" is a tab command, not history
        if g["kind"] in ("back", "forward") and re.search(r"\b(click|press|tap|select|choose|hit|find|type|read|scroll)\b", frag, re.I):
            return False  # "click on history" names a target: that is a command, however back-ish it sounds
        fl = _strip_filler(frag).strip(" .!?,").lower()
        if TOOL_VIDEO_SEEK_RE.match(fl) or TOOL_VIDEO_PAUSE_RE.match(fl) and fl not in ("stop", "pause"):
            return False  # "back ten seconds", "stop the video": media controls, not history / cancel
        return (g["kind"] == "stop" or all(w in FILLERS for w in _norm_words(pre))
                or bool(INTERJECT_RE.search(" " + pre)) or bool(INTERJECT_RE.match(" " + frag)))

    async def _reflex(self, u: Utterance, g: dict) -> None:
        kind = g["kind"]
        u.consumed = len(u.text)  # everything said so far is settled: the reflex replaces it
        u.acted = True
        self.stats["reflexes"] += 1
        self.log(f"reflex: {kind} ({g['confidence']:.2f}, {g['latencyMs']}ms) ← \"{g['fragment']}\"")
        running = self.running if self.running and (self.running.queue or self._busy.locked()) else None
        if kind == "stop" and self.guide is not None:
            if GUIDE_SKIP_RE.match(g["fragment"].strip()):
                skipped = self.guide
                self.guide = None
                self.log(f"guided: skipped {skipped.text}")
                if skipped.queue:
                    self._next_step(skipped)
                else:
                    self._guide_clear("Skipped. That was the last step.")
                return
            self._guide_clear("Stopped the guide.")
            return
        if kind in ("stop", "undo") and (self.pending or self.candidates or running or u.queue):
            # something is pending / running: both mean "drop it"
            self.pending = None
            self.candidates = None
            self._drop_queue(u)
            if running:
                self._drop_queue(running)
            self.emit({"type": "pending", "pending": None})
            self.emit({"type": "candidates", "candidates": []})
            if self._busy.locked():
                try:
                    await (await self.browser.ensure_page()).evaluate("window.stop()")
                except Exception:
                    pass
            await self.browser.overlay("clearCandidates")
            self.say("Stopped.", "info")
            return
        if kind == "stop":  # nothing to stop: acknowledge, and whatever was said before is dropped
            if self.tts and self.tts.speaking:  # "stop" while reading: just go quiet
                self.tts.interrupt()
                await self.browser.push({"type": "audio_stop"})
                await self.browser.overlay("toast", "ok")
                return
            try:  # "stop" with a video playing: pause it
                res = await (await self.browser.ensure_page()).evaluate(VIDEO_JS, {"kind": "pause", "arg": None})
            except Exception:
                res = {}
            if res.get("wasPlaying"):
                self.say("Paused.", "info")
                return
            self.say("Okay.", "info")
            return
        async with self._busy:  # after any in-flight action
            if kind == "undo":
                await self.undo()
                self.say("Undone.", "info")
                return
            action = {"type": REFLEXES[kind], "label": REFLEXES[kind].replace("_", " "), "reflex": True}
            self.emit({"type": "decision", "decision": "act", "action": action, "summary": describe(action),
                       "reasons": [{"name": "gate", "value": f"{kind} ({g['confidence']:.2f})", "threshold": GATE_REFLEX_CONF,
                                    "pass": True, "note": "reflex from the front gate"}],
                       "answers": {}, "latencyMs": g["latencyMs"], "transcript": g["fragment"], "final": u.final})
            result = await self._run_action(action)
            if u.queue:  # "…, then go back, then scroll down"
                self._next_step(u) if result.get("ok") else self._drop_queue(u)

    async def _remap(self, u: Utterance, text: str, g: dict) -> dict | None:
        """Teach / forget a phrasing. Auto-applies on strong evidence, otherwise asks for a yes.
        Returns None when the words do not actually parse as a definition (the ordinary path takes over)."""
        parsed = await self._loop.run_in_executor(None, self.gate.parse_remap, text)
        cue = bool(REMAP_CUE_RE.search(text))
        if not parsed or (not cue and not parsed.get("forget") and
                          (len(_norm_words(parsed["phrase"])) < 1 or all(w in STOPWORDS | FILLERS | {"i", "we"} for w in _norm_words(parsed["phrase"])))):
            if not cue:
                return None
            u.acted = True
            return await self._say(u, "I did not catch what should mean what. Say: when I say X, do Y.", "remap")
        u.acted = True
        if parsed.get("forget") == "*":
            n = self.aliases.clear()
            self.emit({"type": "aliases", "aliases": []})
            return await self._say(u, f"cleared {n} taught phrase{'s' if n != 1 else ''}" if n else "nothing to forget", "remap")
        if "forget" in parsed:
            gone = self.aliases.remove(parsed["forget"] or "")
            self.emit({"type": "aliases", "aliases": self.aliases.as_list()})
            return await self._say(u, f"forgot \"{gone.phrase}\"" if gone else f"no phrase like \"{parsed['forget']}\"", "remap")
        # the command side must be something the decider actually understands
        probe = await self._loop.run_in_executor(None, self.decider.decide, parsed["command"], self.snapshot, None,
                                                 len(self.browser.pages))
        intent = probe["answers"]["intent"]
        if intent["choice"] == "none" or intent["confidence"] < T["intentConfidence"]:
            if not cue:  # Von's remap hunch was wrong: this is an ordinary request
                u.acted = False
                return None
            return await self._say(u, f"\"{parsed['command']}\" is not a command I know, so I did not change anything.", "remap")
        alias = {"phrase": parsed["phrase"], "command": parsed["command"], "intent": intent["choice"]}
        strong = g["remapP"] >= 0.6 or bool(REMAP_CUE_RE.search(text))
        if not strong:
            self.pending = {"type": "remap", **alias, "label": f"make \"{alias['phrase']}\" mean \"{alias['command']}\""}
            self.emit({"type": "pending", "pending": self.pending})
            return await self._say(u, f"Should \"{alias['phrase']}\" mean \"{alias['command']}\"? Say confirm or cancel.", "remap")
        self.aliases.add(alias["phrase"], alias["command"])
        self.emit({"type": "aliases", "aliases": self.aliases.as_list()})
        return await self._say(u, f"ok: \"{alias['phrase']}\" now means {intent['choice'].replace('_', ' ')}", "remap")

    async def _say(self, u: Utterance, text: str, kind: str = "info") -> dict:
        u.done = True
        self.log(text if kind != "read" else text.split("\n", 1)[0][:120] + " …")
        self.say(text, kind)
        return {"decision": kind, "summary": text}

    # ---- LLM escalation -----------------------------------------------------------------
    async def _ask(self, u: Utterance, question: str) -> dict:
        u.acted = True
        if not self.llm:
            return await self._say(u, "No language model is loaded (start with --llm).", "answer")
        if not self.llm.ready:
            if self.llm.error:
                return await self._say(u, "The language model failed to load.", "answer")
            self.say("One moment, loading the language model.", "status")
            await self.browser.overlay("toast", "loading language model…", 6000)
            await self._loop.run_in_executor(None, self.llm.load)
            if not self.llm.ready:
                return await self._say(u, f"The language model failed to load: {self.llm.error}", "answer")
        page = await self.browser.ensure_page()
        try:
            ctx = await page.evaluate(PAGE_TEXT_JS)
        except Exception:
            ctx = {"title": self.snapshot.get("title", ""), "url": page.url, "selection": "", "text": ""}
        body = (ctx.get("selection") or ctx.get("text") or "")[:LLM_PAGE_CHARS]
        messages = [{"role": "system", "content": LLM_SYSTEM},
                    {"role": "user", "content": f"Page title: {ctx.get('title', '')}\nURL: {ctx.get('url', '')}\n\n"
                                                f"Page text:\n{body}\n\nRequest: {question}"}]
        self.stats["asks"] += 1
        t0 = time.time()
        self.emit({"type": "answer", "text": "", "done": False, "question": question})
        await self.browser.overlay("toast", "thinking…", 2000)
        last_emit = [0.0]

        def on_token(text: str) -> None:
            now = time.time()
            if now - last_emit[0] > 0.12:
                last_emit[0] = now
                self._loop.call_soon_threadsafe(self.emit, {"type": "answer", "text": text, "done": False, "question": question})

        try:
            answer = await self._loop.run_in_executor(None, lambda: self.llm.generate(messages, 260, on_token))
        except Exception as e:
            return await self._say(u, f"language model failed: {type(e).__name__}", "answer")
        ms = int((time.time() - t0) * 1000)
        u.done = True
        self.log(f"answer ({ms}ms, {self.llm.stats['tps']} tok/s): {answer[:160]}")
        self.emit({"type": "answer", "text": answer, "done": True, "question": question, "ms": ms})
        self.say(answer, "answer", show=False)  # the panel already shows the streamed answer
        return {"decision": "answer", "summary": answer}

    async def _llm_pick(self, text: str, cands: list[dict]) -> dict | None:
        """Ambiguous target: let the LLM pick among the numbered labels, then let Von veto the pick."""
        if not self.llm or not self.llm.ready or len(cands) < 2:
            return None
        menu = "\n".join(f"{i + 1}. {c['label']}" for i, c in enumerate(cands))
        messages = [{"role": "system", "content": "You pick the on-page element a voice command refers to. Reply with only the number, or 0 if none fits."},
                    {"role": "user", "content": f"Command: {text}\nPage: {self.snapshot.get('title', '')}\nElements:\n{menu}"}]
        try:
            reply = await self._loop.run_in_executor(None, lambda: self.llm.generate(messages, 6))
        except Exception:
            return None
        m = re.search(r"\d+", reply or "")
        n = int(m.group()) if m else 0
        if not 1 <= n <= len(cands):
            return None
        pick = cands[n - 1]
        verdict = await self._loop.run_in_executor(None, self.decider.verify_target, text, pick["label"])
        self.log(f"llm pick {n} \"{pick['label']}\" → von {verdict:.2f}")
        return pick if verdict >= 0.6 else None

    # ---- slides: the user's PDF deck, driven by voice ("next", "last slide", "slide seven", "open my slides")
    def _slide_op(self, t: str) -> tuple[str, int] | None:
        """(op, n) for a slide phrase, else None. Navigation words only count while the deck is on screen."""
        deck = self.browser.deck
        m = SLIDES_OPEN_RE.match(t)
        if m:
            return ("open", 0) if not m.group("path") else ("load", 0)
        if not deck:
            return None
        if SLIDES_EXIT_RE.match(t) and self.browser.on_slides():
            return ("exit", 0)
        m = SLIDES_GOTO_RE.match(t)
        if m:
            w = m.group("n").lower()
            n = int(w) if w.isdigit() else NUMBER_WORDS.get(w, NUMBER_HOMOPHONES.get(w, 0))
            return ("goto", deck.count if n == -1 else n) if n else None
        if SLIDES_WHICH_RE.match(t):
            return ("which", 0)
        if not self.browser.on_slides():
            return None
        for rx, op in ((SLIDES_FIRST_RE, "first"), (SLIDES_END_RE, "end"), (SLIDES_NEXT_RE, "next"), (SLIDES_PREV_RE, "prev")):
            if rx.match(t):
                return (op, 0)
        return None

    async def _slides(self, u: Utterance, text: str) -> dict | None:
        t = _strip_filler(text).replace("\u2019", "'").strip(" .!?,").lower()
        op = self._slide_op(t)
        if not op:
            return None
        u.acted = True
        u.consumed = len(u.text)
        kind, n = op
        deck = self.browser.deck
        t0 = time.time()
        if kind == "load":
            path = Path(SLIDES_OPEN_RE.match(t).group("path")).expanduser()
            if not path.exists():
                return await self._say(u, f"I can't find a PDF at {path}.", "error")
            deck = SlideDeck(str(path))
            await self._loop.run_in_executor(None, deck.load)
            if not deck.ready:
                return await self._say(u, f"I couldn't read that PDF: {deck.error}", "error")
            self.browser.deck = deck
            kind = "open"
        if kind == "open":
            if deck and not deck.loaded.is_set():
                self.emit({"type": "toast", "text": "rendering slides…"})
                await deck.wait()
            if not deck or not deck.ready:
                hint = (f"I couldn't read {Path(deck.path).name}: {deck.error}" if deck else
                        f"Start me with dash dash slides and the PDF path, or put the PDF at {DEFAULT_SLIDES_PATH}.")
                return await self._say(u, f"No slides are loaded. {hint}", "error")
            page = await self.browser.ensure_page()
            if not self.browser.on_slides():
                self.history.append({"action": {"type": "navigate_url", "url": SLIDES_URL, "label": "slides"},
                                     "before_url": page.url, "result": {"ok": True}, "t": time.time()})
                await page.goto(f"{SLIDES_URL}#{deck.current}", wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await self.refresh_snapshot()
            self.log(f"slides: open {deck.count} slides at {deck.current} ({int((time.time() - t0) * 1000)}ms)")
            self.emit({"type": "slides", "n": deck.current, "count": deck.count, "title": deck.title})
            return await self._say(u, f"{deck.title}, {deck.count} slides. Say next, last slide, or slide and a number.", "info")
        page = await self.browser.ensure_page()
        if kind == "exit":
            await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await self.refresh_snapshot()
            return await self._say(u, "Closed the slides.", "info")
        if not self.browser.on_slides():
            await page.goto(f"{SLIDES_URL}#{deck.current}", wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        js = {"next": "window.__slides.next()", "prev": "window.__slides.prev()", "first": "window.__slides.go(1)",
              "end": "window.__slides.go(window.__slides.count)", "goto": f"window.__slides.go({n})",
              "which": "({n: window.__slides.n, count: window.__slides.count})"}[kind]
        try:
            before = await page.evaluate("window.__slides.n")
            state = await page.evaluate(js)
        except Exception as e:
            return await self._say(u, f"The slides page did not respond ({type(e).__name__}).", "error")
        cur, count = state["n"], state["count"]
        deck.current = cur
        ms = int((time.time() - t0) * 1000)
        self.stats["actions"] += 1
        self.stats["action_ms"].append(ms)
        self.log(f"slides: {kind} -> {cur}/{count} ({ms}ms)")
        self.emit({"type": "slides", "n": cur, "count": count, "title": deck.title})
        self.snapshot["url"] = page.url
        self.emit({"type": "suggestions", "items": self.suggestions()})
        if kind == "which":
            return await self._say(u, f"Slide {cur} of {count}.", "info")
        if kind == "goto" and not 1 <= n <= count:
            return await self._say(u, f"There are only {count} slides.", "error")
        if cur == before and kind in ("next", "prev"):
            return await self._say(u, "That's the last slide." if kind == "next" else "This is the first slide.", "info")
        await self.browser.overlay("toast", f"slide {cur} / {count}", 1200)
        return {"decision": "slides", "summary": f"slide {cur}/{count}", "latencyMs": ms}

    @staticmethod
    def _wait_secs(t: str) -> float | None:
        """'wait 2 seconds' / 'hold on a moment' / 'wait ten secs' -> seconds; bare 'wait' is not a pause step."""
        m = TOOL_WAIT_RE.match(t.lower())
        if not m or not (m.group(1) or m.group(2)):
            return None
        n = m.group(1)
        if n is None:
            secs = 1.0
        elif n.replace(".", "", 1).isdigit():
            secs = float(n)
        elif n in NUMBER_WORDS:
            secs = float(NUMBER_WORDS[n])
        elif n in ("a", "an", "one"):
            secs = 1.0
        else:
            return None
        if m.group(2) and m.group(2).startswith("min"):
            secs *= 60
        return min(secs, 120.0)

    def _known_step(self, fragment: str) -> bool:
        """Built-in tool phrases count as steps of a chain without asking Von ("…, slide four, then next")."""
        t = _strip_filler(fragment).replace("\u2019", "'").strip(" .!?,").lower()
        rxs = [TOOL_READ_RE, TOOL_FIND_RE, TOOL_ZOOM_RE, TOOL_WHERE_RE, TOOL_HEADINGS_RE, TOOL_HELP_RE, TOOL_QUIET_RE, TAB_STEP_RE]
        if self._video_op(t) is not None and re.search(r"video|caption|subtitle|pause|speed|resume", t):
            return True
        if self._wait_secs(t) is not None or any(self.aliases.similarity(t, a.phrase) >= 0.75 for a in self.aliases.near(t)):
            return True
        if self.browser.deck:
            rxs += [SLIDES_OPEN_RE, SLIDES_EXIT_RE, SLIDES_NEXT_RE, SLIDES_PREV_RE, SLIDES_FIRST_RE, SLIDES_END_RE,
                    SLIDES_GOTO_RE, SLIDES_WHICH_RE]
        return any(rx.match(t) for rx in rxs)

    async def _step_tool(self, u: Utterance, text: str) -> dict | None:
        """A built-in tool (slides, read, zoom, where am I ...) as one step of an utterance; advances the chain."""
        tool = await (self._tool(u, text) if u.final else self._slides(u, text))
        if tool is None:
            return None
        self.emit({"type": "decision", "decision": "act", "summary": tool.get("summary", "")[:120], "reasons": [],
                   "answers": {}, "latencyMs": 0, "transcript": text, "final": True})
        if u.queue:
            self._next_step(u) if tool.get("decision") != "error" else self._drop_queue(u)
        elif u.guided and u.step_no > 1:
            self.say(f"Done. That was step {u.step_no} of {u.step_total}, the last one.", "guide")
        return tool

    # ---- built-in page tools (no site knowledge): read aloud, find, zoom, where am I, what's here, help
    async def _tool(self, u: Utterance, text: str) -> dict | None:
        r = await self._slides(u, text)
        if r is not None:
            return r
        t = _strip_filler(text).replace("\u2019", "'").strip(" .!?,")
        page = await self.browser.ensure_page()
        if TOOL_QUIET_RE.match(t):
            has_video = False
            if t.lower() == "mute":  # on a page with a video, "mute" is the player's mute, not "be quiet"
                try:
                    has_video = await page.evaluate('[...document.querySelectorAll("video")].some(v => v.getClientRects().length > 0)')
                except Exception:
                    has_video = False
            if has_video:
                r = await self._video(u, t, page)
                if r is not None:
                    return r
            u.acted = True
            self.tts and self.tts.interrupt()
            await self.browser.push({"type": "audio_stop"})
            return {"decision": "tool", "summary": "quiet"}
        if TOOL_HELP_RE.match(t):
            u.acted = True
            return await self._say(u, HELP_TEXT, "answer")
        secs = self._wait_secs(t)
        if secs is not None:
            u.acted = True
            await self.browser.overlay("toast", f"waiting {secs:g}s")
            t0, had_queue = time.time(), bool(u.queue)
            while time.time() - t0 < secs:  # a reflex ("stop") drops the queue; a new utterance supersedes us
                if self.utt is not u or (had_queue and not u.queue):
                    break
                await asyncio.sleep(0.1)
            return {"decision": "tool", "summary": f"waited {secs:g}s"}
        r = await self._video(u, t, page)
        if r is not None:
            return r
        if TOOL_WHERE_RE.match(t):
            u.acted = True
            host = re.sub(r"^www\.", "", urlparse(page.url).hostname or "") or "a blank page"
            title = self.snapshot.get("title") or ""
            i = self.browser.pages.index(page) + 1 if page in self.browser.pages else 1
            where = f"You're on {title}, at {host}." if title else f"You're at {host}."
            return await self._say(u, f"{where} Tab {i} of {len(self.browser.pages)}.", "answer")
        if TOOL_READ_RE.match(t):
            u.acted = True
            try:
                ctx = await page.evaluate(PAGE_TEXT_JS)
            except Exception:
                ctx = {}
            body = (ctx.get("selection") or ctx.get("text") or "").strip()
            if not body:
                return await self._say(u, "There is no readable text on this page.", "error")
            self.log(f"reading {len(body)} chars aloud")
            return await self._say(u, body[:READ_CHARS], "read")
        if TOOL_HEADINGS_RE.match(t):
            u.acted = True
            try:
                heads = await page.evaluate(HEADINGS_JS)
            except Exception:
                heads = []
            if not heads:
                labels = [e["text"] for e in self.snapshot.get("elements", []) if e.get("role") in ("link", "button") and not e.get("below_fold") and e.get("text")]
                heads = list(dict.fromkeys(labels))[:8]
                if not heads:
                    return await self._say(u, "I don't see headings or links on this page.", "error")
                return await self._say(u, "You can click: " + ", ".join(heads) + ".", "answer")
            return await self._say(u, "The sections are: " + ", ".join(heads[:8]) + ".", "answer")
        m = TOOL_ZOOM_RE.match(t)
        if m:
            u.acted = True
            if re.search(r"reset|normal|default", t, re.I):
                self.zoom = 1.0
            elif re.search(r"out|smaller|shrink|reduce|decrease", t, re.I):
                self.zoom = max(0.5, round(self.zoom - 0.25, 2))
            else:
                self.zoom = min(3.0, round(self.zoom + 0.25, 2))
            await self.browser.push({"type": "zoom", "level": self.zoom})
            self.log(f"zoom {self.zoom}")
            return await self._say(u, f"Zoom {int(self.zoom * 100)} percent.", "info")
        m = TOOL_FIND_RE.match(t)
        if m:
            needle = m.group(1).strip(" \"'“”")
            if needle and len(needle.split()) <= 8:
                u.acted = True
                try:
                    n = await page.evaluate("(t) => window.__vb ? window.__vb.find(t) : 0", needle)
                except Exception:
                    n = 0
                if not n:
                    return await self._say(u, f"I don't see \"{needle}\" on this page.", "error")
                return await self._say(u, f"Found {n} match{'es' if n != 1 else ''} for \"{needle}\"; the first one is highlighted.", "info")
        return None

    # ---- media: pause/play, captions, speed, mute, seek on whatever <video> the page has (YouTube, Vimeo, news sites ...)
    def _video_op(self, t: str) -> tuple[str, object, str] | None:
        """Phrase -> (kind, arg, spoken confirmation) or None. Site-agnostic: it drives the HTMLMediaElement."""
        tl = t.lower()
        if TOOL_VIDEO_PAUSE_RE.match(tl):
            return "pause", None, "Paused."
        if TOOL_VIDEO_PLAY_RE.match(tl):
            return "play", None, "Playing."
        m = TOOL_VIDEO_CC_RE.match(tl)
        if m:
            want = "off" if re.search(r"\b(off|hide|disable|remove)\b", tl) else "on" if re.search(r"\b(on|show|enable|add|want|need|can i|could you)\b", tl) else "toggle"
            return "cc", want, {"on": "Captions on.", "off": "Captions off.", "toggle": "Toggled captions."}[want]
        m = TOOL_VIDEO_MUTE_RE.match(tl)
        if m:
            gd = m.groupdict()
            muted = not (gd.get("un") or tl == "unmute" or (gd.get("off") or gd.get("off2") or "") == "on")
            return "mute", muted, "Muted." if muted else "Sound on."
        m = TOOL_VIDEO_SEEK_RE.match(tl)
        if m:
            n = m.group("n").lower()
            secs = {"ten": 10, "five": 5, "thirty": 30, "fifteen": 15, "twenty": 20, "sixty": 60, "a minute": 60, "one minute": 60,
                    "two minutes": 120}.get(n)
            if secs is None:
                secs = int(n) * (60 if re.search(r"\bmin", tl) else 1)
            if m.group("back"):
                secs = -secs
            return "seek", secs, f"{'Back' if secs < 0 else 'Ahead'} {abs(secs)} seconds."
        sp = parse_speed(tl)
        if sp is not None:
            return "speed", sp, ""
        return None

    async def _video(self, u: Utterance, t: str, page) -> dict | None:
        op = self._video_op(t)
        if op is None:
            return None
        kind, arg, said = op
        explicit = bool(re.search(r"video|movie|clip|player|playback|caption|subtitle|\bcc\b|pause|resume|unpause|speed|mute|rewind|\bplay\b", t, re.I))
        res: dict = {}
        tries = 5 if kind == "cc" else 3  # a player that is still booting has no <video>/controls (or reports captions unavailable) yet
        for attempt in range(tries):
            for frame in [page.main_frame] + [f for f in page.frames if f is not page.main_frame]:
                try:
                    res = await frame.evaluate(VIDEO_JS, {"kind": kind, "arg": arg})
                except Exception:
                    res = {}
                if res.get("ok") or res.get("video"):
                    break
            if res.get("ok") or attempt == 2:
                break
            await asyncio.sleep(0.6)
        if not res.get("video") and not res.get("ok"):
            if not explicit:
                return None  # "faster" on a page without a video: let Von decide what it meant
            u.acted = True
            return await self._say(u, "I don't see a video on this page.", "error")
        u.acted = True
        if not res.get("ok"):
            msg = "This video has no captions." if res.get("reason") == "no captions" else "I couldn't do that with this video."
            return await self._say(u, msg, "error")
        if kind == "speed":
            rate = res.get("rate")
            said = f"Speed {rate:g}x." if isinstance(rate, (int, float)) else "Speed changed."
        self.log(f"video: {kind} {arg!r} -> {res}")
        await self.browser.overlay("toast", said.rstrip(".").lower(), 1200)
        return await self._say(u, said, "info")

    # ---- main decision path -------------------------------------------------------------
    async def decide_now(self, trigger: str = "manual") -> dict | None:
        u = self.utt
        if not u or not u.live or u.acted:
            return None
        if u.final:  # the reflex path has priority over the decision path: wait for its verdict
            await self.gate_now(u)
            if u.acted or self.utt is not u or not u.live:
                return None
        elif self.session and self.voice_on:
            # local Whisper: interim text is for the gate (reflexes, barge-in) only; the VAD delivers the
            # final sentence ~0.7 s after the speaker stops, so acting on half a sentence buys nothing
            return None
        if self._busy.locked():
            log.debug("decide_now(%s): busy, deferring %r", trigger, u.live)
            self._debounce = self._loop.call_later(0.25, lambda: asyncio.ensure_future(self.decide_now("busy-retry")))
            return None
        async with self._busy:
            text = u.live
            final = u.final
            silent_ms = int((time.time() - u.changed_at) * 1000)

            if self.pending and self.pending.get("type") == "remap":
                g0 = u.gate if u.gated == last_fragment(text) else await self._loop.run_in_executor(None, self.gate.classify, text)
                yes = parse_yes_no(text)
                if yes is True:
                    self.aliases.add(self.pending["phrase"], self.pending["command"])
                    self.emit({"type": "aliases", "aliases": self.aliases.as_list()})
                    msg = f"ok: \"{self.pending['phrase']}\" now means {self.pending['intent'].replace('_', ' ')}"
                    self.pending = None
                    self.emit({"type": "pending", "pending": None})
                    u.acted = True
                    return await self._say(u, msg, "remap")
                if yes is False or (g0["family"] == "reflex" and g0["kind"] in ("stop", "undo")):
                    self.pending = None
                    self.emit({"type": "pending", "pending": None})
                    u.acted = True
                    return await self._say(u, "ok, nothing changed", "remap")
                if not final:
                    return None
                self.pending = None  # the user moved on to something else
                self.emit({"type": "pending", "pending": None})

            if self.pending and self.pending.get("type") != "remap" and parse_yes_no(text) is not None:
                # a plain yes/no to "say confirm or cancel" is decided lexically, never left to the model
                action, yes = self.pending, parse_yes_no(text)
                self.pending = None
                u.acted = True
                self.emit({"type": "pending", "pending": None})
                if yes is False:
                    self.say("Cancelled.", "info")
                    return {"decision": "cancel", "summary": "cancelled"}
                self.emit({"type": "decision", "decision": "act", "action": action, "summary": f"confirmed: {describe(action)}",
                           "reasons": [], "answers": {}, "latencyMs": 0, "transcript": text, "final": final})
                await self._run_action({**action, "confirmed": True})
                return {"decision": "act", "action": action}

            if final and self.guide is not None and not self.pending and not self.candidates:
                if GUIDE_GO_RE.match(text.strip()):
                    return self._guide_go(u)
                if GUIDE_SKIP_RE.match(text.strip()):
                    skipped = self.guide
                    u.acted = True
                    self.log(f"guided: skipped {skipped.text}")
                    if skipped.queue:
                        self._next_step(skipped)
                    else:
                        self._guide_clear("Skipped. That was the last step.")
                    return {"decision": "ignore", "summary": "skipped step"}
                if re.match(r"^(?:stop|cancel|never mind|nevermind|quit|that's enough|thats enough)(?: (?:the )?(?:guide|guided mode|steps))?[.!]?$", text.strip(), re.I):
                    u.acted = True
                    self._guide_clear("Stopped the guide.")
                    return {"decision": "ignore", "summary": "guide stopped"}
                self._guide_clear()  # anything else: the user moved on, the guide is abandoned

            if final and not self.pending and not self.candidates and parse_yes_no(text) is not None:
                u.done = True
                self.log(f"ignored (nothing to confirm): {text}")
                self.emit({"type": "decision", "decision": "ignore", "summary": "nothing to confirm", "reasons": [], "answers": {},
                           "latencyMs": 0, "transcript": text, "final": True})
                return {"decision": "ignore", "summary": "nothing to confirm"}

            chain = bool(GUIDE_CUE_RE.match(text)) or len(STEP_SEP.split(clean_transcript(text))) > 1
            if final and not self.pending and not self.candidates and not chain:
                tool = await self._step_tool(u, text)
                if tool is not None:
                    return tool

            # front gate verdict for the whole live text (cached when the gate already saw this fragment)
            g = u.gate if (u.gated == last_fragment(text) and u.gate) else None
            if g is None or g["fragment"] != text:
                if final or silent_ms >= PAYLOAD_SILENCE_MS:
                    g = await self._loop.run_in_executor(None, self.gate.classify, text)
                    self.stats["gate_ms"].append(g["latencyMs"])
                    if self.utt is not u or u.acted or u.live != text:
                        return None
            if g and g["fragment"] == text and not self.pending and not self.candidates and not GUIDE_CUE_RE.match(text):
                if g["family"] == "remap" and final and (REMAP_CUE_RE.search(text) or
                                                         (g["confidence"] >= 0.6 and len(STEP_SEP.split(text)) == 1)):
                    r = await self._remap(u, text, g)
                    if r is not None:
                        return r
                if g["family"] == "ask" and g["confidence"] >= GATE_ASK_P:  # questions need the whole sentence
                    return await self._ask(u, text) if final else None
                if g["family"] == "chatter" and g["confidence"] >= 0.7 and final:
                    u.done = True
                    self.log(f"ignored (gate: chatter {g['confidence']:.2f}): {text}")
                    return {"decision": "ignore", "summary": "not talking to the browser"}
                if g.get("alias"):
                    al = g["alias"]
                    text = self.aliases.apply(text, Alias(al["phrase"], al["command"]))
                    self.log(f"alias: \"{al['phrase']}\" → \"{al['command']}\"")

            # deterministic number pick while candidates are on screen
            if self.candidates:
                n = parse_candidate_pick(text, len(self.candidates["items"]))
                if n:
                    c = self.candidates["items"][n - 1]
                    pi = self.candidates["pendingIntent"]
                    action = {"type": pi["type"], "targetId": c["id"], "text": pi.get("text"), "label": c["label"]}
                    self.candidates = None
                    self.emit({"type": "candidates", "candidates": []})
                    u.acted = True
                    self.emit({"type": "decision", "decision": "act", "summary": f"picked {n}: {describe(action)}",
                               "reasons": [{"name": "candidate_pick", "value": n, "threshold": "-", "pass": True, "note": "number spoken"}],
                               "answers": {}, "latencyMs": 0, "transcript": text})
                    await self._run_action(action)
                    return {"decision": "act", "action": action}

            spoken = u.live
            if final and not u.split and not self.pending and not self.candidates:
                u.split = True
                m_guide = GUIDE_CUE_RE.match(text)
                if m_guide:
                    text = text[m_guide.end():].strip(" ,.") or text
                    u.guided = True
                steps = await self._loop.run_in_executor(None, self.decider.split_steps, text, self._known_step)
                if len(steps) > 1 or u.guided:
                    steps = [self._alias_step(s) for s in steps]
                    text = steps[0]
                    u.text, u.consumed = text, 0
                    u.queue = steps[1:]
                    u.step_no, u.step_total = 1, len(steps)
                    self.log(f"{len(steps)} steps{' (guided)' if u.guided else ''}: " + " | ".join(steps))
                    self.emit({"type": "steps", "steps": steps, "utteranceId": u.id, "guided": u.guided})
                    spoken = text
                    if u.guided:
                        self.say(f"Step 1 of {len(steps)}: {text}.", "guide")
                if chain and not self.pending and not self.candidates:
                    tool = await self._step_tool(u, text)
                    if tool is not None:
                        return tool
            res = await self._loop.run_in_executor(None, self.decider.decide, text, self.snapshot, self.pending, len(self.browser.pages))
            self.stats["decisions"] += 1
            self.stats["decision_ms"].append(res["latencyMs"])
            if self.utt is not u or u.acted:
                return None
            if u.live != spoken:  # newer words arrived while we were thinking: decide again
                self._loop.call_soon(lambda: asyncio.ensure_future(self.decide_now("rerun")))
                return None
            policy = evaluate_policy(res["answers"], res["candidates"], self.snapshot, silent_ms=silent_ms, is_final=final,
                                     pending=self.pending)
            self.emit({"type": "decision", **policy, "answers": res["answers"], "candidates_spans": res["candidates"],
                       "latencyMs": res["latencyMs"], "transcript": text, "final": final, "trigger": trigger})
            d = policy["decision"]
            if d == "wait":
                if not final:
                    retry = policy.get("retryInMs") or (SILENCE_COMPLETE_MS - silent_ms)
                    if retry > 0:
                        self._debounce = self._loop.call_later(max(0.05, retry / 1000), lambda: asyncio.ensure_future(self.decide_now("silence")))
                elif not u.resnapped and self.history and time.time() - self.history[-1]["t"] < 6:
                    # the page may still be rendering after the last action: let it settle, look again
                    u.resnapped = True
                    await _settle(await self.browser.ensure_page(), 4000)
                    await self.refresh_snapshot()
                    self._loop.call_soon(lambda: asyncio.ensure_future(self.decide_now("resnap")))
                elif len(_norm_words(text)) <= 2 and not lexical_site(text) and not u.queue:
                    u.done = True  # a stray word or two that leads nowhere: not a command, no nagging
                    self.log(f"ignored ({policy['summary']}): {text}")
                    self.say(f"I didn't catch that: \"{text}\".", "info", speak=False)
                else:
                    u.done = True
                    self.log(f"wait: {policy['summary']}", "warn")
                    self._drop_queue(u)
                    self.say(NOT_FOUND.get(policy["summary"], f"Sorry, {policy['summary']}."), "error")
                return policy
            if d == "ignore":
                if final:
                    u.done = True
                    self.log(f"ignored: {text}")
                    self._drop_queue(u)
                return policy
            if d == "cancel":
                self.pending = None
                u.acted = True
                self.emit({"type": "pending", "pending": None})
                self.say("Cancelled.", "info")
                return policy
            if d == "confirm":
                self.pending = policy["action"]
                u.acted = True
                self._drop_queue(u)
                self.emit({"type": "pending", "pending": self.pending})
                await self.browser.overlay("highlight", policy["action"].get("targetId"), 3000)
                self.say(f"This looks like it has side effects: {describe(policy['action'])}. Say confirm or cancel.", "ask")
                return policy
            if d == "disambiguate":
                items = [{**c, "n": i + 1} for i, c in enumerate(policy["candidates"])]
                pi = policy["pendingIntent"]
                pick = await self._llm_pick(text, items) if final else None
                if pick and self.utt is u and not u.acted:
                    u.acted = True
                    action = {"type": pi["type"], "targetId": pick["id"], "text": pi.get("text"), "label": pick["label"]}
                    self.emit({"type": "decision", "decision": "act", "action": action, "summary": f"llm+von picked: {describe(action)}",
                               "reasons": policy["reasons"] + [{"name": "llm_pick", "value": pick["n"], "threshold": "von ≥ 0.6",
                                                                "pass": True, "note": pick["label"]}],
                               "answers": res["answers"], "latencyMs": res["latencyMs"], "transcript": text, "final": final})
                    result = await self._run_action(action)
                    if u.queue:
                        self._next_step(u) if result.get("ok") else self._drop_queue(u)
                    return {"decision": "act", "action": action}
                if final and ordinal_in(text) is None and best_lexical_element(text, self.snapshot.get("elements") or [])[1] == 0:
                    # nothing on the page shares a word with what was asked: let Von veto the guesses
                    verdicts = [await self._loop.run_in_executor(None, self.decider.verify_target, text, c["label"]) for c in items]
                    items = [c for c, v in zip(items, verdicts) if v >= 0.5]
                    if not items:
                        u.done = True
                        self._drop_queue(u)
                        self.log(f"no element matches: {text}", "warn")
                        self.say(NOT_FOUND["no matching element on this page"], "error")
                        return {"decision": "wait", "summary": "no matching element on this page"}
                    items = [{**c, "n": i + 1} for i, c in enumerate(items)]
                self.candidates = {"items": items, "pendingIntent": pi}
                u.acted = True
                self._drop_queue(u)
                self.emit({"type": "candidates", "candidates": items})
                self.say("Which one? " + ", or ".join(f"{c['n']}, {c['label'][:40]}" for c in items) + ".", "ask")
                await self.browser.overlay("candidates", items)
                return policy
            if d == "act":
                a = policy["action"]
                said_w = {w[:4] for w in _norm_words(text)}
                act_w = {w[:4] for w in _norm_words(f"{a.get('type', '')} {a.get('label', '')}".replace("_", " "))}
                if len(said_w) <= 2 and not (said_w & act_w) and not lexical_site(text) and not u.queue and \
                        not a.get("targetId") and (g is None or (g["confidence"] < 0.6 and not g.get("alias"))) and \
                        not self.pending and not self.candidates:
                    if not final or g is None:
                        return None  # too little said, too unsure: let the final transcript decide
                    u.done = True  # "zap": the gate is unsure and the decider's pick is a guess; do nothing
                    self.log(f"ignored (gate {g['family']} {g['confidence']:.2f}, would {describe(a)}): {text}")
                    self.say(f"I didn't catch that: \"{text}\".", "info", speak=False)
                    return {"decision": "ignore", "summary": "unrecognised fragment"}
                if final and a.get("type") == "click_element" and a.get("targetId") and ordinal_in(text) is None and \
                        not self.candidates and best_lexical_element(text, self.snapshot.get("elements") or [])[1] == 0:
                    # Von picked an element that shares no word with the request: make it defend the choice
                    v = await self._loop.run_in_executor(None, self.decider.verify_target, text, a.get("label", ""))
                    if self.utt is not u or u.acted:
                        return None
                    if v < 0.5:
                        u.done = True
                        self._drop_queue(u)
                        self.log(f"vetoed {describe(a)} (von {v:.2f}) for: {text}", "warn")
                        self.say(NOT_FOUND["no matching element on this page"], "error")
                        return {"decision": "wait", "summary": "no matching element on this page"}
                u.acted = True
                if self.pending and policy["action"].get("confirmed"):
                    self.pending = None
                    self.emit({"type": "pending", "pending": None})
                self.candidates = None
                result = await self._run_action(policy["action"])
                if u.queue:
                    if result.get("ok"):
                        self._next_step(u)
                    else:
                        self._drop_queue(u)
                elif u.guided and result.get("ok"):
                    self.say(f"Done. That was step {u.step_no} of {u.step_total}, the last one.", "guide")
                return policy
        return None

    def _alias_step(self, step: str) -> str:
        for a in self.aliases.near(step):
            if set(_norm_words(a.phrase)) <= set(_norm_words(step)) or self.aliases.similarity(step, a.phrase) >= 0.8:
                return self.aliases.apply(step, a)
        return step

    def _drop_queue(self, u: Utterance) -> None:
        if u.queue:
            self.log(f"stopped before: {' | '.join(u.queue)}", "warn")
            u.queue = []

    def _next_step(self, u: Utterance) -> None:
        if self.utt is not u and self.utt and self.utt.text and not self.utt.acted and not self.utt.done:
            self.log(f"new request \"{self.utt.text}\" takes over", "warn")
            self._drop_queue(u)  # the user has started saying something else: that wins
            return
        nxt = Utterance(id=f"{u.id}+", text=u.queue[0], final=True, split=True, queue=u.queue[1:],
                        guided=u.guided, step_no=u.step_no + 1, step_total=u.step_total)
        u.queue = []
        if u.guided:
            self.guide = nxt
            self.emit({"type": "guide", "step": nxt.step_no, "total": nxt.step_total, "text": nxt.text})
            self.say(f"Step {nxt.step_no} of {nxt.step_total}: {nxt.text}. Say next, skip, or stop.", "guide")
            return
        self.utt = nxt
        self.emit({"type": "transcript", "text": nxt.text, "final": True, "utteranceId": nxt.id})
        self._loop.call_soon(lambda: asyncio.ensure_future(self.decide_now("step")))

    def _guide_clear(self, spoken: str | None = None) -> None:
        self.guide = None
        self.emit({"type": "guide", "step": 0, "total": 0, "text": ""})
        if spoken:
            self.say(spoken, "guide")

    def _guide_go(self, u: Utterance) -> dict:
        """The user said "next" while a guided step is waiting: run it as the live utterance."""
        nxt = self.guide
        self.guide = None
        self.emit({"type": "guide", "step": 0, "total": 0, "text": ""})
        u.acted = True
        self.utt = nxt
        self.emit({"type": "transcript", "text": nxt.text, "final": True, "utteranceId": nxt.id})
        self._loop.call_soon(lambda: asyncio.ensure_future(self.decide_now("guide")))
        return {"decision": "act", "summary": f"guided step {nxt.step_no}: {nxt.text}"}

    async def _run_action(self, action: dict) -> dict:
        t0 = time.time()
        self.running = self.utt
        before_url = self.snapshot.get("url")
        try:
            result = await self.execute_fn(action, self.browser)
        except Exception as e:
            result = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:160]}"}
        ms = int((time.time() - t0) * 1000)
        self.stats["actions"] += 1
        self.stats["action_ms"].append(ms)
        self.history.append({"action": action, "before_url": before_url, "result": result, "t": time.time()})
        self.emit({"type": "action", "action": action, "result": result, "ms": ms})
        if result.get("ok"):
            self._remember("a", describe(action), "ok")
        self.log(f"{'✓' if result.get('ok') else '✗'} {describe(action)} ({ms}ms) {result.get('detail', '')}",
                 "info" if result.get("ok") else "error")
        if not result.get("ok"):
            self.say(f"Sorry, I could not {describe(action)}. {result.get('spoken', '')}".strip(), "error")
        await self.refresh_snapshot()
        return result

    async def undo(self) -> None:
        if not self.history:
            return
        last = self.history[-1]
        t = last["action"]["type"]
        page = await self.browser.ensure_page()
        if t in ("navigate_url", "click_element", "press_enter") or (t == "type_into_field" and last["action"].get("submit")):
            try:
                await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            except Exception:
                pass
        elif t == "type_into_field":
            try:
                await page.locator(f'[data-vb-id="{last["action"]["targetId"]}"]').first.fill("")
            except Exception:
                pass
        elif t == "scroll_down":
            await page.evaluate("() => window.scrollBy(0, -window.innerHeight * 0.85)")
        elif t == "scroll_up":
            await page.evaluate("() => window.scrollBy(0, window.innerHeight * 0.85)")
        elif t == "open_new_tab":
            await page.close()
        self.history.pop()
        self.log("undo")
        await self.refresh_snapshot()

    def ui_state(self) -> dict:
        dm = self.stats["decision_ms"]
        am = self.stats["action_ms"]
        gm = self.stats["gate_ms"]
        return {
            "type": "state", "snapshot": self.snapshot, "pending": self.pending,
            "candidates": self.candidates["items"] if self.candidates else [],
            "tabs": self.browser.tabs(), "model": self.decider.model.info, "llm": self.llm.info if self.llm else None,
            "aliases": self.aliases.as_list(),
            "stats": {"decisions": self.stats["decisions"], "actions": self.stats["actions"], "reflexes": self.stats["reflexes"],
                      "asks": self.stats["asks"],
                      "avg_decision_ms": int(sum(dm) / len(dm)) if dm else 0, "avg_action_ms": int(sum(am) / len(am)) if am else 0,
                      "avg_gate_ms": int(sum(gm) / len(gm)) if gm else 0},
        }

    async def close(self) -> None:
        await self.browser.close()


# --------------------------------------------------------------------------------------
# Server: localhost UI (aiohttp) with Chrome Web Speech API
# --------------------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Whippet (debug console)</title>
<style>
:root{--bg:#0b1020;--card:#121a2e;--fg:#e5e7eb;--muted:#8b93a7;--acc:#f59e0b;--ok:#22c55e;--bad:#ef4444;--blue:#3b82f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,Segoe UI,Inter,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:12px 18px;border-bottom:1px solid #1f2a44}
header h1{font-size:16px;margin:0}header .pill{background:var(--card);padding:3px 10px;border-radius:999px;color:var(--muted);font-size:12px}
main{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;padding:14px 18px}
.card{background:var(--card);border-radius:12px;padding:14px}.card h2{margin:0 0 8px;font-size:12px;letter-spacing:.06em;color:var(--muted);text-transform:uppercase}
#transcript{font-size:22px;min-height:48px}#transcript .interim{color:var(--muted);font-weight:400}
button{background:var(--blue);color:#fff;border:0;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer}button.sec{background:#26314f}
button.rec.on{background:var(--bad)}input[type=text]{flex:1;background:#0b1020;border:1px solid #26314f;color:var(--fg);border-radius:8px;padding:8px 10px}
.row{display:flex;gap:8px;align-items:center;margin-top:8px}
#decision .sum{font-size:16px;font-weight:600;margin-bottom:6px}.reason{display:flex;gap:8px;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted)}
.reason .ok{color:var(--ok)}.reason .no{color:var(--bad)}
#log{font-family:ui-monospace,Menlo,monospace;font-size:12px;max-height:220px;overflow:auto;white-space:pre-wrap}
#elements{font-family:ui-monospace,Menlo,monospace;font-size:11px;max-height:300px;overflow:auto;color:var(--muted)}
#cands span{display:inline-block;background:var(--blue);color:#fff;border-radius:6px;padding:3px 8px;margin:3px}
#pending{color:var(--acc);font-weight:600}
#gate{font-size:12px;color:var(--muted)}#gate b{color:var(--acc)}#answer{font-size:15px;white-space:pre-wrap;min-height:20px}
#aliases div{display:flex;gap:8px;align-items:center;font-size:13px;margin:3px 0}#aliases button{padding:2px 8px;font-size:11px}
label.tog{color:var(--muted);font-size:12px;display:flex;gap:4px;align-items:center}
</style></head><body>
<header><h1>Whippet</h1><span class="pill" id="ws">connecting…</span><span class="pill" id="model">model: loading</span>
<span class="pill" id="llm">llm: off</span><span class="pill" id="stats"></span><span class="pill" id="tabs"></span></header>
<main>
<div>
 <div class="card"><h2>Transcript</h2><div id="transcript"><span class="interim">Say something like “go to wikipedia”…</span></div>
  <div class="row"><button id="rec" class="rec">🎙 Start listening</button>
   <input id="cmd" type="text" placeholder="…or type a command and press Enter"><button id="send" class="sec">Send</button>
   <button id="undo" class="sec">Undo</button><button id="resnap" class="sec">Re-snapshot</button>
   <label class="tog"><input type="checkbox" id="speak" checked> speak replies</label></div>
  <div class="row"><span id="gate">gate: —</span></div>
  <div class="row"><span id="pending"></span><span id="cands"></span></div></div>
 <div class="card" style="margin-top:14px"><h2>Decision</h2><div id="decision"><div class="sum">—</div></div></div>
 <div class="card" style="margin-top:14px"><h2>Answer (LLM)</h2><div id="answer" style="color:var(--muted)">ask “summarize this page”, “what does this article say about …”</div></div>
 <div class="card" style="margin-top:14px"><h2>Log</h2><div id="log"></div></div>
</div>
<div><div class="card"><h2>Page</h2><div id="page" style="color:var(--muted);margin-bottom:8px">—</div><div id="elements"></div></div>
 <div class="card" style="margin-top:14px"><h2>Your phrases</h2><div id="aliases" style="color:var(--muted)">none yet — say “when I say yeet this tab, close the tab”</div></div></div>
</main>
<script>
const $=(id)=>document.getElementById(id);const esc=(s)=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let ws;function connect(){ws=new WebSocket((location.protocol==="https:"?"wss://":"ws://")+location.host+"/ws");
ws.onopen=()=>{$("ws").textContent="connected"};ws.onclose=()=>{$("ws").textContent="disconnected — retrying";setTimeout(connect,1000)};
ws.onmessage=(ev)=>{const m=JSON.parse(ev.data);handlers[m.type]&&handlers[m.type](m)}}
const send=(o)=>ws&&ws.readyState===1&&ws.send(JSON.stringify(o));
function renderSnapshot(s){if(!s)return;$("page").textContent=`${s.title||""} — ${s.url||""} (${s.site}, ${s.elements?.length||0} of ${s.rawCount||0} elements, search box ${s.searchBoxId||"none"})`;
$("elements").innerHTML=(s.elements||[]).map(e=>`<div>${e.id} ${e.role} "${esc(e.text)}"${e.href?" → "+esc(e.href):""}${e.below_fold?" ↓":""}</div>`).join("")}
function renderState(st){renderSnapshot(st.snapshot);$("model").textContent=`von ${st.model.revision.slice(0,7)} on ${st.model.device} (${st.model.dtype})`;
$("llm").textContent=st.llm?(st.llm.ready?`llm: ${st.llm.model.split("/").pop()}${st.llm.tps?" "+st.llm.tps+" tok/s":""}`:`llm: ${st.llm.error?"error":st.llm.lazy?"on first question":"loading…"}`):"llm: off";
$("stats").textContent=`${st.stats.decisions} decisions ~${st.stats.avg_decision_ms}ms · gate ~${st.stats.avg_gate_ms}ms · ${st.stats.actions} actions ~${st.stats.avg_action_ms}ms · ${st.stats.reflexes} reflexes`;$("tabs").textContent=`${st.tabs.length} tab(s)`;
$("pending").textContent=st.pending?`⚠ say "confirm" to ${st.pending.label||st.pending.type}`:"";renderCands(st.candidates);renderAliases(st.aliases)}
function renderCands(c){$("cands").innerHTML=(c||[]).map(x=>`<span>${x.n}: ${esc(x.label)}</span>`).join("")}
function renderAliases(a){if(!a||!a.length){$("aliases").textContent="none yet — say “when I say yeet this tab, close the tab”";return}
$("aliases").innerHTML=a.map(x=>`<div>“<b>${esc(x.phrase)}</b>” → ${esc(x.command)} <button class="sec" onclick='send({type:"forget",phrase:${JSON.stringify(x.phrase)}})'>forget</button></div>`).join("")}
function speak(t){if(!$("speak").checked||!window.speechSynthesis)return;speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.rate=1.05;speechSynthesis.speak(u)}
const handlers={state:renderState,snapshot:(m)=>renderSnapshot(m.snapshot),tabs:(m)=>$("tabs").textContent=`${m.tabs.length} tab(s)`,
transcript:(m)=>{$("transcript").innerHTML=m.final?esc(m.text):`${esc(m.text)} <span class="interim">…</span>`},
decision:(m)=>{const rs=(m.reasons||[]).map(r=>`<div class="reason"><span class="${r.pass?"ok":"no"}">${r.pass?"✓":"✗"}</span><span>${esc(r.name)}=${esc(r.value)}</span><span>≥ ${esc(r.threshold)}</span><span>${esc(r.note||"")}</span></div>`).join("");
$("decision").innerHTML=`<div class="sum">${esc(m.decision.toUpperCase())}: ${esc(m.summary||"")} <span style="color:var(--muted);font-weight:400">(${m.latencyMs}ms)</span></div>${rs}`},
action:(m)=>{},pending:(m)=>{$("pending").textContent=m.pending?`⚠ say "confirm" to ${m.pending.label||m.pending.type}`:""},candidates:(m)=>renderCands(m.candidates),
gate:(m)=>{$("gate").innerHTML=`gate: <b>${esc(m.family==="reflex"?m.kind:m.family)}</b> ${(m.confidence*100).toFixed(0)}% on “${esc(m.fragment)}” (${m.latencyMs}ms)${m.alias?` · alias “${esc(m.alias.phrase)}”`:""}`},
answer:(m)=>{$("answer").style.color="";$("answer").textContent=m.text||(m.done?"":"thinking…")},aliases:(m)=>renderAliases(m.aliases),
say:(m)=>{if(m.kind==="answer"||m.kind==="ask"||m.kind==="remap"||m.kind==="error")speak(m.text)},
log:(m)=>{const d=$("log");d.textContent+=`[${new Date(m.t*1000).toLocaleTimeString()}] ${m.text}\n`;d.scrollTop=d.scrollHeight}};
$("send").onclick=()=>{const t=$("cmd").value.trim();if(!t)return;$("cmd").value="";send({type:"command",text:t})};
$("cmd").onkeydown=(e)=>{if(e.key==="Enter")$("send").onclick()};$("undo").onclick=()=>send({type:"undo"});$("resnap").onclick=()=>send({type:"snapshot"});
let rec=null,on=false,base=0;
$("rec").onclick=()=>{const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR){alert("Web Speech API not available — use Chrome.");return}
if(on){on=false;rec&&rec.stop();$("rec").textContent="🎙 Start listening";$("rec").classList.remove("on");return}
rec=new SR();rec.continuous=true;rec.interimResults=true;rec.lang="en-US";rec.maxAlternatives=1;base++;
rec.onresult=(ev)=>{for(let i=ev.resultIndex;i<ev.results.length;i++){const r=ev.results[i];const text=r[0].transcript;send({type:"transcript",text,final:r.isFinal,utteranceId:`u${base}-${i}`})}};
rec.onend=()=>{if(on){base++;try{rec.start()}catch{}}};rec.onerror=(e)=>console.warn("speech",e.error);rec.start();on=true;$("rec").textContent="■ Stop listening";$("rec").classList.add("on")};
connect();
</script></body></html>
"""


async def serve(controller: Controller, host: str, port: int) -> None:
    from aiohttp import WSMsgType, web

    clients: set = set()

    def broadcast(msg: dict) -> None:
        data = json.dumps(msg, default=str)
        for ws in list(clients):
            asyncio.ensure_future(ws.send_str(data))
    controller.on(broadcast)

    async def index(_req):
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def state(_req):
        return web.json_response(controller.ui_state(), dumps=lambda o: json.dumps(o, default=str))

    async def ws_handler(req):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(req)
        clients.add(ws)
        await ws.send_str(json.dumps(controller.ui_state(), default=str))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except Exception:
                    continue
                t = m.get("type")
                if t == "transcript":
                    controller.handle_transcript(m.get("text", ""), bool(m.get("final")), m.get("utteranceId"))
                elif t == "command":
                    controller.handle_command(m.get("text", ""))
                elif t == "undo":
                    await controller.undo()
                elif t == "snapshot":
                    await controller.refresh_snapshot()
                elif t == "forget":
                    controller.aliases.remove(m.get("phrase", ""))
                    controller.emit({"type": "aliases", "aliases": controller.aliases.as_list()})
        finally:
            clients.discard(ws)
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/state", state)
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("UI: http://%s:%d/  (open in Chrome for the microphone)", host, port)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


# --------------------------------------------------------------------------------------
# Tests: decision integration cases on fixtures (captured live when the fixture dir is absent)
# --------------------------------------------------------------------------------------
SEARCH_RESULTS = {
    "url": "https://duckduckgo.com/?q=jev+typesafe", "title": "jev typesafe at DuckDuckGo", "site": "duckduckgo", "searchBoxId": "e02",
    "elements": [
        {"id": "e01", "role": "link", "text": "DuckDuckGo home", "href": "duckduckgo.com"},
        {"id": "e02", "role": "combobox", "text": "jev typesafe", "placeholder": "Search privately"},
        {"id": "e03", "role": "button", "text": "Search"},
        {"id": "e05", "role": "link", "text": "All", "href": "duckduckgo.com"},
        {"id": "e06", "role": "link", "text": "Images", "href": "duckduckgo.com"},
        {"id": "e07", "role": "link", "text": "Videos", "href": "duckduckgo.com"},
        {"id": "e08", "role": "link", "text": "News", "href": "duckduckgo.com"},
        {"id": "e12", "role": "button", "text": "Search Settings"},
        {"id": "e20", "role": "link", "text": "TypeSafe — Jev, the System One model", "href": "typesafe.ai"},
        {"id": "e21", "role": "link", "text": "typesafe.ai", "href": "typesafe.ai"},
        {"id": "e22", "role": "link", "text": "Jev 1.13 | TypeSafe Documentation", "href": "docs.typesafe.ai/models/jev"},
        {"id": "e23", "role": "link", "text": "docs.typesafe.ai", "href": "docs.typesafe.ai"},
        {"id": "e24", "role": "link", "text": "GitHub - typesafe-ai/typesafe-sdk-js: TypeScript SDK", "href": "github.com/typesafe-ai/typesafe-sdk-js"},
        {"id": "e25", "role": "link", "text": "github.com", "href": "github.com"},
        {"id": "e26", "role": "link", "text": "TypeSafe (@typesafe_ai) / X", "href": "x.com/typesafe_ai"},
        {"id": "e27", "role": "button", "text": "More results"},
        {"id": "e28", "role": "link", "text": "Settings", "href": "duckduckgo.com/settings", "below_fold": True},
        {"id": "e29", "role": "link", "text": "Privacy Policy", "href": "duckduckgo.com/privacy", "below_fold": True},
    ],
}
FORM_PAGE = {
    "url": "https://shop.example.com/checkout", "title": "Checkout — Example Shop", "site": "generic", "searchBoxId": None,
    "elements": [
        {"id": "e01", "role": "link", "text": "Example Shop"},
        {"id": "e02", "role": "textbox", "text": "", "placeholder": "Email address"},
        {"id": "e03", "role": "textbox", "text": "", "placeholder": "Card number"},
        {"id": "e04", "role": "select", "text": "Country"},
        {"id": "e05", "role": "button", "text": "Place order"},
        {"id": "e06", "role": "link", "text": "Back to cart"},
        {"id": "e07", "role": "button", "text": "Delete account"},
    ],
}
FIXTURE_URLS = {
    "example": "https://example.com/", "hn": "https://news.ycombinator.com/",
    "wikipedia-main": "https://en.wikipedia.org/wiki/Main_Page", "wikipedia-article": "https://en.wikipedia.org/wiki/Alan_Turing",
}
# (name, transcript, fixture, expectations)
TEST_CASES: list[dict] = [
    dict(name="scroll a bit", transcript="scroll down a bit", snapshot="wikipedia-article", intent="scroll_down", decision="act", amount="little"),
    dict(name="scroll to bottom", transcript="scroll all the way to the bottom", snapshot="wikipedia-article", intent="scroll_down", decision="act", amount="end"),
    dict(name="scroll up", transcript="scroll up", snapshot="hn", intent="scroll_up", decision="act"),
    dict(name="go to wikipedia", transcript="go to wikipedia", snapshot="example", intent="navigate_url", decision="act", url="wikipedia.org"),
    dict(name="open youtube", transcript="open youtube", snapshot="hn", intent="navigate_url", decision="act", url="youtube.com"),
    dict(name="spoken domain", transcript="go to example dot com", snapshot="hn", intent="navigate_url", decision="act", url="example.com"),
    dict(name="search web", transcript="search for jev typesafe", snapshot="example", intent="search_web", decision="act", text="jev typesafe", url="duckduckgo.com"),
    dict(name="search on site with box", transcript="search for alan turing", snapshot="wikipedia-main", intent="search_web", decision="act", text="alan turing", target_action="searchbox"),
    dict(name="search named site", transcript="search youtube for lofi beats", snapshot="hn", intent="search_web", decision="act", text="lofi beats", url="youtube.com/results"),
    dict(name="look up", transcript="look up the weather in berlin", snapshot="example", intent="search_web", decision="act", text="the weather in berlin"),
    dict(name="click first result", transcript="click the first result", snapshot=SEARCH_RESULTS, intent="click_element", target="e20", decision="act"),
    dict(name="click github result", transcript="click the github link", snapshot=SEARCH_RESULTS, intent="click_element", target="e24", decision="act"),
    dict(name="click by text", transcript="click the more information link", snapshot="example", intent="click_element", target="e01", decision="act"),
    dict(name="click hn new", transcript="click new", snapshot="hn", intent="click_element", target="e03", decision="act"),
    dict(name="click hn comments", transcript="open the comments tab", snapshot="hn", intent="click_element", target="e05", decision="act"),
    dict(name="go back", transcript="go back", snapshot="wikipedia-article", intent="go_back", decision="act"),
    dict(name="type into box", transcript="type hello world into the search box", snapshot="wikipedia-main", intent="type_into_field", target_action="searchbox", text="hello world", decision="act"),
    dict(name="reload", transcript="refresh the page", snapshot="hn", intent="reload", decision="act"),
    dict(name="new tab", transcript="open a new tab", snapshot="hn", intent="open_new_tab", decision="act"),
    dict(name="chit-chat ignored", transcript="so anyway I think we should get lunch", snapshot="hn", decision="ignore"),
    dict(name="filler ignored", transcript="um okay so", snapshot="hn", decision_in=["ignore", "wait"]),
    dict(name="partial go to waits", transcript="go to", snapshot="hn", final=False, decision="wait"),
    dict(name="partial search waits", transcript="search for", snapshot="hn", final=False, decision="wait"),
    dict(name="destructive asks confirm", transcript="click place order", snapshot=FORM_PAGE, intent="click_element", target="e05", decision="confirm"),
    dict(name="destructive delete", transcript="press delete account", snapshot=FORM_PAGE, intent="click_element", target="e07", decision="confirm"),
    dict(name="type email", transcript="type bob at example dot com in the email field", snapshot=FORM_PAGE, intent="type_into_field", target="e02", decision="act"),
    dict(name="missing element waits", transcript="click sign in", snapshot="example", intent="click_element", decision_in=["wait", "disambiguate"]),
]


async def capture_fixtures(fixture_dir: Path) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    b = Browser()
    await b.launch(headless=True, profile_dir=str(fixture_dir / "_profile"))
    try:
        for name, url in FIXTURE_URLS.items():
            f = fixture_dir / f"{name}.json"
            if f.exists():
                continue
            page = await b.ensure_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await page.wait_for_timeout(800)
            snap = await b.snapshot()
            f.write_text(json.dumps(snap, indent=1))
            log.info("captured fixture %s (%d elements)", name, len(snap["elements"]))
    finally:
        await b.close()


def run_tests(decider: Decider, fixture_dir: Path, only: str | None = None) -> int:
    def fixture(name: str) -> dict:
        return json.loads((fixture_dir / f"{name}.json").read_text())

    results = []
    for c in TEST_CASES:
        if only and only.lower() not in c["name"].lower():
            continue
        snap = fixture(c["snapshot"]) if isinstance(c["snapshot"], str) else c["snapshot"]
        r = decider.decide(c["transcript"], snap)
        a = r["answers"]
        final = c.get("final", True)
        policy = evaluate_policy(a, r["candidates"], snap, silent_ms=1000 if final else 0, is_final=final)
        fails: list[str] = []
        act = policy.get("action") or {}
        if c.get("intent") and a["intent"]["choice"] != c["intent"]:
            fails.append(f"intent {a['intent']['choice']} != {c['intent']}")
        if c.get("decision") and policy["decision"] != c["decision"]:
            fails.append(f"decision {policy['decision']} != {c['decision']} ({policy.get('summary')})")
        if c.get("decision_in") and policy["decision"] not in c["decision_in"]:
            fails.append(f"decision {policy['decision']} not in {c['decision_in']} ({policy.get('summary')})")
        if c.get("target") and act.get("targetId") != c["target"]:
            fails.append(f"target {act.get('targetId')} != {c['target']}")
        if c.get("target_action") == "searchbox" and act.get("targetId") != snap.get("searchBoxId"):
            fails.append(f"target {act.get('targetId')} != search box {snap.get('searchBoxId')}")
        if c.get("text") and (act.get("text") or act.get("query")) != c["text"]:
            fails.append(f"text {act.get('text') or act.get('query')!r} != {c['text']!r}")
        if c.get("url") and c["url"] not in (act.get("url") or ""):
            fails.append(f"url {act.get('url')} lacks {c['url']}")
        if c.get("amount") and act.get("amount") != c["amount"]:
            fails.append(f"amount {act.get('amount')} != {c['amount']}")
        ok = not fails
        results.append(dict(name=c["name"], ok=ok, latency=r["latencyMs"], fails=fails))
        tgt = a.get("target", {})
        print(f"  {'✓' if ok else '✗'} {c['name']:<26} {r['latencyMs']:>5}ms  intent={a['intent']['choice']}({a['intent']['confidence']:.2f}) "
              f"target={tgt.get('choice')}({tgt.get('confidence', 0):.2f}) complete={a['complete']['noul']:.2f} cmd={a['is_command']['noul']:.2f} "
              f"destr={a['destructive']['noul']:.2f} → {policy['decision']}" + ("\n      " + "; ".join(fails) if fails else ""))
    passed = sum(1 for r in results if r["ok"])
    lat = sorted(r["latency"] for r in results)
    print(f"\n{passed}/{len(results)} passed  latency p50={lat[len(lat) // 2]}ms p95={lat[int(len(lat) * 0.95) - 1]}ms mean={sum(lat) // len(lat)}ms  "
          f"device={decider.model.device} calls={decider.model.calls} pairs={decider.model.pairs}")
    return 0 if passed == len(results) else 1


# --------------------------------------------------------------------------------------
# Demo: word-by-word replay through the real controller against live sites
# --------------------------------------------------------------------------------------
async def run_demo(controller: Controller, word_ms: int = 120) -> int:
    b = controller.browser

    def url() -> str:
        return b.page.url if b.page else ""

    async def scroll_y() -> int:
        return int(await (await b.ensure_page()).evaluate("() => window.scrollY"))

    steps: list[dict] = [
        dict(say="go to wikipedia", expect=lambda: "wikipedia.org" in url()),
        dict(say="search for alan turing", expect=lambda: re.search(r"Alan_Turing|search=alan", url(), re.I) is not None),
        dict(say="scroll down a bit", expect_async=lambda: scroll_y(), cmp=lambda y: y > 50),
        dict(say="scroll to the bottom", expect_async=lambda: scroll_y(), cmp=lambda y: y > 2000),
        dict(say="scroll up a page", expect=lambda: True),
        dict(say="go back", expect=lambda: "wikipedia.org" in url() and "Alan_Turing" not in url()),
        dict(say="open example dot com", expect=lambda: "example.com" in url()),
        dict(say="click the more information link", expect=lambda: "iana.org" in url()),
        dict(say="go to hacker news", expect=lambda: "news.ycombinator.com" in url()),
        dict(say="click the new link", expect=lambda: "news.ycombinator.com/newest" in url()),
        dict(say="click on a link please", follow_up="the first one", expect=lambda: not re.search(r"news\.ycombinator\.com/newest$", url())),
        dict(say="search duckduckgo for typesafe jev", expect=lambda: re.search(r"duckduckgo\.com/\?q=typesafe(%20|\+)jev", url()) is not None),
        dict(say="open a new tab", expect=lambda: len(b.pages) == 2),
        dict(say="close this tab", expect=lambda: len(b.pages) == 1),
        dict(say="so anyway I think we should get lunch", no_action=True, expect=lambda: True),
    ]
    passed = 0
    for i, s in enumerate(steps):
        actions_before = controller.stats["actions"]
        words = s["say"].split()
        uid = f"demo-{i}"
        acted_at_word = None
        for w in range(1, len(words) + 1):
            controller.handle_transcript(" ".join(words[:w]), final=(w == len(words)), utterance_id=uid)
            await asyncio.sleep(word_ms / 1000)
            if acted_at_word is None and controller.stats["actions"] > actions_before:
                acted_at_word = w
        # wait for the utterance to settle (decision + action)
        for _ in range(80):
            await asyncio.sleep(0.1)
            if controller.utt and controller.utt.acted and not controller._busy.locked():
                break
        if s.get("follow_up") and controller.candidates:
            controller.handle_transcript(s["follow_up"], final=True, utterance_id=uid + "-pick")
            for _ in range(60):
                await asyncio.sleep(0.1)
                if controller.stats["actions"] > actions_before and not controller._busy.locked():
                    break
        await asyncio.sleep(0.5)
        acted = controller.stats["actions"] > actions_before
        if s.get("no_action"):
            ok = not acted
        else:
            val = await s["expect_async"]() if s.get("expect_async") else None
            ok = acted and (s["cmp"](val) if s.get("cmp") else s["expect"]())
        passed += ok
        dm = controller.stats["decision_ms"]
        print(f"  {'✓' if ok else '✗'} {s['say']:<45} acted={'word ' + str(acted_at_word) + '/' + str(len(words)) if acted_at_word else ('yes' if acted else 'no'):<12} "
              f"url={url()[:70]} lastDecision={dm[-1] if dm else '-'}ms")
    st = controller.ui_state()["stats"]
    print(f"\n{passed}/{len(steps)} demo steps passed  decisions={st['decisions']} avg={st['avg_decision_ms']}ms  actions={st['actions']} avg={st['avg_action_ms']}ms")
    return 0 if passed == len(steps) else 1


async def say_and_wait(controller: Controller, text: str, timeout_s: float = 45.0, stream_ms: int = 0) -> dict:
    """Feed one utterance and wait until the controller has settled. Returns a report.
    stream_ms > 0 feeds it word by word as interim transcripts (like Web Speech), then the final."""
    events: list[dict] = []
    controller.on(events.append)
    actions_before = controller.stats["actions"]
    uid = f"cmd-{int(time.time() * 1000)}"
    if stream_ms:
        words = text.split()
        for i in range(1, len(words)):
            controller.handle_transcript(" ".join(words[:i]), final=False, utterance_id=uid)
            await asyncio.sleep(stream_ms / 1000)
    controller.handle_transcript(text, final=True, utterance_id=uid)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        await asyncio.sleep(0.1)
        u = controller.utt
        if u and (u.acted or u.done) and not u.queue and not controller._busy.locked() and not controller._gate_lock.locked():
            break
    await asyncio.sleep(0.3)
    controller.listeners.remove(events.append)
    decisions = [e for e in events if e["type"] == "decision"]
    acts = [e for e in events if e["type"] == "action"]
    b = controller.browser
    return {
        "gate": [f"{g['family']}/{g['kind']} {g['confidence']:.2f} “{g['fragment']}” {g['latencyMs']}ms" for g in events if g["type"] == "gate"],
        "said": [s["text"] for s in events if s["type"] == "say" or (s["type"] == "answer" and s.get("done"))],
        "text": text, "decisions": [f"{d['decision']}:{d.get('summary', '')}" for d in decisions],
        "actions": [f"{describe(a['action'])} -> {'ok' if a['result'].get('ok') else 'FAIL ' + str(a['result'].get('detail'))} {a['ms']}ms" for a in acts],
        "acted": controller.stats["actions"] > actions_before, "url": b.page.url if b.page else "",
        "title": controller.snapshot.get("title", ""), "tabs": len(b.pages),
        "decision_ms": [d.get("latencyMs") for d in decisions], "candidates": controller.candidates["items"] if controller.candidates else [],
        "pending": controller.pending,
    }


async def run_script(controller: Controller, lines: list[str]) -> int:
    """Run utterances one per line against the live browser. Lines starting with '#' are comments;
    '~ text' streams the words as interim speech; a trailing ' => regex' asserts on URL | title | tabs | said."""
    passed = total = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            if line:
                print(line)
            continue
        text, _, expect = line.partition(" => ")
        stream, nowait = text.startswith("~"), text.startswith("!")
        text = text.lstrip("~! ")
        r = await say_and_wait(controller, text, stream_ms=140 if stream else 0, timeout_s=0.8 if nowait else 45.0)
        ok = True
        if expect:
            total += 1
            hay = f"{r['url']} | {r['title']} | tabs={r['tabs']} | said={' / '.join(r['said'])} | decided={' / '.join(r['decisions'])}"
            ok = re.search(expect.strip(), hay, re.I) is not None and (r["acted"] or bool(r["said"]) or "ignore" in expect)
            passed += ok
        mark = ("✓" if ok else "✗") if expect else "·"
        print(f"{mark} {'~ ' if stream else ''}{text}")
        for g in r["gate"]:
            print(f"      gate     {g}")
        for d in r["decisions"]:
            print(f"      decision {d}")
        for s in r["said"]:
            print(f"      said     {s[:200]}")
        for a in r["actions"]:
            print(f"      action   {a}")
        if r["candidates"]:
            print(f"      candidates {[c['label'] for c in r['candidates']]}")
        if r["pending"]:
            print(f"      pending  {describe(r['pending'])}")
        print(f"      -> {r['url'][:90]}  [{r['title'][:50]}]  tabs={r['tabs']}  decide={r['decision_ms']}ms")
    st = controller.ui_state()["stats"]
    print(f"\n{passed}/{total} assertions passed  decisions={st['decisions']} avg={st['avg_decision_ms']}ms  gate avg={st['avg_gate_ms']}ms  "
          f"actions={st['actions']} avg={st['avg_action_ms']}ms  reflexes={st['reflexes']}  asks={st['asks']}")
    return 0 if passed == total else 1


async def run_walkthrough(controller: Controller, lines: list[str]) -> None:
    """`--walkthrough`: a narrated guided tour in the live browser. Line grammar:
         > text      Whippet says this (spoken) and shows it in the panel - narration
         text        an utterance, exactly as a user would say/type it ("~ text" streams it word by word,
                     "! text" does not wait for Whippet to finish speaking - to demo interrupting it)
         wait N      pause N seconds (default 1.5 after every utterance)
         # text      comment, printed only
       Ends by handing the browser over to the user with voice mode on if hearing works."""
    t0 = time.time()
    while time.time() - t0 < 60 and ((controller.tts and not controller.tts.ready and not controller.tts.error)
                                     or (controller.stt and not controller.stt.ready and not controller.stt.error)):
        await asyncio.sleep(0.2)  # let Pocket TTS/Whisper finish loading so the tour is spoken
    controller.log("walkthrough: start")
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            text = line.lstrip("> ")
            controller.say(text, "info")
            await asyncio.sleep(0.4)
            while controller.tts and controller.tts.speaking:
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.4)
            continue
        if line.startswith("wait "):
            await asyncio.sleep(float(line.split()[1]))
            continue
        stream = line.startswith("~")
        nowait = line.startswith("!")
        text = line.lstrip("~! ")
        r = await say_and_wait(controller, text, stream_ms=160 if stream else 0)
        t1 = time.time()
        while controller.tts and controller.tts.speaking and not nowait and time.time() - t1 < 40:
            await asyncio.sleep(0.1)
        print(f"· {text}  -> {r['url'][:80]}  {r['actions'][-1] if r['actions'] else ''}")
        await asyncio.sleep(1.5)
    if controller.stt and controller.stt.ready and not controller.voice_on:
        await controller.set_voice(True)
    controller.log("walkthrough: done - it's yours")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def speech_test(tts_id: str, voice: str, stt_id: str) -> int:
    """Load Pocket TTS + Whisper, report cold/warm latencies, and round-trip a spoken sentence through both."""
    import numpy as np
    tts = TTS(tts_id, voice)
    tts.load()
    print(f"tts  {tts_id} ({voice}) ready={tts.ready} load={tts.load_ms}ms {tts.error or ''}")
    if not tts.ready:
        return 1
    out: dict = {}
    for text in ["Going back.", "Sorry, I could not find anything like that on this page.",
                 "Which one? 1, Getting started, or 2, Documentation, or 3, Download."]:
        done = threading.Event()
        chunks: list[bytes] = []
        t0 = time.time()
        first = [0.0]

        def sink(pcm: bytes, gen: int, last: bool) -> None:
            if not chunks:
                first[0] = time.time() - t0
            chunks.append(pcm)
            if last:
                done.set()
        tts.speak(text, sink)
        done.wait(30)
        audio = b"".join(chunks)
        secs = len(audio) / 2 / TTS_RATE
        print(f"     first audio {first[0] * 1000:.0f}ms  total {((time.time() - t0) * 1000):.0f}ms  {secs:.1f}s of speech  \"{text[:48]}\"")
        out[text] = audio
    if stt_id.lower() in ("off", "none", ""):
        return 0
    stt = STT(stt_id)
    stt.load()
    print(f"stt  {stt_id} ready={stt.ready} load={stt.load_ms}ms {stt.error or ''}")
    if not stt.ready:
        return 1
    for text, audio in out.items():  # Pocket TTS -> Whisper round trip (resampled 24k -> 16k)
        x = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
        n = int(len(x) * STT_RATE / TTS_RATE)
        x16 = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        t0 = time.time()
        heard = stt.transcribe(x16, final=True)
        print(f"     {((time.time() - t0) * 1000):.0f}ms  heard: \"{heard}\"")
    print(f"     silence -> \"{stt.transcribe(np.zeros(STT_RATE * 2, dtype=np.float32))}\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Whippet - local voice-controlled browser (single file)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--cdp", default=None, help="attach to Chrome over CDP, e.g. ws://127.0.0.1:9222/devtools/browser/…")
    ap.add_argument("--profile", default=DEFAULT_PROFILE_DIR)
    ap.add_argument("--start-url", default="https://duckduckgo.com/")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--test", action="store_true", help="run decision integration tests on fixtures")
    ap.add_argument("--only", default=None, help="filter --test cases by name substring")
    ap.add_argument("--fixtures", default=str(Path(__file__).with_name("fixtures")), help="fixture dir (captured live if missing)")
    ap.add_argument("--demo", action="store_true", help="live headless browser automation demo")
    ap.add_argument("--say", default=None, help="run one typed command against the start URL and exit")
    ap.add_argument("--script", default=None, help="file of utterances (one per line, optional ' => regex' assertion) to run headlessly")
    ap.add_argument("--llm", default=DEFAULT_LLM, metavar="MLX_MODEL",
                    help="mlx-lm chat model for summaries/questions (e.g. mlx-community/Qwen3.5-35B-A3B-4bit); 'off' to disable")
    ap.add_argument("--aliases", default=DEFAULT_ALIASES_PATH, help="where taught phrases are stored ('' = memory only)")
    ap.add_argument("--tts", default=DEFAULT_TTS, metavar="MLX_MODEL", help="TTS model (mlx-audio, Pocket TTS by default); 'off' to disable spoken feedback")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="TTS voice: alba, marius, javert, jean, fantine, cosette, eponine, azelma")
    ap.add_argument("--stt", default=DEFAULT_STT, metavar="MLX_MODEL",
                    help="mlx-whisper model for local speech recognition (e.g. mlx-community/whisper-small-mlx); 'off' = browser Web Speech")
    ap.add_argument("--fake-mic", default=None, metavar="WAV", help="feed this 16 kHz mono WAV as the microphone (voice-mode testing)")
    ap.add_argument("--voice-on", action="store_true", help="start with voice mode enabled")
    ap.add_argument("--ptt", action="store_true", help="start in push-to-talk: the mic only listens while Space is held (Alt+P toggles)")
    ap.add_argument("--speech-test", action="store_true", help="load Pocket TTS + Whisper, print latencies and exit")
    ap.add_argument("--doctor", action="store_true", help="check every dependency and model, print fixes, exit")
    ap.add_argument("--slides", default=DEFAULT_SLIDES_PATH if Path(DEFAULT_SLIDES_PATH).exists() else None, metavar="PDF",
                    help=f"a PDF deck to present by voice: 'open my slides', 'next', 'last slide', 'slide 7' (default {DEFAULT_SLIDES_PATH} if present)")
    ap.add_argument("--walkthrough", nargs="?", const=str(Path(__file__).with_name("scripts") / "walkthrough.txt"), default=None,
                    metavar="FILE", help="narrated guided demo in the live browser, then keep running")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if args.doctor:
        return doctor(args.tts, args.stt, args.llm)
    if args.speech_test:
        return speech_test(args.tts, args.voice, args.stt)

    model = VonModel(device=args.device, dtype=args.dtype, revision=args.revision)
    decider = Decider(model)

    if args.test:
        fdir = Path(args.fixtures)
        if not all((fdir / f"{n}.json").exists() for n in FIXTURE_URLS):
            log.info("capturing fixtures into %s", fdir)
            asyncio.run(capture_fixtures(fdir))
        return run_tests(decider, fdir, args.only)

    llm = None if args.llm.lower() in ("off", "none", "") else LLM(args.llm)  # loads on the first page question
    aliases = AliasStore(Path(args.aliases).expanduser() if args.aliases else None)
    batch = args.demo or bool(args.say) or bool(args.script) or args.test
    tts = None if batch or args.tts.lower() in ("off", "none", "") else TTS(args.tts, args.voice)
    stt = None if batch or args.stt.lower() in ("off", "none", "") else STT(args.stt)
    if tts:  # Pocket TTS is small: load it first so the very first reply is already spoken
        threading.Thread(target=lambda: (tts.load(), stt and stt.load()), name="speech-load", daemon=True).start()
    elif stt:
        threading.Thread(target=stt.load, name="stt-load", daemon=True).start()

    if not args.slides:  # no --slides: ~/.whippet/slides.pdf, then scripts/whippet.pdf (the "how it works" deck), then any PDF nearby
        here = Path(__file__).resolve().parent
        found = sorted(here.glob("*.pdf")) + sorted((here / "scripts").glob("*.pdf"))
        found = sorted(found, key=lambda f: (f.name != "whippet.pdf", f.name == "sample_slides.pdf", str(f)))
        args.slides = str(found[0]) if found else None
    deck = SlideDeck(args.slides) if args.slides else None
    if deck:
        threading.Thread(target=deck.load, name="slides-load", daemon=True).start()

    async def app() -> int:
        browser = Browser()
        browser.deck = deck
        try:
            await browser.launch(headless=args.headless or args.demo or bool(args.say) or bool(args.script), cdp=args.cdp,
                                 profile_dir=args.profile, start_url="about:blank" if args.demo else args.start_url)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                log.error("Chromium for Playwright is not installed.  fix: %s -m playwright install chromium", sys.executable)
                return 2
            raise
        controller = Controller(browser, decider, aliases=aliases, llm=llm, tts=tts, stt=stt)
        controller.on(lambda m: log.debug("event %s", m.get("type")) if m.get("type") not in ("log",) else None)
        await controller.start()
        if args.voice_on:
            await controller.set_voice(True)
        if args.ptt:
            await controller.set_ptt(True)
        if args.fake_mic:
            asyncio.ensure_future(controller.play_fake_mic(Path(args.fake_mic)))
        try:
            if args.demo:
                return await run_demo(controller)
            if args.script:
                return await run_script(controller, Path(args.script).read_text().splitlines())
            if args.say:
                print(json.dumps(await say_and_wait(controller, args.say), default=str, indent=1))
                return 0
            if args.walkthrough:
                asyncio.ensure_future(run_walkthrough(controller, Path(args.walkthrough).read_text().splitlines()))
            await serve(controller, args.host, args.port)
            return 0
        finally:
            await controller.close()

    try:
        return asyncio.run(app())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
