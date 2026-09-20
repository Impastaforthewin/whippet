# Whippet

A voice-controlled browser that runs its models on your Mac.

Whippet lets you open pages, follow links, fill in fields, and move between tabs by talking. It can read a page aloud, answer questions about it, control a video, or walk through a task one step at a time. The goal is to make everyday browsing less dependent on a mouse or being able to see the screen.

The whole application lives in one Python file, including the browser panel. Speech recognition, command decisions, spoken feedback, and page questions run locally on Apple Silicon. No API key required.

## What you can say

```text
go to wikipedia
search for whippet
open the link about italian greyhound
read the page
stop
```

You can give several instructions in one sentence:

```text
open a new tab, go to example dot com, then scroll down
```

Or ask Whippet to take them one at a time:

```text
guide me through: go to wikipedia, search for project mercury, then open the link about alan shepard
```

The first step runs immediately. Say `next` to continue, `skip` to move past a step, or `stop` to end the sequence.

You can teach it your own phrases, too:

```text
when I say yeet this tab, close the tab
yeet this tab
```

Taught phrases persist between sessions. Say `forget yeet this tab` to remove one.

| Task | Try saying |
| --- | --- |
| Get your bearings | “where am I” or “what's on this page” |
| Follow a result | “click the first story” or “click the second result” |
| Ask about the page | “summarize this page” or “what is this page about” |
| Control a video | “pause the video”, “captions on”, “speed 1.5x”, “rewind ten seconds” |
| Present a PDF | “open my slides”, “next slide”, “slide four”, “last slide” |
| Find something | “find the word history” |

Commands can also be typed into the panel.

## Run it

You need an **Apple Silicon Mac** and **Python 3.11 or newer**. Python 3.12 is a good starting point. The speech and page-question models use MLX.

Download or clone this repository, open a terminal in its folder, and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
python whippet.py --doctor
python whippet.py
```

If you use another supported Python version, replace `python3.12` in the first command with that interpreter.

The first run downloads model weights from Hugging Face. Allow time and several gigabytes of disk space for those downloads. `--doctor` checks dependencies and loads the speech models; Von loads when the browser starts, and Qwen loads on the first page question.

Whippet opens Chromium with a panel on the right. Press **Option+V** to enable voice mode and allow microphone access when prompted.

| Shortcut on Mac | Action |
| --- | --- |
| Option+V | Toggle voice mode |
| Option+P | Toggle push-to-talk |
| Hold Space | Talk while push-to-talk is enabled |
| Option+K | Focus the command box |
| Option+Space | Hide or show the panel |

Space still types normally when a text field has focus. In a noisy room, start with push-to-talk:

```bash
python whippet.py --ptt
```

To use a PDF deck:

```bash
python whippet.py --slides "/path/to/your/deck.pdf"
```

Then say “open my slides.” A sample deck is not included in this minimal distribution.

## How it works

Whisper transcribes speech as it arrives. A local **Von** decision model sorts those fragments into commands, interruptions, taught phrases, and questions. For browser actions, Whippet collects the page's links, buttons, and fields, then uses Von to choose among concrete candidates. Playwright carries out the action.

The command path can react to “stop” or “go back” while work is still in progress. When a target is ambiguous, Whippet can offer numbered choices. Actions flagged as having side effects are held for `confirm` or `cancel`.

**Qwen** handles questions and summaries using extracted page text. It loads only when needed. **Pocket TTS** speaks replies, errors, and confirmations.

| Component | Role |
| --- | --- |
| [Von](https://huggingface.co/wfzyx/von-1.0) | Command classification and action selection; ModernBERT NLI through PyTorch |
| [Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | Local speech recognition through `mlx-whisper`; base for interim transcripts, large-v3-turbo for final transcripts |
| [Pocket TTS](https://huggingface.co/mlx-community/pocket-tts) | Spoken feedback through `mlx-audio` |
| [Qwen3.5-4B](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | Page questions and summaries through `mlx-lm`, in 4-bit precision |
| [Playwright](https://playwright.dev/python/) | Chromium control and page inspection |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF rendering and text extraction for slides |

The Python code, panel HTML/CSS/JavaScript, and wordmark font are all in `whippet.py`. Model weights and Chromium are downloaded separately.

## Local processing

With the local models loaded, Whippet processes microphone audio, command decisions, page questions, and spoken replies on your Mac. Websites still receive normal browsing traffic, and initial setup needs internet access.

If local speech recognition fails or is disabled with `--stt off`, Whippet falls back to the browser's Web Speech API. That fallback may use a remote service and may be unavailable in Playwright's Chromium. Keep local Whisper enabled for local speech processing.

Browser state is stored in `~/.whippet/profile`, and taught phrases in `~/.whippet/aliases.json`.

## Useful options

```bash
python whippet.py --start-url https://en.wikipedia.org
python whippet.py --voice-on
python whippet.py --llm off
python whippet.py --help
```

`--llm off` disables page questions and summaries while keeping browser commands available.

Decision fixtures can be run with `python whippet.py --test`. In this minimal distribution, the first test run captures missing fixtures from live websites, so it needs internet access and results can depend on those pages.

## Current limits

The full local setup targets Apple Silicon. Speech recognition is configured for English. Sites that block automated browsers can interrupt navigation and search, and unusual page layouts or video players may need manual interaction.

Page answers depend on the text Whippet can extract. PDF questions use the deck's text layer; scanned slides without text will have little to answer from.

Whippet began as a Python port of the Jev voice browser, replacing its remote decision API with local Von inference.
