# Text Summarizer: Dialogue Summarization

A text summarization app that generates short, abstractive summaries of conversations. I fine-tuned **T5-small** on the **SAMSum** dialogue dataset and wrapped the model in a **FastAPI** backend with a simple web UI: paste text, click *Summarize*, get a summary.

## Features

- Fine-tuned T5-small for abstractive dialogue summarization
- Text cleaning and tokenization pipeline (512 input / 150 target tokens)
- Summary generation with beam search (4 beams)
- FastAPI backend with a `POST /summarize/` endpoint
- Web UI (HTML/CSS/JavaScript) served by the same app

## Tech Stack

Python, PyTorch, Hugging Face Transformers, Pandas, FastAPI, Uvicorn, HTML/CSS/JavaScript

## Training Details

| Item | Value |
|---|---|
| Base model | `t5-small` |
| Dataset | SAMSum (4,000 random training samples, 500 random validation samples) |
| Epochs | 6 |
| Batch size | 8 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max input / target length | 512 / 150 tokens |
| Hardware | Apple MPS (Mac GPU) |

Preprocessing (in the notebook) removes line breaks, extra spaces and HTML tags, and lowercases the text before tokenization.

## Project Structure

```
.
├── app.py                    # FastAPI app (API + serves the UI)
├── index.html                # Web UI
├── text_summarizer.ipynb     # Data prep, fine-tuning and testing
├── requirements.txt
└── saved_summary_model/      # Created by the notebook (not included in the repo)
```

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

The model weights are too large for GitHub, so they are not in this repo. Generate them by running the notebook:

1. Download the SAMSum data as `samsum-train.csv` and `samsum-validation.csv` (columns: `id`, `dialogue`, `summary`) and put them in the project folder.
2. Open `text_summarizer.ipynb` and run all cells. This fine-tunes T5-small and saves it to `./saved_summary_model`.

### 3. Run the app

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## API

**`POST /summarize/`**

Request:

```json
{
  "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow :-)"
}
```

Response:

```json
{
  "summary": "..."
}
```

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/summarize/ \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I will bring you tomorrow."}'
```

Interactive API docs are available at http://127.0.0.1:8000/docs.

## Limitations

- Trained on a 4,000-sample subset of SAMSum, so it works best on short, chat-style dialogues.
- Inputs longer than 512 tokens are truncated.
- On long, news-style text the model tends to pick out a couple of sentences rather than truly summarize.
- Not yet evaluated with ROUGE.

## Possible Improvements

- Evaluate with ROUGE on the validation/test set
- Train on the full SAMSum dataset or a larger model (t5-base)
- Deploy the app online
