# GarmentPro BI — Shiny for Python

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
shiny run --reload app.py
```

InsightBot uses Gemini 3.8 Flash for generation and Gemini Embedding 2 for
glossary retrieval. Both run on Google's Gemini Developer API free tier, subject
to its model availability and rate limits. Create a key in
[Google AI Studio](https://aistudio.google.com/apikey), keep the project on the
Free Tier, and set `GOOGLE_API_KEY` in the process environment. `GEMINI_API_KEY`
is also accepted. Do not link billing or enable auto-reload if you need a strict
zero-spend setup; requests will stop when the free quota is exhausted.

In PowerShell, set the key in the terminal used to launch Shiny:

```powershell
$env:GOOGLE_API_KEY = "your-key"
shiny run --reload app.py
```

For Posit hosting, configure `GOOGLE_API_KEY` as a private project/deployment
environment variable, never as a committed file or client-side setting. The app
initializes the LlamaIndex engines on the first chat message, reads the dataset
locally, and stores Chroma data under `storage/`.

Free-tier prompts and responses may be used by Google to improve its products.
InsightBot sends chart context, glossary text for embeddings, and a dataframe
preview used by PandasQueryEngine to Google. Do not use it with confidential or
personal data unless that handling is approved. PandasQueryEngine executes
model-generated Python in the local app process; this is not an operating-system
sandbox.

Put the UCI CSV at `data/productivity.csv`.

The app has:
- Executive Overview
- Productivity Drivers
- Productivity Diagnostics
- Forecasting
- Productivity Simulator
- AI Assistant
- About page

InsightBot answers using the chart snapshots visible on its page, the loaded
dataset, and definitions in `context/`. Add approved business definitions to
`context/metrics.md` and `context/glossary.md`; the local Chroma index is created
on first use and reused across restarts.

The demo runs even without the CSV by generating synthetic data. Replace the demo formulas with your validated ML/forecast models before final presentation.
