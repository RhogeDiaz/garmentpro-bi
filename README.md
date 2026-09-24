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

Put the UCI CSV at `data/productivity.csv`.

The app has:
- Executive Overview
- Productivity Drivers
- Productivity Diagnostics
- Forecasting
- Productivity Simulator
- AI Assistant
- About page

The demo runs even without the CSV by generating synthetic data. Replace the demo formulas with your validated ML/forecast models before final presentation.
