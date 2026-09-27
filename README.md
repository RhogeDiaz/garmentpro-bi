# GarmentPro BI — Shiny for Python

## Run Locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
shiny run --reload app.py
```

Set the Gemini key in your shell before starting the app. In PowerShell:

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
shiny run --reload app.py
```

Do not put the key in `app.py`, a committed `.env` file, or GitHub. The app defaults to `gemini-3.8-flash`; set `GEMINI_MODEL` as an environment variable only if you need a different supported model. Without a key, the chatbot uses its local prepared answers. Gemini request failures return `Google servers are busy as of the moment`.

The chatbot sends Gemini a compact summary of the chart data on the current page, not a screenshot of the chart. Put the UCI CSV at `data/productivity.csv`. The Random Forest files are loaded from `models/rf_finishing.pkl` and `models/rf_sewing.pkl`.

## Deploy To Posit Connect Cloud

1. Confirm `app.py`, `requirements.txt`, `data/productivity.csv`, and both files under `models/` are included in the GitHub repository. The models are loaded from these relative paths at startup. Do not commit API keys or either virtual environment.
2. Push the latest project changes to GitHub. Connect Cloud's free GitHub workflow requires a public repository; private repositories require a plan that supports them.
3. Sign in to [Posit Connect Cloud](https://connect.posit.cloud/) and choose **Publish**, then select **Shiny** and the project repository.
4. Select the branch to deploy and set `app.py` as the primary file. Choose Python 3.13 in Advanced Settings to match local development.
5. In the deployment's secret variables, add `GEMINI_API_KEY` and paste the key there. Do not put the key in the repository. Optionally add `GEMINI_MODEL` if you want to override the default model.
6. Publish and inspect the deployment logs. Open the published app, test a chat question on Overview and Drivers, and verify that Simulator predictions load both model files.
7. With GitHub auto-republish enabled, pushing later changes to the connected branch will redeploy the app. Update the secret in Connect Cloud's settings if you rotate the API key.

The app works without the CSV by generating synthetic data. The pickle models were trained with scikit-learn 1.6.1, which is pinned in `requirements.txt` to match their saved estimator version.
