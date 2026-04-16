# Contributing to Shopper Intervention

Note: For detailed setup and Quick Start instructions (installing dependencies, running the API and dashboard, fetching training data), see the project's README — Quick Start section: [Quick Start](README.md#quick-start). This file focuses on workflow, branching, and PR conventions.

Thanks for your interest in contributing. This document outlines the workflow for making changes without breaking the main branch.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/smbrownai/shopper_intervention.git
cd shopper_intervention
```

### 2. Set up your environment

```bash
python -m venv mlenv
source mlenv/bin/activate
pip install -r requirements.txt
```

---

## Branching Convention

Never commit directly to `main`. Always work on a named branch.

| Branch prefix | Use for |
|---|---|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `data/` | Dataset or preprocessing changes |
| `docs/` | README, comments, documentation |

**Example:**
```bash
git checkout -b feature/add-xgboost-model
```

---

## Daily Workflow

### Before starting work — get the latest

```bash
git checkout main
git pull
git checkout your-branch
git merge main
```

### While working — commit often

```bash
git add .
git commit -m "Short description of what changed"
```

Good commit messages are specific:
- ✅ `Add revenue threshold feature to features.py`
- ✅ `Fix null handling in preprocessing pipeline`
- ❌ `changes`
- ❌ `update`

### When ready to share — push your branch

```bash
git push origin your-branch-name
```

Then open a **Pull Request** on GitHub. Add a short description of what changed and why.

---

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Make sure the API still runs: `uvicorn api.main:app --reload`
- Make sure the Streamlit app still runs: `streamlit run ui/app.py`
- If you changed `train.py`, include a note on model performance changes

---

## Project Structure

```
shopper_intervention/
├── data/                  ← Dataset (do not modify the source CSV)
├── scripts/
│   ├── features.py        ← Shared preprocessing pipeline
│   └── train.py           ← MLflow training experiment
├── api/
│   └── main.py            ← FastAPI /predict and /predict-batch endpoints
├── ui/
│   └── app.py             ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Questions

Open a GitHub Issue or reach out via [shawnnext.ai](https://shawnnext.ai).
