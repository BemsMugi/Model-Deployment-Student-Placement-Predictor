# Student Placement Predictor

A Streamlit application that predicts whether a student is likely to be placed in a job and, when placed, estimates their salary package in LPA (lakhs per annum). It also includes visualizations of the student dataset.

## Live Demo

Try the deployed app on Streamlit: [Student Placement Predictor](https://student-placement-5glcw4hlx7qv5yrou95tpf.streamlit.app/).

## Features

- Predicts placement status: placed or not placed.
- Shows the classifier's placement probability when available.
- Estimates salary for students predicted to be placed.
- Displays an academic, technical, soft-skill, entrance exam, and experience profile.
- Explores dataset target distributions, numeric feature correlations, and descriptive statistics.

## Models and reported metrics

- **Placement classification:** Decision Tree Classifier; the app reports 100% accuracy.
- **Salary regression:** Linear Regression; the app reports an R² score of 0.0248.

The salary model's reported R² is low, so its estimates should be treated as rough outputs and not as reliable salary forecasts. Metrics are reported by the application; consult the model evaluation workflow and test data for full context.

## Project files

```text
.
├── app.py                 # Streamlit interface and prediction logic
├── B.csv                  # Dataset used by the visualization page
├── best_classifier.pkl    # Trained placement classification model
├── best_regressor.pkl     # Trained salary regression model
├── requirements.txt       # Python dependencies
└── README.md
```

## Requirements

- Python 3.9 or later is recommended.
- Packages listed in `requirements.txt`.

## Run locally

Clone or download this repository, then open a terminal in the project folder. Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Using the app

1. Choose **Prediction** from the sidebar.
2. Enter the student's academic, skill, experience, and other details.
3. Select **Predict Now** to view placement status, probability, and (when applicable) estimated salary.
4. Choose **Data Visualization** to explore the dataset.
5. Choose **About Model** to view the algorithms and reported metrics.

## Notes

- Keep `best_classifier.pkl` and `best_regressor.pkl` in the same directory as `app.py`; the application loads both files from there.
- Keep `B.csv` in the project directory to use the visualization page.
- The application is intended for demonstration and educational purposes. Its predictions should not be used as the sole basis for hiring or compensation decisions.

