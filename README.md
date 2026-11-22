# Feature Importance Analyzer

A Streamlit app that:

1. loads a CSV file,
2. lets the user choose the target column,
3. detects whether it is a classification or regression problem,
4. trains the best model (PyCaret),
5. shows the most important features and describes the results in plain language.

## Tech stack

- Python, Streamlit, PyCaret

## Getting started

```bash
conda activate env_ds_ai_s3
pip install -r requirements.txt
streamlit run app.py
```
