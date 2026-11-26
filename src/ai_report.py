"""Optional extended report: an OpenAI image-to-text model analyses the feature importance chart."""

import base64
import io
import json

import openai
import pandas as pd
from dotenv import dotenv_values
from matplotlib.figure import Figure
from openai import OpenAI

OPENAI_MODEL = "gpt-4o-mini"  # understands text and images; cheap
TOP_FEATURES_FOR_AI = 10      # only the top of the ranking is sent

SYSTEM_PROMPT = """You are a senior data analyst writing for business stakeholders \
who have no technical background.

You receive an image of a feature importance chart and a short automatic summary \
of the same analysis. Analyse the chart and write a short report in Markdown \
with these sections:
### Key findings
### What drives the target
### How reliable is this
### Questions worth investigating

Rules:
- Use ONLY the numbers and facts provided. If the chart and the text disagree, \
trust the exact numbers in the text. Never invent numbers, columns or causes.
- Explain technical terms (R², accuracy, F1, feature importance) in plain words.
- Feature importance shows what the model relies on, NOT cause and effect. Say so when relevant.
- If model quality is weak or suspicious, say clearly that conclusions are uncertain.
- Keep it under 300 words."""


class AIReportError(Exception):
    """Raised with a user-friendly message when the AI report cannot be generated."""


def get_key_from_env_file() -> str | None:
    """Return OPENAI_API_KEY from a local .env file, if there is one (for local testing)."""
    return dotenv_values(".env").get("OPENAI_API_KEY") or None


def render_importance_chart(importance: pd.DataFrame, metric: str) -> bytes:
    """Draw the feature importance chart as a PNG image (bytes) to send to the AI."""
    top = importance.head(TOP_FEATURES_FOR_AI).iloc[::-1]  # reversed: most important on top

    fig = Figure(figsize=(8, 0.45 * len(top) + 1.5))
    ax = fig.subplots()
    # Negative values are random noise (no effect), so they are drawn as zero
    bars = ax.barh(top["feature"].astype(str), top["importance"].clip(lower=0), color="#1f77b4")
    ax.bar_label(bars, labels=[f"{share:.1%}" for share in top["share"]], padding=3)
    ax.set_xlabel(f"Drop in {metric} when the feature is shuffled")
    ax.set_title("Feature importance (share of total in labels)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buffer = io.BytesIO()  # a "file" kept in memory
    fig.savefig(buffer, format="png", dpi=120)
    return buffer.getvalue()


def to_data_url(png_bytes: bytes) -> str:
    """Encode PNG bytes as a data URL, the format OpenAI accepts for images."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def build_payload(*, description: str, importance: pd.DataFrame, ignored_columns: list[str]) -> dict:
    """Collect ONLY aggregated results to send to OpenAI (never the user's data rows)."""
    top = importance.head(TOP_FEATURES_FOR_AI)
    return {
        "automatic_summary": description,
        "feature_ranking": [
            {"feature": str(feature), "share_of_importance": round(float(share), 3)}
            for feature, share in zip(top["feature"], top["share"])
        ],
        "columns_ignored_by_user": ignored_columns,
    }


def generate_ai_report(api_key: str, payload: dict, chart_png: bytes) -> str:
    """Send the chart image and the summary to OpenAI. Returns the report as Markdown."""
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.3,  # low = factual and consistent, high = more creative
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Summary of the analysis:\n"
                            + json.dumps(payload, ensure_ascii=False, indent=2),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(chart_png), "detail": "high"},
                        },
                    ],
                },
            ],
        )
    except openai.AuthenticationError as err:
        raise AIReportError("The API key is not valid. Please check it and try again.") from err
    except openai.RateLimitError as err:
        raise AIReportError(
            "OpenAI refused the request: rate limit reached or no credit left on this account."
        ) from err
    except openai.APIConnectionError as err:
        raise AIReportError("Could not connect to OpenAI. Check your internet connection.") from err
    except openai.OpenAIError as err:
        raise AIReportError(f"OpenAI returned an error: {err}") from err

    return response.choices[0].message.content or ""
