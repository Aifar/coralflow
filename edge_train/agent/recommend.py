"""Web-based dataset recommendations using LLM knowledge.

Leverages the LLM's training knowledge of public ML datasets to recommend
datasets from HuggingFace, Kaggle, UCI ML Repository, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_train.agent.llm import LLMClient


def recommend_datasets(task: str, llm: LLMClient | None = None) -> str:
    """Recommend public datasets for a given ML task.

    Sends the task description to the LLM, which draws on its training
    knowledge of major public dataset repositories to return a curated
    list with descriptions, sizes, and download URLs.

    Parameters:
        task: Natural language task description, e.g. 'spam detection for email'.
        llm: LLM client for generating recommendations.

    Returns:
        Formatted list of dataset recommendations with download instructions.
    """
    if not task.strip():
        return "Please provide a task description, e.g. 'spam detection for email'."

    prompt = f"""The user needs a dataset for this ML task: "{task}"

Recommend 3-5 public datasets from well-known repositories (HuggingFace Datasets, Kaggle, UCI ML Repository, TensorFlow Datasets, etc.).

For each dataset provide:
1. **Name** — short, recognizable
2. **Source** — e.g. Kaggle, HuggingFace, UCI
3. **Description** — 1 sentence about what it contains
4. **Size** — approximate rows and classes
5. **URL** — direct link to the dataset page
6. **Download** — a concrete curl or wget command if the dataset is directly downloadable

Focus on:
- Freely available, commonly used datasets
- Realistic sizes for edge ML (hundreds to low tens of thousands of rows)
- Text classification datasets (since CoralFlow supports text modality)

Format as a numbered list. Be specific with URLs and commands — no placeholders."""

    if llm:
        try:
            resp = llm.chat([{"role": "user", "content": prompt}])
            if resp.content:
                return resp.content
        except Exception:
            pass

    return (
        f"Could not generate specific recommendations for '{task}'.\n\n"
        "Try these resources directly:\n"
        "  • HuggingFace Datasets: https://huggingface.co/datasets\n"
        "  • Kaggle Datasets: https://kaggle.com/datasets\n"
        "  • UCI ML Repository: https://archive.ics.uci.edu\n"
        "  • TensorFlow Datasets: https://tensorflow.org/datasets"
    )
