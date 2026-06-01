"""Optional: evaluate the RAG app with LangSmith (LLM-as-judge).

Requires: pip install langsmith openai
Set LANGSMITH_API_KEY and OPENAI_API_KEY in your .env first.

This mirrors the evaluation lecture: build a test dataset, run the app,
and grade outputs with correctness + concision metrics.
"""

import os

import openai
from dotenv import load_dotenv
from langsmith import Client, wrappers

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

client = Client()
oai = wrappers.wrap_openai(openai.OpenAI())

DATASET = "RAG Eval Demo"


def build_dataset() -> str:
    ds = client.create_dataset(DATASET)
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {"inputs": {"question": "What is LangChain?"},
             "outputs": {"answer": "A framework for building LLM applications"}},
            {"inputs": {"question": "What is a vector store?"},
             "outputs": {"answer": "A database that stores and searches embeddings"}},
        ],
    )
    return ds.id


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    prompt = (
        "Grade the predicted answer as CORRECT or INCORRECT.\n"
        f"Question: {inputs['question']}\n"
        f"Reference: {reference_outputs['answer']}\n"
        f"Predicted: {outputs['response']}\nGrade:"
    )
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "CORRECT" in resp.choices[0].message.content.upper()


def concision(outputs: dict, reference_outputs: dict) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])


if __name__ == "__main__":
    print("Building dataset:", build_dataset())
    print("Now wire your RAGPipeline.ask() into client.evaluate(...) to score it.")
