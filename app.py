"""Command-line interface for the RAG pipeline.

Usage:
    python app.py ingest --pdf-dir data/pdf
    python app.py ask "What is the attention mechanism?"
    python app.py chat
"""

import argparse

from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()


def _print_answer(result: dict) -> None:
    print("\n" + "=" * 60)
    print(result["answer"])
    print("-" * 60)
    print(f"confidence: {result['confidence']}")
    for s in result["sources"]:
        print(f"  - {s['source']} (page {s['page']}, score {s['score']})")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Document QA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDFs into the vector store")
    p_ingest.add_argument("--pdf-dir", default="data/pdf")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")

    sub.add_parser("chat", help="Interactive Q&A loop")

    args = parser.parse_args()
    pipe = RAGPipeline()

    if args.command == "ingest":
        n = pipe.ingest(args.pdf_dir)
        print(f"Done. {n} chunks indexed.")
    elif args.command == "ask":
        _print_answer(pipe.ask(args.question))
    elif args.command == "chat":
        print("Type your question (or 'exit'):")
        while True:
            q = input("> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                _print_answer(pipe.ask(q))


if __name__ == "__main__":
    main()
