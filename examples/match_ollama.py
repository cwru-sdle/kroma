"""Match two small ontologies, using a model running locally in Ollama as the judge.

Needs no API key, only KROMA and a running Ollama (https://ollama.com):

    ollama pull llama3.1
    python match_ollama.py --model llama3.1

Writes the alignments to the output folder.
"""
import argparse
from pathlib import Path

import requests

import kroma

HERE = Path(__file__).resolve().parent
OLLAMA_CHAT = "http://localhost:11434/api/chat"
# Small local models tend to state their confidence in words, which KROMA reads as 0.
REPLY_FORMAT = "\n\nReply with only the two tags: <answer>Yes or No</answer><confidence>0-10</confidence>"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--output", type=Path, default=HERE / "output", help="Folder for the results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def judge(system, user):
        """KROMA accepts any function that takes the two prompts and returns text."""
        response = requests.post(OLLAMA_CHAT, timeout=300, json={
            "model": args.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + REPLY_FORMAT},
            ],
        })
        response.raise_for_status()
        return response.json()["message"]["content"]

    result = kroma.match(HERE / "anatomy_a.ttl", HERE / "anatomy_b.owl", llm=judge, workers=1)

    result.save(args.output / "alignments.csv")
    result.save(args.output / "alignments.rdf")

    print(f"{len(result)} alignments, {result.pairs_judged} pairs judged by {args.model}")
    for a in result.alignments:
        print(f"  {a.source_label} -> {a.target_label} (confidence {a.confidence}, {a.method})")


if __name__ == "__main__":
    main()
