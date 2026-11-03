"""Match two small ontologies, using a model running locally in Ollama as the judge.

Needs no API key, only KROMA and a running Ollama (https://ollama.com):

    ollama pull llama3.1
    python match_ollama.py --model llama3.1

Writes the alignments and the model's raw replies to the output folder.
"""
import argparse
from pathlib import Path

import requests

import kroma

HERE = Path(__file__).resolve().parent
OLLAMA_CHAT = "http://localhost:11434/api/chat"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--output", type=Path, default=HERE / "output", help="Folder for the results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replies = []

    def judge(system, user):
        """KROMA accepts any function that takes the two prompts and returns text."""
        response = requests.post(OLLAMA_CHAT, timeout=300, json={
            "model": args.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        response.raise_for_status()
        reply = response.json()["message"]["content"]
        source = user.split("Source Concept Names: ")[1].splitlines()[0]
        target = user.split("Target Concept Names: ")[1].splitlines()[0]
        replies.append(f"### {source} -> {target}\n{reply}\n")
        return reply

    result = kroma.match(HERE / "anatomy_a.ttl", HERE / "anatomy_b.owl", llm=judge, workers=1)

    result.save(args.output / "alignments.csv")
    result.save(args.output / "alignments.rdf")
    (args.output / "replies.txt").write_text("\n".join(replies), encoding="utf-8")

    print(f"{len(result)} alignments: {result.exact_matches} exact, "
          f"{result.pairs_judged} pairs judged in {result.llm_calls} model calls")
    for a in result.alignments:
        print(f"  {a.source_label} -> {a.target_label} ({a.method}, confidence {a.confidence})")


if __name__ == "__main__":
    main()
