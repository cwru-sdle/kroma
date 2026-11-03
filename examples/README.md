# Example: match two small ontologies

This example aligns two ontologies of five anatomy classes each, using a model
running locally in [Ollama](https://ollama.com) as the judge. It needs no API key.

```bash
pip install kroma-om
ollama pull llama3.1
python match_ollama.py --model llama3.1
```

The results go to `output/`: `alignments.csv`, `alignments.rdf` (OAEI format)
and `replies.txt` (the model's raw replies).

## Saved run

The files in `output/` come from `--model llama3.1-32k`, which is Llama 3.1 8B
with a larger context window, at temperature 0. Two runs gave identical files.
Another model will give different answers.

| Source | Expected target | KROMA's answer | Result |
|:--|:--|:--|:--|
| heart | cardiac organ | cardiac organ | Correct, an exact match through the synonym "Heart" |
| organ | body organ | body organ | Correct |
| lung | lungs | lungs | Correct |
| heart valve | cardiac valve | cardiac organ | Wrong |
| kidney | renal organ | none | Missed |

Three of the five expected matches were found. This small example shows how
KROMA works, not how well it scores; see the
[paper](https://doi.org/10.1007/978-3-032-09527-5_34) for that.

- **heart valve:** the model accepted cardiac organ, the first candidate it
  judged, so KROMA never asked about cardiac valve.
- **kidney:** the default lexical retrieval never sent this pair to the model.
  Names this different need a neural embedder.
- **Confidence 0:** this model writes "I am 8 out of 10 confident" instead of
  the `<confidence>` tag, and KROMA counts a missing tag as 0. A
  `min_confidence` above 0 would drop these matches.
