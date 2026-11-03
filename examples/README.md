# Example: match two small ontologies

This example aligns two small anatomy ontologies, using a model running locally
in [Ollama](https://ollama.com) as the judge. It needs no API key.

```bash
pip install kroma-om
ollama pull llama3.1
python match_ollama.py --model llama3.1
```

## Saved run

| Source | Target | Confidence (0-10) | Decided by |
|:--|:--|:--:|:--|
| heart | cardiac organ | 10 | Exact match, through the synonym "Heart" |
| lung | lungs | 8 | Model |
| organ | body organ | 8 | Model |

- **Model:** `llama3.1-32k`, Llama 3.1 8B with a larger context window, in Ollama
- **Temperature:** 0
- **KROMA settings:** defaults, with lexical retrieval
- **Data:** `anatomy_a.ttl` and `anatomy_b.owl`, three classes each, written for this demo
- **Output:** [`alignments.csv`](output/alignments.csv) and [`alignments.rdf`](output/alignments.rdf) (OAEI format)

Exact matches always get confidence 10 from KROMA; the other scores are the
model's own. This demo shows how KROMA works, not how well it scores. See the
[paper](https://doi.org/10.1007/978-3-032-09527-5_34) for benchmark results.
