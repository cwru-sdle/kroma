# Example

Match two small anatomy ontologies with a local [Ollama](https://ollama.com)
model. No API key needed.

```bash
pip install kroma-om
ollama pull llama3.1
python match_ollama.py --model llama3.1
```

## Results

| Source | Target | Confidence (0-10) | Decided by |
|:--|:--|:--:|:--|
| heart | cardiac organ | 10 | Exact match on a synonym |
| lung | lungs | 8 | Model |
| organ | body organ | 8 | Model |

Run with `llama3.1-32k` (Llama 3.1 8B) at temperature 0 and KROMA's default
settings, on a small dataset written for this demo. For benchmark results, see
the [paper](https://doi.org/10.1007/978-3-032-09527-5_34).

Saved output: [`alignments.csv`](output/alignments.csv),
[`alignments.rdf`](output/alignments.rdf).
