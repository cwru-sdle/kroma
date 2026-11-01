<div align="center">

# KROMA

**Ontology matching with knowledge retrieval and large language models**

[![PyPI](https://img.shields.io/pypi/v/kroma-om)](https://pypi.org/project/kroma-om/)
[![Docs](https://readthedocs.org/projects/kroma/badge/?version=latest)](https://kroma.readthedocs.io/en/latest/)
[![ISWC 2025](https://img.shields.io/badge/ISWC-2025-b31b1b)](https://doi.org/10.1007/978-3-032-09527-5_34)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](https://github.com/cwru-sdle/kroma/blob/main/LICENSE)

[**Documentation**](https://kroma.readthedocs.io/en/latest/) ·
[Quick start](#quick-start) ·
[Results](#results) ·
[Paper](https://doi.org/10.1007/978-3-032-09527-5_34) ·
[Citation](#citation)

</div>

KROMA finds the concepts that two ontologies share. It retrieves likely
matches for each concept, accepts identical names directly, and asks a large
language model about the rest, showing it each concept's labels, synonyms,
parents and children.

## Quick start

```bash
pip install kroma-om openai
export OPENAI_API_KEY=sk-...

kroma match mouse.owl human.owl -o alignments.csv
```

Or from Python:

```python
import kroma

result = kroma.match("mouse.owl", "human.owl", model="gpt-4o-mini")
result.save("alignments.csv")    # .csv, .tsv, or .rdf (OAEI format)
```

KROMA reads any RDF format (`.owl`, `.ttl`, `.rdf`, …) and works with OpenAI,
Together AI, or any function that takes a prompt and returns text. See the
[documentation](https://kroma.readthedocs.io/en/latest/) for all
options.

## Results

F1 (%) on six OAEI tracks, from the [ISWC 2025 paper](https://doi.org/10.1007/978-3-032-09527-5_34):

| Track | KROMA | Best baseline |
|:--|:--:|:--:|
| Mouse–Human | **94.94** | 92.20 |
| NCIT–DOID | **98.63** | 94.80 |
| Nell–DBpedia | **97.08** | 96.00 |
| YAGO–Wikidata | **95.54** | 92.19 |
| ENVO–SWEET | **93.18** | 83.70 |
| MI–MatOnto | **61.25** | 32.97 |

Ablations, other models and cost are in the
[full paper](https://github.com/cwru-sdle/kroma/blob/main/full.pdf).

## Reproducing the paper

```bash
git clone https://github.com/cwru-sdle/kroma.git && cd kroma
./setup.sh && source .venv/bin/activate
uv pip install ".[models]"

kroma run --method_config kroma_scibert_envo_sweet \
          --llm meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
```

Datasets, configs and flags are described in
[Benchmark experiments](https://kroma.readthedocs.io/en/latest/usage.html).

## Citation

```bibtex
@inproceedings{nguyen_2025_kroma,
  title     = {KROMA: Ontology Matching with Knowledge Retrieval and Large Language Models},
  author    = {Lam Nguyen and Erika Barcelos and Roger French and Yinghui Wu},
  booktitle = {The Semantic Web -- ISWC 2025: 24th International Semantic Web Conference, Proceedings, Part I},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  pages     = {629--649},
  year      = {2025},
  doi       = {10.1007/978-3-032-09527-5_34}
}
```

## License

[CC BY 4.0](https://github.com/cwru-sdle/kroma/blob/main/LICENSE) ·
Developed at the [SDLE Research Center](https://github.com/cwru-sdle),
Case Western Reserve University.
