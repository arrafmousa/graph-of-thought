# Configuration

`configs/` mirrors `src/main/`: one folder per orchestrator, with the same name, so the
path tells you which entrypoint consumes the config (`AGENTS.md` §7.5).

```text
configs/<orchestrator>/schema.json          # schema for that orchestrator's configs
configs/<orchestrator>/<experiment>/<config>.json
```

| Folder | Orchestrator / entrypoint | What it runs |
| --- | --- | --- |
| [`run_pipeline/`](run_pipeline/README.md) | `src/main/run_pipeline` — `scripts/run.py`, `scripts/dashboard_demo.py` | Synthetic step workload that exercises the manifest/telemetry/dashboard pipeline. |
| [`tuning_pipeline/`](tuning_pipeline/README.md) | `src/main/tuning_pipeline` — `scripts/tune_graph.py` | Phase 1 — sweep merge heuristics × thresholds to pick the Phase 2 settings. |
| [`graph_pipeline/`](graph_pipeline/README.md) | `src/main/graph_pipeline` — `scripts/generate_graphs.py` | Phase 2 — reasoning-graph generation with the chosen heuristic + threshold. |
| [`semantic_evaluation_pipeline/`](semantic_evaluation_pipeline/README.md) | `src/main/semantic_evaluation_pipeline` — `scripts/evaluate_merges.py` | Five-dataset graph sweep, Azure OpenAI Batch pair judging, continuation statistics, and whole-graph quality reports. |
| [`train_pipeline/`](train_pipeline/README.md) | `src/main/train_pipeline` — `scripts/train.py` | Sequence-classification fine-tuning (GPU, remote). |

`demo/` experiments are CPU-only and dependency-light (safe to run locally); the others
(`gsm8k/`, `math/`, `sst2/`) pull models and datasets from the Hugging Face Hub and are
meant for the remote GPU runtime (`AGENTS.md` §31). Semantic evaluation additionally
reads `AZURE_OPENAI_API_KEY` from `.env` locally or Colab Secrets remotely.

Config file names describe what they run (model / dataset / workload). No configuration
file may sit directly in `configs/`.
