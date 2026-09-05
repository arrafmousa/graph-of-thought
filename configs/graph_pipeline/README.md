# `graph_pipeline` configs

Orchestrator: `src/main/graph_pipeline` — schema: [`schema.json`](schema.json).
Entrypoint: `python scripts/generate_graphs.py --config <config>`.

Phase 2 of the reasoning-graph pipeline: sample reasoning chains per question, then
consolidate them into a reasoning graph using a **single** merge heuristic + threshold
(chosen from a Phase 1 sweep, see
[`../tuning_pipeline/README.md`](../tuning_pipeline/README.md)). Writes traces,
consolidated graphs, and a sampled per-question HTML graph report.

| Config | Runs |
| --- | --- |
| [`demo/synthetic_cpu_graphs.json`](demo/synthetic_cpu_graphs.json) | Synthetic token model + synthetic arithmetic questions, 3 questions × 6 chains on CPU — local smoke test, no downloads. |
| [`gsm8k/llama1b_gsm8k_graphs.json`](gsm8k/llama1b_gsm8k_graphs.json) | `meta-llama/Llama-3.2-1B-Instruct` on 100 GSM8K test questions × 6 chains, fp16 on CUDA — the real run, executed on the remote GPU runtime. |
