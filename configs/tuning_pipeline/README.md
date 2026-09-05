# `tuning_pipeline` configs

Orchestrator: `src/main/tuning_pipeline` — schema: [`schema.json`](schema.json).
Entrypoint: `python scripts/tune_graph.py --config <config>`.

Phase 1 of the reasoning-graph pipeline: sweep `tuning.heuristics` × `tuning.thresholds`
over a small question sample and emit one dashboard per combination plus a comparison
dashboard and `tuning_summary.json`. Inspect the high-similarity and borderline merge
samples to pick the heuristic + threshold to set in the Phase 2 config
([`../graph_pipeline/README.md`](../graph_pipeline/README.md)).

| Config | Runs |
| --- | --- |
| [`demo/synthetic_cpu_merge_sweep.json`](demo/synthetic_cpu_merge_sweep.json) | Synthetic model + arithmetic questions, 8 questions, 3 heuristics × 3 thresholds on CPU — local smoke test of the sweep and its dashboards. |
| [`gsm8k/llama1b_gsm8k_merge_sweep.json`](gsm8k/llama1b_gsm8k_merge_sweep.json) | `meta-llama/Llama-3.2-1B-Instruct` on 25 GSM8K questions × 10 chains, 3 heuristics × 5 thresholds, fp16 on CUDA — the real sweep, run on the remote GPU runtime. |
