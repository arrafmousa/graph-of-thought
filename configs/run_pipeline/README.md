# `run_pipeline` configs

Orchestrator: `src/main/run_pipeline` — schema: [`schema.json`](schema.json).

Generic runs of the synthetic demo workload: a random value in `[value_min, value_max]`
emitted once per step. They exercise the full reproducibility pipeline (run manifest,
telemetry stream, HTML dashboard) with no model or dataset dependency, so they run
anywhere on CPU.

| Config | Runs |
| --- | --- |
| [`demo/synthetic_workload_smoke.json`](demo/synthetic_workload_smoke.json) | 20 instant steps — the fast smoke test used locally and in CI: `python scripts/run.py --config configs/run_pipeline/demo/synthetic_workload_smoke.json` |
| [`demo/live_dashboard_preview.json`](demo/live_dashboard_preview.json) | 15 steps, one every 2 s, dashboard auto-refresh every 2 s — for watching the dashboard update live: `python scripts/dashboard_demo.py` |
