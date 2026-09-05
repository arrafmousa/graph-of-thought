---
name: local-cpu-dev
description: 'Develop and run this repository locally on CPU (no GPU on this machine). Use when: local development, "run locally", "run the pipeline", "run tests", "validate repo", quick iteration, CPU-only smoke test, debugging the orchestrator/libraries, editing configs, or preparing changes before a remote GPU run. Parallel to the remote-gpu-run skill. Follows AGENTS.md.'
argument-hint: 'Optional: config to run locally (default configs/run_pipeline/demo/synthetic_workload_smoke.json)'
---

# Local CPU Development

Iterate on this repository locally on CPU. GPU workloads run remotely via the
`remote-gpu-run` skill; everything else (code, configs, tests, CPU smoke tests)
happens here.

## When to use
- Editing library objects, orchestrators, configs, or tests.
- CPU-only smoke tests of the pipeline.
- Triggers: "run locally", "run tests", "validate repo", "quick iteration".

## Procedure
1. Make changes following `AGENTS.md` — one project-owned class per file, libraries
   under `src/libs/*` isolated, composition only in `src/main/*`, configuration
   explicit (no hidden defaults).
2. Validate architecture/policy:
   ```
   python scripts/validate_repo.py
   ```
3. Run the test suite:
   ```
   python -m pytest -q
   ```
4. Run the pipeline locally:
   ```
   python scripts/run.py --config configs/run_pipeline/demo/synthetic_workload_smoke.json
   ```
   - Terminal progress (ASCII tables + sparkline graph) prints when
     `run.terminal_progress` is true in the config.
   - The live HTML dashboard is written to `output/<run_id>/dashboard.html`.
5. Validate the produced run:
   ```
   python scripts/validate_run.py output/<run_id>
   ```

## Constraints
- **CPU only.** Do not depend on CUDA/GPU here; anything GPU-only belongs in the
  `remote-gpu-run` skill.
- Keep the validator and tests green before committing.
- Secrets live in `.env` (git-ignored); never commit them.

## Handing off to a GPU
When a change needs a GPU (e.g. training), switch to the `remote-gpu-run` skill:
commit, push, run `notebooks/colab_run.ipynb` on Colab, and download the results
back into `output/`.
