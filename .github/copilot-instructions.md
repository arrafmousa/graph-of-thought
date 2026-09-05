# Copilot / Agent Instructions

**Before planning, generating, editing, refactoring, testing, or reviewing any
code in this repository, you MUST read [`AGENTS.md`](../AGENTS.md) and comply
with it.** `AGENTS.md` is the authoritative, repository-wide engineering
contract and takes precedence over general conventions.

Key invariants (see `AGENTS.md` for the full, binding text):

- **Dependency direction is strict.** `src/main/*` may depend on `src/libs/*`;
  `src/libs/*` must never import `src/main/*` or a sibling library object.
  Compose libraries only in orchestrators under `src/main/`.
- **One project-owned class per file.** `__init__.py` exposes the package API.
- **Configuration is explicit.** No hidden project-owned default values. Required
  fields live in `configs/` and are validated against a versioned schema; missing
  fields must fail loudly (no silent fallbacks).
- **`configs/` mirrors `src/main/`.** One config folder per orchestrator, same name:
  `configs/<orchestrator>/<experiment>/<config>.json`, with the orchestrator's schema
  colocated as `configs/<orchestrator>/schema.json` and a `README.md` in every config
  folder (e.g. `configs/run_pipeline/demo/synthetic_workload_smoke.json`). No config
  files directly in `configs/`. See `AGENTS.md` §7.5.
- **Every run is reproducible.** Runs write `output/<run_id>/run_manifest.json`,
  `telemetry.jsonl`, and a static `dashboard.html`. A run must not do substantive
  work before a valid manifest and telemetry stream exist.
- **Secrets never enter artifacts.** Use `.env` (git-ignored); document names in
  `.env.example`.
- **GPU runs are remote (Colab).** The local machine has no GPU. Commit + push to
  GitHub, then run via `notebooks/colab_run.ipynb`. On the remote there is no local
  data access — datasets and models come from the Hugging Face Hub (explicit repo
  IDs/revisions in `configs/`), and `output/<run_id>/` is zipped and downloaded back.
  See `AGENTS.md` §31.
- **Enforcement is executable.** Run `python scripts/validate_repo.py` and the
  test suite before declaring a task done; a task is incomplete if the repository
  violates `AGENTS.md`.

Validation entrypoints:

```bash
python scripts/validate_repo.py          # architecture / policy checks
python -m pytest -q                       # unit + architecture tests
python scripts/run.py --config configs/run_pipeline/demo/synthetic_workload_smoke.json
python scripts/validate_run.py output/<run_id>
```
