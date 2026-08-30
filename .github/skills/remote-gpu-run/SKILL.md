---
name: remote-gpu-run
description: 'Run this repository''s GPU workloads on Google Colab (the local machine has no GPU). Use when: training a model, GPU inference, benchmarking, "run on GPU", "train remotely", "run on Colab", "remote run", "push and run", "run on T4/A100", GPU experiment, or any run that needs CUDA. Datasets and models come from the Hugging Face Hub only; results are downloaded back into output/. See AGENTS.md section 31.'
argument-hint: 'Optional: config to run on the remote (e.g. configs/train_sst2.json)'
---

# Remote GPU Run (Colab)

Execute GPU workloads for this repo on Google Colab. Development is local/CPU; GPU
execution is remote. This is the authoritative flow from `AGENTS.md` section 31.

## When to use
- Training, GPU inference, or benchmarking that needs a GPU (T4/A100/…).
- Triggers: "run on GPU", "train the model", "run on Colab", "remote run".
- For CPU-only local iteration use the `local-cpu-dev` skill instead.

## Hard requirements (section 31)
- **No local data on the remote.** All datasets and models MUST load from the
  Hugging Face Hub, addressed by explicit repo IDs + revisions in `configs/`.
- **Repo must be reachable from Colab**: either public, or a `GH_TOKEN` Colab secret.
- **Secrets** (e.g. `HF_TOKEN`) come from Colab secrets/env, never committed.

## Procedure
1. Validate locally first — both must pass:
   ```
   python scripts/validate_repo.py
   python -m pytest -q
   ```
2. Ensure the run config in `configs/` names explicit HF model/dataset IDs + revisions
   (no hidden defaults). These get recorded in the run manifest.
3. Commit and push (the remote runs the exact pushed commit):
   ```
   git add -A
   git commit -m "<message>"
   git push
   ```
4. Open the notebook in Colab. Fastest is the **Open in Colab** badge at the top of
   `notebooks/colab_run.ipynb`, or this direct link (pins the `master` branch):
   ```
   https://colab.research.google.com/github/arrafmousa/graph-of-thought/blob/master/notebooks/colab_run.ipynb
   ```
   (Alternatively: File → Open notebook → GitHub → `arrafmousa/graph-of-thought`, branch
   `master`.) Then set **Runtime → Change runtime type → GPU**.
5. Run the cells in order: clone → (install deps + HF auth) →
   `python scripts/run.py --config <config>` → the final cell zips and downloads
   `output/<run_id>/`.
6. Copy the downloaded `output/<run_id>/` into this repo's git-ignored `output/`,
   then validate it locally:
   ```
   python scripts/validate_run.py output/<run_id>
   ```

## Outputs & model weights (zip and download)
All run artifacts — output files, folders, checkpoints, and **model weights** — are
written to `output/<run_id>/` on the Colab runtime's **local, ephemeral** disk. When
the session ends they are lost, so the run directory MUST be **zipped and downloaded**
before you disconnect:
- Ensure training code writes weights/checkpoints under `output/<run_id>/`
  (e.g. `output/<run_id>/checkpoints/`, `output/<run_id>/artifacts/`), not to arbitrary
  paths, so they land inside the archive.
- The notebook's final cell zips `output/<run_id>/` and triggers a browser download;
  copy the unzipped folder into this repo's git-ignored `output/`.

## Notes
- Uncommitted local changes do NOT run remotely — always push first.
- Manifest, telemetry, and dashboard are produced identically on the remote; keep them.
- The GPU is only exercised by code that uses it (e.g. a training orchestrator);
  the demo workload is CPU-only.
