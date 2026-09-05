# `train_pipeline` configs

Orchestrator: `src/main/train_pipeline` — schema: [`schema.json`](schema.json).
Entrypoint: `python scripts/train.py --config <config>`.

Sequence-classification fine-tuning runs. Separate from the reasoning-graph pipeline; kept
as the reference workload for the training dashboard template. Models and datasets come
from the Hugging Face Hub, so these run on the remote GPU runtime (`AGENTS.md` §31).

| Config | Runs |
| --- | --- |
| [`sst2/distilbert_sst2_finetune.json`](sst2/distilbert_sst2_finetune.json) | Fine-tunes `distilbert-base-uncased` for binary sentiment on GLUE SST-2 (100 train / 100 eval samples, 3 epochs, batch 16, lr 5e-5, fp16). |
