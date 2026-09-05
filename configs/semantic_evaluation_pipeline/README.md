# `semantic_evaluation_pipeline` configs

Orchestrator: `src/main/semantic_evaluation_pipeline` — schema: [`schema.json`](schema.json).
Entrypoint: `python scripts/evaluate_merges.py --config <config>`.

The run has two stages so the expensive GPU work is preserved even if the Azure judge
fails: `--stage generate` produces traces + all graphs and persists the judge inputs;
`--stage judge --run-dir output/<run_id>` resumes judging + reporting on that run without
regenerating. `--stage all` (default) does both in one process.

This pipeline randomly samples configured Hugging Face math datasets using explicit
per-dataset seeds, generates complete reasoning traces once, sweeps merge heuristics
and thresholds offline, and saves every inferred graph. An Azure OpenAI Batch judge
then classifies every unique accepted merge pair from the two reasoning prefixes at
the merge point. Only merges whose join point is after the first token
(`tuning.min_join_token_index`) are judged, so trivial shared-root joins are excluded.
Final answers are withheld from the judge and compared separately.

Outputs include all trace and graph JSON, Azure batch request/status/result files,
pair judgments, continuation-agreement statistics, whole-graph quality scores, a
static summary report, and selected per-graph HTML reports for manual inspection.

For each accepted merge, `same_final_answer_probability` is the empirical probability
that the two completed source chains end in the same normalized answer. This is a
continuation-agreement proxy, not proof that their full future derivations are identical.
The whole-graph score first averages the configured semantic-judge and continuation
weights into merge precision, then combines that precision with node reduction using
the configured F-score beta. Keep the component metrics visible when interpreting the
ranking: node reduction is only a coverage proxy because no ground-truth merge recall
labels exist yet.

The Azure key is read only from the environment variable named by
`judge.api_key_environment_variable`; use `.env` locally and Colab Secrets remotely.

| Config | Runs |
| --- | --- |
| [`demo/synthetic_cpu_semantic_evaluation.json`](demo/synthetic_cpu_semantic_evaluation.json) | Small deterministic CPU test using synthetic data and judge. |
| [`math/llama1b_five_dataset_pilot.json`](math/llama1b_five_dataset_pilot.json) | Cheap pilot: 5 seeded questions each from the same five datasets; identical model/judge/merge settings; use to validate the pipeline before the full run. |
| [`math/llama1b_five_dataset_azure_batch.json`](math/llama1b_five_dataset_azure_batch.json) | 20 seeded questions each from GSM8K, MATH-500, AIME 2025, SVAMP, and ASDiv; six Llama chains per question; 15 merge settings; Azure OpenAI `gpt-5.1` Batch judge. |