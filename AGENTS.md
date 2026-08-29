# AGENTS.md — Repository Engineering Commandments

This file is the normative engineering contract for every coding agent working in this repository.

The agent MUST read this file before planning, editing, generating, refactoring, testing, or reviewing code.

These rules are cumulative and repository-wide. They are not merely guidelines for newly written code.

If this file changes, the agent MUST:
1. treat the newest version as authoritative;
2. inspect the existing repository for violations of the changed rules;
3. refactor affected existing code until the repository is compliant;
4. update tests, validators, hooks, CI checks, configuration schemas, telemetry, and documentation as needed;
5. complete the requested task only after the repository passes the current policy checks.

A task is NOT complete if the requested feature works but the repository violates this file.

---

## 1. Core Engineering Principles

1. Prefer explicit behavior over implicit behavior.
2. Prefer isolation over coupling.
3. Prefer composition in orchestrators over cross-library dependencies.
4. Prefer configuration over hard-coded behavior.
5. Prefer reproducible runs over ad-hoc execution.
6. Prefer machine-enforced invariants over conventions that rely on memory.
7. Prefer observable execution over opaque execution.
8. Prefer repository-wide consistency over local convenience.
9. Do not preserve legacy structure merely to minimize the diff if it violates the current commandments.
10. Never silently weaken, bypass, or disable a policy check to make a task pass.

---

## 2. Canonical Repository Structure

The repository MUST follow this conceptual structure:

```text
.
├── AGENTS.md
├── .env
├── .env.example
├── .gitignore
├── configs/
│   └── ...
├── output/
│   └── <run_id>/
│       ├── run_manifest.json
│       ├── telemetry.jsonl
│       ├── dashboard.html
│       ├── logs/
│       ├── artifacts/
│       └── ...
├── scripts/
│   ├── validate_repo.py
│   ├── validate_run.py
│   ├── render_dashboard.py
│   └── ...
├── src/
│   ├── libs/
│   │   ├── <object_1>/
│   │   │   ├── ...
│   │   ├── <object_2>/
│   │   │   ├── ...
│   │   └── ...
│   └── main/
│       ├── <orchestrator_1>/
│       │   ├── ...
│       ├── <orchestrator_2>/
│       │   ├── ...
│       └── ...
└── tests/
    ├── libs/
    │   ├── <object_1>/
    │   ├── <object_2>/
    │   └── ...
    └── main/
        ├── <orchestrator_1>/
        ├── <orchestrator_2>/
        └── ...
```

Exact filenames inside each component may vary, but the dependency and isolation rules below may not.

---

## 3. Dependency Direction Is Strict

The allowed dependency direction is:

```text
src/main/*  --->  src/libs/*
```

The following are forbidden:

```text
src/libs/*  -X->  src/main/*
src/libs/object_1  -X->  src/libs/object_2
src/libs/object_2  -X->  src/libs/object_1
```

### 3.1 Library isolation

Each direct child of `src/libs/` is an independently isolated library object.

A library object MUST NOT import:
- any module from `src/main/`;
- any sibling library object under `src/libs/`;
- implementation details belonging to another library object.

A library object may depend on:
- the Python standard library;
- third-party packages declared by the project;
- modules inside its own library object.

If two library objects need to cooperate, they MUST be composed by an orchestrator under `src/main/`.

Do not create hidden dependency paths through utility modules, global registries, import side effects, or shared mutable state.

### 3.2 Orchestrators

Only code under `src/main/` may compose multiple library objects.

Each orchestrator MUST have a clear responsibility and MUST not become a dumping ground for reusable implementation logic.

Reusable capability belongs in a properly isolated library object. Composition and workflow sequencing belong in `src/main/`.

---

## 4. One Class Per File

Every source file that defines a class MUST define exactly one project-owned class.

Do not place multiple project-owned classes in one file for convenience.

This applies to:
- abstract base classes;
- protocols/interfaces;
- concrete implementations;
- adapters;
- orchestrators;
- configuration models;
- telemetry models;
- report models;
- exceptions when represented as classes.

A module may contain supporting functions/constants only when they are tightly related to the single class in that file and do not represent a second class-level responsibility.

`__init__.py` files should primarily expose package APIs and SHOULD NOT contain class implementations.

---

## 5. Multiple Implementations Require an Explicit Contract

When one logical capability has multiple interchangeable implementations, the capability MUST be represented by an explicit architectural contract.

Use an appropriate language mechanism such as:
- an abstract base class;
- a protocol/interface;
- a deliberately designed base class.

Each implementation:
- MUST live in its own file;
- MUST implement or inherit from the shared contract;
- MUST expose behavior through that contract rather than through implementation-specific assumptions in the orchestrator.

The orchestrator MUST depend on the contract, not on accidental details of a particular implementation.

Do not create inheritance merely to share unrelated utility code. Inheritance/interfaces are for a genuine shared behavioral contract.

---

## 6. Tests Mirror the Source Structure

Tests MUST mirror the production structure.

Examples:

```text
src/libs/object_1/...      -> tests/libs/object_1/...
src/libs/object_2/...      -> tests/libs/object_2/...
src/main/orchestrator_1/... -> tests/main/orchestrator_1/...
```

Every meaningful behavior change MUST include or update tests.

At minimum, the repository MUST maintain tests/checks for:
- library isolation;
- forbidden import directions;
- one-class-per-file;
- required configuration;
- run-manifest validity;
- telemetry validity;
- dashboard generation;
- task-specific behavior.

A feature is incomplete if its tests do not reflect the architecture that implements it.

---

## 7. Configuration Is Explicit

### 7.1 No project-owned default values

Project-owned behavior MUST NOT depend on hidden or implicit default values.

User-controlled, experiment-controlled, model-controlled, training-controlled, inference-controlled, evaluation-controlled, reporting-controlled, and orchestration-controlled behavior MUST be explicitly represented in configuration.

Do not add behavior-changing defaults to:
- function arguments;
- constructors;
- CLI arguments;
- configuration models;
- orchestration code;
- training code;
- inference code;
- evaluation code;
- telemetry/reporting code.

If the repository owns the behavior, the caller or configuration must specify it explicitly.

### 7.2 Third-party defaults

A third-party library's untouched default behavior does NOT need to be redundantly copied into project configuration.

However, if this repository intentionally changes, overrides, or relies on a non-default value for a third-party parameter, that parameter MUST:
1. be explicit in project configuration;
2. be passed explicitly to the third-party API;
3. be recorded in the run manifest.

Example:

BAD:
```python
trainer = Trainer(model=model, lr=3e-5)
```

if `3e-5` is an experiment choice hidden in source code.

GOOD:
```python
trainer = Trainer(model=model, lr=config.training.learning_rate)
```

with `training.learning_rate` explicitly required by the run configuration.

### 7.3 No configuration hidden in source code

Do not hard-code experiment/model/runtime choices in implementation files.

Configuration SHOULD live under `configs/` in a structured format such as YAML, JSON, or TOML and MUST be validated against an explicit schema.

All required project-owned fields MUST fail validation when absent.

Do not silently fill missing fields.

### 7.4 Effective configuration

Every run MUST preserve the exact effective configuration used for that run.

The effective configuration MUST be included directly in, or referenced immutably by, `run_manifest.json`.

It must be possible to determine exactly which explicit values were used without reading source code or guessing package defaults that the project intentionally overrode.

---

## 8. Environment Variables and Secrets

A local `.env` file MUST be supported and expected for local execution.

`.env` MUST be ignored by Git.

A version-controlled `.env.example` MUST document required environment variable names without containing real secrets.

Never write secrets, API keys, access tokens, passwords, private credentials, or sensitive `.env` values into:
- `run_manifest.json`;
- telemetry;
- logs;
- dashboards;
- committed configuration;
- test snapshots.

If an environment variable affects reproducibility but contains a secret, record only safe metadata such as the variable name, provider, non-secret identifier, or a redacted value.

---

## 9. Output Directory

All run-generated artifacts MUST live under:

```text
output/<run_id>/
```

`output/` MUST be in `.gitignore`.

Do not scatter run artifacts across the repository.

A run-specific directory SHOULD contain only artifacts associated with that run.

A recommended layout is:

```text
output/<run_id>/
├── run_manifest.json
├── telemetry.jsonl
├── dashboard.html
├── logs/
├── artifacts/
├── checkpoints/
├── predictions/
└── metrics/
```

Only create subdirectories relevant to the task.

---

## 10. Every Run Requires a Reproducibility Manifest

Every executable run that produces reportable output MUST create:

```text
output/<run_id>/run_manifest.json
```

The manifest is mandatory.

A run MUST fail before doing substantive work if a valid manifest cannot be created.

A run MUST fail final validation if the manifest is missing, malformed, incomplete, or inconsistent with the run.

### 10.1 Required manifest content

The exact schema may evolve, but the manifest MUST capture enough information to reproduce and audit the run.

At minimum it MUST contain:

```json
{
  "run_id": "...",
  "timestamp_start_utc": "...",
  "timestamp_end_utc": "...",
  "status": "...",
  "entrypoint": "...",
  "command": "...",
  "working_directory": "...",
  "git": {
    "commit": "...",
    "branch": "...",
    "dirty": false
  },
  "environment": {
    "python_version": "...",
    "platform": "...",
    "relevant_package_versions": {}
  },
  "configuration": {},
  "configuration_hash": "...",
  "inputs": {},
  "outputs": {},
  "model": {},
  "randomness": {
    "seeds": {}
  },
  "telemetry": {
    "path": "telemetry.jsonl",
    "schema_version": "..."
  },
  "dashboard": {
    "template": "...",
    "path": "dashboard.html"
  },
  "duration_seconds": 0,
  "errors": []
}
```

Fields that are not applicable may be represented explicitly according to the manifest schema, but they must not be silently invented.

### 10.2 Manifest lifecycle

The runner MUST:

1. validate configuration;
2. allocate a unique `run_id`;
3. create `output/<run_id>/`;
4. create the initial manifest;
5. create/initialize telemetry;
6. begin substantive execution;
7. update manifest status during/following execution;
8. record final outputs, duration, errors, and completion status;
9. validate the final manifest;
10. generate/update the HTML dashboard.

If steps 1–5 fail, substantive execution MUST NOT begin.

If final manifest validation fails, the run is considered failed even if the main computation completed.

### 10.3 Reproducible invocation

The manifest MUST preserve the exact entrypoint and invocation parameters.

The repository MUST provide a clear way to reconstruct the invocation from the manifest.

Avoid free-form prose as the only record of execution parameters.

---

## 11. Telemetry Is a First-Class Output

Every reportable action MUST emit structured telemetry.

Telemetry MUST be machine-readable and append-friendly. The default architectural choice SHOULD be:

```text
output/<run_id>/telemetry.jsonl
```

Each line represents one telemetry event.

The telemetry schema MUST be versioned.

### 11.1 Common event fields

Events SHOULD use a common envelope such as:

```json
{
  "schema_version": "...",
  "run_id": "...",
  "timestamp_utc": "...",
  "event_type": "...",
  "component": "...",
  "phase": "...",
  "step": null,
  "metrics": {},
  "latency_ms": null,
  "message": null,
  "error": null,
  "payload": null
}
```

Task-specific fields may be added through explicitly versioned schemas.

### 11.2 Reportable actions

Examples include:
- run start/end;
- phase start/end;
- model load;
- training step;
- validation step;
- checkpoint save;
- inference request;
- inference result;
- evaluation result;
- retry;
- timeout;
- warning;
- exception;
- resource measurement;
- artifact creation;
- external API call;
- cache hit/miss where relevant.

The agent MUST decide which actions are reportable for the task and instrument them.

Do not use unstructured print statements as the only observability mechanism for important behavior.

---

## 12. HTML Dashboard Is Mandatory for Model/Experiment Runs

For model training, inference, evaluation, benchmarking, experimentation, or other telemetry-rich workflows, each run MUST produce:

```text
output/<run_id>/dashboard.html
```

The dashboard MUST be generated from telemetry and the run manifest.

It MUST NOT require a running backend server to inspect historical results.

The repository MUST maintain a static rendering script, for example:

```text
scripts/render_dashboard.py
```

The renderer MUST be deterministic with respect to its input manifest/telemetry.

A completed run is incomplete if its required dashboard cannot be generated.

---

## 13. Dashboard Templates

The reporting system MUST support multiple dashboard templates appropriate to different task classes.

At minimum, architecture SHOULD support templates for:

1. generic run;
2. model training;
3. model inference;
4. evaluation/benchmarking.

Add new templates when a task has materially different observability needs.

Do not force unrelated tasks into an unsuitable dashboard.

### 13.1 Common dashboard capabilities

Depending on the task, dashboards SHOULD support components such as:
- summary tiles;
- time-series graphs;
- histograms;
- distributions;
- tables;
- structured logs;
- warnings;
- errors and stack traces;
- elapsed time;
- step time;
- latency;
- throughput;
- reliability/success rate;
- retry rate;
- resource utilization;
- model outputs/inference examples;
- confidence/scores when available;
- sliding-window statistics;
- training/validation loss;
- learning-rate progression;
- checkpoint events;
- evaluation metrics;
- per-category breakdowns;
- input/output counts;
- failure examples.

Only show metrics meaningful for the run.

Do not fabricate unavailable metrics.

### 13.2 Dashboard data source

The dashboard renderer MUST read the standardized run artifacts.

Business/model execution code MUST NOT contain bespoke HTML generation logic.

Execution emits telemetry.
The reporting layer reads telemetry.
The reporting layer renders HTML.

Keep these responsibilities separated.

---

## 14. Training-Specific Requirements

When implementing model training, configuration MUST explicitly expose all project-chosen training behavior, including applicable values such as:
- model identifier/path;
- dataset identifiers/paths;
- train/validation split behavior;
- epochs or step budget;
- batch sizes;
- gradient accumulation;
- optimizer choice when project-selected;
- learning rate when overridden;
- scheduler choice when overridden;
- warmup when overridden;
- precision mode when project-selected;
- checkpoint policy;
- evaluation cadence;
- save cadence;
- seed(s);
- device/distributed settings controlled by this repository;
- stopping criteria;
- data preprocessing choices;
- augmentation choices;
- sampling choices.

Parameters left entirely at untouched third-party defaults do not need to be copied merely for verbosity.

Every project-chosen override MUST be explicit and recorded.

Training telemetry SHOULD expose enough information to reconstruct progress and diagnose instability.

---

## 15. Inference-Specific Requirements

When implementing model inference, configuration MUST explicitly expose all project-chosen inference behavior, including applicable values such as:
- model identifier/path;
- prompt/template identifier;
- input source;
- batching behavior;
- decoding/generation overrides;
- sampling overrides;
- timeout;
- retry policy;
- concurrency;
- output destination;
- caching behavior;
- evaluation hooks;
- telemetry payload policy.

If generation parameters are intentionally changed from model/library defaults, they MUST be explicit in configuration and the manifest.

Inference telemetry SHOULD support:
- request counts;
- success/failure counts;
- latency;
- throughput;
- retries;
- error classes;
- token counts when available;
- model outputs or safe excerpts when configured;
- rolling/sliding-window latency and reliability.

---

## 16. No Hidden Fallbacks

Do not silently recover from missing required configuration, invalid inputs, architecture violations, or missing reproducibility artifacts by selecting a fallback value.

If the user has not configured a required project-owned behavior:
- fail clearly;
- identify the missing field;
- explain where it must be configured.

Fallback behavior is allowed only when it is itself an explicit configured policy.

---

## 17. Validation and Enforcement Are Mandatory

These commandments MUST be backed by executable checks wherever reasonably possible.

The repository MUST maintain a repository validation entrypoint, for example:

```bash
python scripts/validate_repo.py
```

It MUST fail with a non-zero exit code on policy violations.

A run validation entrypoint SHOULD exist, for example:

```bash
python scripts/validate_run.py output/<run_id>
```

### 17.1 Repository validation should check

Where applicable:
- required top-level structure;
- `.env.example` presence;
- `.env` is ignored by Git;
- `output/` is ignored by Git;
- forbidden imports;
- sibling library imports;
- library-to-main imports;
- one-class-per-file;
- test/source structural alignment;
- required config-schema properties;
- prohibited project-owned defaults;
- telemetry/reporting infrastructure;
- dashboard renderer availability.

### 17.2 Run validation should check

Where applicable:
- run directory exists;
- manifest exists;
- manifest schema is valid;
- telemetry exists and matches schema;
- manifest references existing artifacts;
- configuration hash is valid;
- required output files exist;
- required dashboard exists;
- dashboard can be regenerated;
- completion status is internally consistent.

---

## 18. Git Hooks and CI

Policy enforcement MUST run automatically.

The preferred enforcement layers are:

### Local pre-commit/pre-push

Run fast checks such as:
- formatting/linting;
- architecture validation;
- one-class-per-file;
- forbidden defaults;
- unit tests appropriate for the changed area.

### CI

Run the full policy suite:
- repository validation;
- full tests;
- architecture tests;
- configuration-schema tests;
- representative run-manifest tests;
- telemetry-schema tests;
- dashboard rendering tests.

Do not rely on a Git hook alone because hooks can be skipped.

Do not rely on CI alone because feedback should also be available locally.

If enforcement infrastructure is missing, creating it takes precedence over adding new feature code that depends on these rules.

---

## 19. Architecture Tests Are Part of the Product

The agent MUST treat architecture tests as first-class tests.

At minimum, tests SHOULD detect:
- `src/libs/*` importing `src/main/*`;
- one library object importing another library object;
- modules containing more than one project-owned class;
- project-owned default arguments where prohibited;
- missing required configuration fields;
- runs that proceed without a valid manifest.

Prefer AST/import-graph/schema-based validation over fragile regex checks.

---

## 20. Configuration and Manifest Schema Versioning

Configuration, manifest, and telemetry schemas MUST be versioned.

When a schema changes:
- update validators;
- update producers;
- update consumers;
- update dashboard templates/renderers;
- update tests;
- provide an intentional migration path when historical artifacts need continued support.

Do not silently reinterpret old fields under new semantics.

---

## 21. Run IDs and Artifact Identity

Every run MUST have a unique stable identifier.

A run ID SHOULD be deterministic enough to be readable and unique enough to avoid collision, for example using:
- UTC timestamp;
- short random/UUID suffix;
- optional task name.

Do not overwrite a previous run directory.

Historical run artifacts are immutable unless a clearly identified post-processing action is being performed.

If a dashboard is regenerated later, the underlying manifest/telemetry should remain unchanged and the regeneration action should itself be auditable when relevant.

---

## 22. Errors Must Be Observable

Exceptions and failures that affect a run MUST be:
- logged;
- represented in telemetry;
- summarized in the manifest;
- visible in the dashboard when a dashboard is required.

Do not swallow exceptions.

Do not convert failures into success merely to preserve pipeline continuity.

If a retry policy exists, it must be explicitly configured and retry attempts must be observable.

---

## 23. Logging

Use structured logging for operationally important events.

Logs for a run belong under:

```text
output/<run_id>/logs/
```

Logging configuration is configuration.

Do not let important state exist only in console output.

Avoid duplicate sources of truth: telemetry is for structured events/metrics; logs are for diagnostic detail.

---

## 24. Determinism and Randomness

Any project-controlled randomness MUST be explicitly configurable.

Seeds used by the run MUST be recorded in the manifest.

If deterministic execution is impossible, the manifest MUST still record the relevant seeds and environment details so the run can be reproduced as closely as practical.

Do not introduce a hidden random seed.

---

## 25. External Services and Models

When a run uses an external model, API, service, dataset, or remote dependency, record safe identifying metadata needed for reproducibility, such as:
- provider;
- model/version/deployment identifier;
- endpoint alias when safe;
- dataset version;
- API/package version;
- request configuration overrides.

Do not record secrets.

Where a provider exposes a moving alias, prefer recording the resolved version when it is available.

---

## 26. Agent Workflow for Every Task

For every user request, the coding agent MUST follow this workflow.

### Step 1 — Read policy

Read `AGENTS.md` before planning the change.

### Step 2 — Inspect relevant architecture

Identify:
- affected library objects;
- affected orchestrators;
- dependency boundaries;
- configuration impact;
- test impact;
- telemetry/reporting impact;
- run reproducibility impact.

### Step 3 — Check whether the commandments changed

If the task changes `AGENTS.md` or introduces a rule that conflicts with current code:
- identify existing violations;
- include required refactoring in the task scope;
- do not defer compliance as unrelated cleanup.

### Step 4 — Design before editing

Prefer a design that preserves isolated libraries and composes them in `src/main/`.

Do not solve orchestration problems by coupling libraries.

### Step 5 — Make configuration explicit

Any newly introduced project-owned behavior must be represented in validated configuration before implementation relies on it.

### Step 6 — Add observability

If the task introduces reportable execution, add/update:
- telemetry events;
- manifest fields;
- relevant dashboard template components.

### Step 7 — Implement

Keep one class per file and respect all dependency boundaries.

### Step 8 — Test

Add/update unit, integration, architecture, schema, and reporting tests as applicable.

### Step 9 — Validate

Run the repository policy validator and relevant tests.

For run-producing functionality, execute an appropriate representative run or fixture and validate:
- manifest;
- telemetry;
- dashboard.

### Step 10 — Refactor until compliant

If validation exposes pre-existing or newly introduced violations relevant to the current policy, fix them.

Do not declare success with known violations.

### Step 11 — Report completion

The final response to the user SHOULD summarize:
- what changed;
- configuration added/changed;
- architecture impact;
- tests run;
- policy validation result;
- example run/output location when applicable;
- any intentional limitations.

---

## 27. Definition of Done

A coding task is done only when all applicable statements are true:

- requested behavior is implemented;
- `AGENTS.md` has been followed;
- affected existing code is compliant with current commandments;
- library boundaries remain intact;
- one-class-per-file is preserved;
- configuration contains no hidden project-owned defaults;
- project-selected model/training/inference overrides are explicit;
- tests are present and passing;
- repository validation passes;
- run-producing code cannot proceed without a valid manifest;
- telemetry is emitted for reportable actions;
- model/experiment runs generate a valid static HTML dashboard;
- outputs are isolated under `output/<run_id>/`;
- reproducibility metadata is sufficient to reconstruct the run;
- no secrets were persisted into artifacts;
- no policy check was bypassed.

If any applicable item is false, the task is not complete.

---

## 28. Conflict Resolution

If a user request conflicts with this file, the agent MUST explicitly identify the conflict.

The agent MUST NOT silently violate these commandments.

The user may change the commandments by editing or explicitly requesting an update to `AGENTS.md`.

Once updated, the new policy becomes repository-wide and existing affected code must be brought into compliance.

---

## 29. Policy Priority Inside the Repository

For repository engineering decisions:

1. explicit user instruction for the current task;
2. the latest `AGENTS.md`;
3. project configuration/schema;
4. existing implementation conventions.

A lower-priority convention may not override a higher-priority rule.

When an explicit user instruction changes an engineering commandment, update `AGENTS.md` as part of the task so future tasks use the new rule consistently.

---

## 30. Required Bootstrap Behavior

If a repository adopts this file but does not yet contain the enforcement infrastructure described here, the coding agent MUST bootstrap the missing infrastructure before or alongside feature work.

At minimum bootstrap:
- canonical folders;
- `.gitignore` entries for `.env` and `output/`;
- `.env.example`;
- configuration validation;
- repository architecture validator;
- run manifest schema/validator;
- telemetry schema/writer;
- static dashboard renderer;
- representative dashboard templates;
- tests for the policy invariants;
- local hook configuration;
- CI policy check.

The purpose of this file is not merely to describe desired style. The repository must progressively encode these commandments into executable constraints so violations become difficult to introduce accidentally.

---

## 31. Remote GPU Execution Model (Colab)

The local development machine has no GPU. GPU workloads (training, GPU inference, benchmarking) run on a remote Google Colab runtime, not locally. This section governs how such runs are prepared, executed, and returned.

### 31.1 Execution flow

1. Develop and validate locally on CPU as usual; keep `python scripts/validate_repo.py` and the test suite green.
2. Commit and push to the GitHub remote (`origin`).
3. On Colab, run `notebooks/colab_run.ipynb`, which clones the repository and runs the entrypoint (`python scripts/run.py --config ...`) on a GPU runtime.
4. The remote runs the exact pushed commit. Uncommitted local changes do NOT run remotely — push first.

### 31.2 No local data or filesystem coupling on the remote

- The remote has no access to this machine's files. Code MUST NOT depend on local paths that are not part of the git repository.
- All datasets and models MUST be obtained from the Hugging Face Hub (e.g. `datasets.load_dataset`, `transformers`, `huggingface_hub`), addressed by explicit repository IDs and revisions.
- Credentials such as `HF_TOKEN` are provided via environment/Colab secrets and are never committed (section 8).

### 31.3 External identifiers are configuration

- Dataset IDs, model IDs, and their revisions are project-owned choices and MUST be explicit in `configs/` and recorded in the run manifest (sections 7 and 25). No hidden default dataset or model.

### 31.4 Results return path

- Runs write to `output/<run_id>/` on the remote (manifest, telemetry, dashboard, artifacts) exactly as locally.
- The Colab workflow MUST package the run directory (e.g. zip `output/<run_id>/`) and download it from the remote.
- Downloaded artifacts are placed under this repository's git-ignored `output/` for inspection. Automatic upload to a shared store is permitted when explicitly configured; the manifest remains the source of truth.

### 31.5 Repository access from the remote

- The remote clone requires read access: either the repository is public, or a token (e.g. `GH_TOKEN`) is supplied as a Colab secret. Tokens are never committed.

### 31.6 Invariants preserved

- Reproducibility, manifest/telemetry/dashboard generation, one-class-per-file, and dependency-direction rules apply identically to remote runs. Remote execution changes only where code runs, never the engineering contract.

