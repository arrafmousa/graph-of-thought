# Research Plan: T4 POC for Token-Level SBC Reasoning Graphs

**Status:** design specification for the first scientific proof-of-concept  
**Primary goal:** build a reproducible graph-generation and consolidation pipeline that can later support SBC-level detection and SBC-guided test-time search.  
**Hardware constraint:** one NVIDIA T4-class GPU with 16 GB VRAM.  
**Research-code requirement:** every experiment must be configurable, reproducible, inspectable, and produce enough graph/token statistics to debug scientific failures rather than only software failures.

---

# 1. POC objective

The first proof-of-concept should answer one scientific question before attempting a larger search system:

> **Can token-level hidden representations from a small LLM be used to construct a meaningful reasoning graph in which multiple sampled Chains of Thought branch, rejoin when they reach sufficiently equivalent latent states, and provide stable graph structure for later SBC labeling?**

The POC should **not** begin by fine-tuning the base LLM.

The base LLM should initially remain frozen. We only need it to:

1. generate multiple reasoning traces for a multiple-choice question;
2. expose the hidden representation for every generated token;
3. expose the probability/log-probability of every generated token;
4. allow us to build and inspect the raw and consolidated reasoning graphs;
5. later provide features for a lightweight SBC detector.

This separation is scientifically useful because it lets us determine whether the graph and latent signal exist before introducing model-training confounds.

---

# 2. Recommended POC model and T4 constraints

## 2.1 Recommended first model

Use:

**`meta-llama/Llama-3.2-1B-Instruct`**

as the first POC model.

Reasons:

- approximately 1.23B parameters;
- small enough that FP16 inference and a lightweight detector are practical on a 16 GB T4;
- already part of the intended model family for the larger study;
- gives us a clean path from the POC to later experiments with Llama 3.2 3B, Qwen3 1.5B/1.7B, Gemma 2 2B-it, and SmolLM2 1.7B.

The NVIDIA T4 has 16 GB of GPU memory. For the POC, use **FP16**, not BF16, and keep the base model frozen.

References:

- NVIDIA T4 product brief: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/T4%20Product%20Brief.pdf
- Llama 3.2 1B Instruct: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

## 2.2 Memory rules for the research code

1. **Never keep all hidden states for all chains on the GPU.**
   - Extract the selected hidden representation.
   - Immediately detach it.
   - Move it to CPU.
   - Store it in the trace artifact.

2. **Do not train the base LLM in the first POC.**
   - `requires_grad=False` for all LLM parameters.
   - Only the future SBC probe/detector receives gradients.

3. **Do not request all layers unless an experiment explicitly requires them.**
   - Default: last hidden layer only.
   - The code must support selecting one or multiple layers later.

4. **Use small generation batches.**
   - Initial safe default: 1–2 chains at a time during offline graph generation.
   - Raise only after measuring peak memory.

5. **Cap reasoning length.**
   - Initial proposed value: 256 generated tokens per chain.
   - Exact value is an open question below.

6. **Write hidden states to CPU-backed artifacts instead of keeping a dataset in GPU memory.**

7. Log:
   - peak allocated GPU memory;
   - peak reserved GPU memory;
   - tokens/second;
   - generation wall time;
   - hidden-state extraction wall time.

Quantization should **not** be the default for the first latent-geometry experiment. A 1B model fits in FP16, and changing numerical representation could itself alter hidden-state distances. Quantization can later become an ablation.

---

# 3. What the graph represents

There are two related graph objects.

## 3.1 Raw Chain-of-Thought graph

For a question \(q\), sample \(N\) independent reasoning traces:

\[
C_i = (x_{i,1}, x_{i,2}, \ldots, x_{i,T_i})
\]

where each \(x_{i,t}\) is one generated token.

Initially every chain is independent after the common prompt.

A raw token node is:

\[
v_{i,t}
\]

and represents:

> **the autoregressive reasoning state immediately after chain \(i\) generated token \(t\).**

The node is not merely the visible token text.

Each node must contain at least:

- `question_id`
- `chain_id`
- `token_index`
- `token_id`
- decoded token text
- cumulative generated text up to this token, or a reference to it
- hidden representation \(h_{i,t}\)
- hidden layer identifier
- generated-token probability
- generated-token log-probability
- entropy of the next-token distribution if collected
- generation depth
- whether the chain has terminated
- final answer correctness once known

A raw edge

\[
v_{i,t-1} \rightarrow v_{i,t}
\]

means:

> the model generated token \(t\) from the autoregressive context represented by the preceding state.

Before consolidation, the graph is effectively a set of \(N\) chains sharing one root prompt.

---

# 4. Multiple-choice generation protocol

Every example must be converted to a deterministic multiple-choice prompt format.

The model is explicitly instructed to:

1. reason through the problem;
2. provide its Chain of Thought;
3. end the entire generation with **exactly one option letter**;
4. generate nothing after that final letter.

Example instruction:

> Reason through the problem step by step. After the reasoning, output the final answer as exactly one option letter from {A, B, C, D}. The last character of your response must be that answer letter. Do not write anything after it.

## 4.1 Answer parsing

The evaluator takes the **last generated alphabetic character** after trimming whitespace.

It is valid only if it belongs to the question's allowed option set.

For a four-choice problem:

\[
\hat y \in \{A,B,C,D\}
\]

If the last letter is outside the option set or no valid final letter exists:

- mark the generation as `invalid_answer_format`;
- count it as incorrect for accuracy;
- keep the trace for graph/debugging analysis unless we explicitly decide otherwise.

Log separately:

- correct traces;
- incorrect traces;
- invalid-format traces.

This prevents formatting failures from silently being mixed with reasoning failures.

---

# 5. Generating the raw traces

For every question:

1. tokenize the common prompt once;
2. sample \(N\) independent reasoning chains using different random draws;
3. for every generated token:
   - record token ID/text;
   - record selected hidden state(s);
   - record log-probability of the sampled token;
   - record optional next-token entropy/top-k probabilities;
4. stop when:
   - EOS is reached; or
   - the configured maximum number of generated tokens is reached;
5. parse the final answer letter;
6. attach terminal correctness to the chain.

All sampling settings must be written into the artifact:

- temperature;
- top-p;
- top-k, if used;
- repetition/presence penalty, if used;
- maximum new tokens;
- random seed;
- model revision/hash.

No generation parameter should exist only inside source code.

---

# 6. Graph artifact before consolidation

Every question should produce a versioned raw trace artifact before any node merging happens.

Recommended structure:

```text
outputs/
  run_<run_id>/
    config.yaml
    environment.json
    question_<id>/
      question.json
      raw_traces.jsonl
      hidden_states/
        chain_000.safetensors
        chain_001.safetensors
        ...
      raw_graph.json
      raw_stats.json
```

This is important because consolidation heuristics will change.

We should **never need to regenerate the expensive LLM traces simply because we want to try a new clustering threshold.**

The raw generation stage and graph-consolidation stage must therefore be separate commands.

Conceptual CLI:

```bash
python -m sbc.generate --config configs/poc.yaml
python -m sbc.build_graph --run outputs/run_001 --merge-policy cosine_last --threshold 0.97
```

---

# 7. Token-level latent consolidation

The consolidated graph is produced by identifying nodes from different chains that appear to represent sufficiently equivalent reasoning states.

If:

\[
v_{i,t} \sim v_{j,s}
\]

then both raw trajectories should point to one shared conceptual node in the consolidated graph.

The graph therefore changes from independent chains into a DAG with:

- **branches**: one state develops into different reasoning futures;
- **joins**: multiple chains reach what the heuristic believes is the same reasoning state.

A join node should have:

\[
d_{in} > 1
\]

A branching node should have:

\[
d_{out} > 1
\]

---

# 8. Important semantic rule: merging does NOT mean averaging two LLM states

The graph merge is a **search abstraction**, not a new Transformer state.

We must never do:

\[
h_{\text{merged}} = (h_i+h_j)/2
\]

and then attempt to continue autoregressive decoding from that artificial state.

The Transformer continuation depends on the full preceding context and KV cache.

Therefore, after two active reasoning chains are declared equivalent:

1. the graph records that both paths reached one conceptual merged node;
2. **only one physical chain/KV cache survives**;
3. the other chain is terminated as a redundant continuation;
4. all future generation from the merged node uses the surviving representative chain.

This creates a **virtual merge**.

---

# 9. Which chain survives a merge?

This must be deterministic and logged.

The initial POC rule should follow the proposed criterion:

> **Continue the chain whose model confidence over the most recent tokens is higher on average.**

For a chain \(C_i\), define recent confidence over a window of \(w\) tokens.

Recommended primary measure:

\[
\operatorname{conf}(C_i)
=
\frac{1}{w}
\sum_{k=t-w+1}^{t}
\log p(x_{i,k}\mid x_{i,<k},q)
\]

Use mean **log-probability**, rather than multiplying probabilities, for numerical stability.

When two states merge:

\[
C^*
=
\arg\max_{C\in\{C_i,C_j\}}
\operatorname{conf}(C)
\]

Then:

- \(C^*\) retains its KV cache and may be expanded later;
- the other chain is marked `terminated_by_merge`;
- both incoming paths remain visible in the graph.

The merge event must record:

```json
{
  "winner_chain": "...",
  "loser_chain": "...",
  "winner_recent_mean_logprob": "...",
  "loser_recent_mean_logprob": "...",
  "confidence_window": "...",
  "merge_heuristic": "...",
  "merge_threshold": "...",
  "similarity": "..."
}
```

The code should support alternative representative-selection policies later, but **recent mean token confidence** is the first POC policy.

---

# 10. Candidate merge restrictions

Raw hidden-state similarity alone is not sufficient to define a safe merge.

The POC should support constraints independently from the similarity heuristic.

## 10.1 Different-chain constraint

Only merge nodes from different chains.

Do not collapse two states in the same chain during the first POC.

## 10.2 No ancestor/descendant merge

Never merge a node with one of its own ancestors or descendants.

Otherwise the graph can create artificial cycles.

The consolidated graph must remain a DAG.

Every graph build should assert:

```text
is_directed_acyclic_graph == True
```

and fail loudly otherwise.

## 10.3 Depth/progress constraint

The POC should make the candidate window configurable:

```yaml
merge:
  depth_policy: same_depth | absolute_window | unrestricted
  max_depth_difference: 4
```

The scientific sweep should compare at least:

1. same token depth only;
2. token depth within ±δ;
3. unrestricted depth, while still forbidding ancestor/descendant merges.

**This choice is scientifically unresolved and is listed again in the questions section.**

---

# 11. Pluggable node-consolidation heuristics

The POC must treat merging as an experimental component.

There should be a common interface:

```python
similarity = merge_metric(state_a, state_b, context)
should_merge = similarity >= threshold
```

The code must be able to rerun all graph construction from stored raw traces without regenerating LLM outputs.

## H0 — Token identity baseline

Merge only if:

- decoded/generated token is identical; and
- optional depth restrictions are satisfied.

This is intentionally weak and serves as a sanity baseline.

## H1 — Raw last-layer hidden-state cosine

\[
s(i,j)=\cos(h_i,h_j)
\]

This is the simplest latent-state heuristic.

## H2 — L2-normalized / centered hidden-state cosine

Center or standardize hidden states before measuring cosine similarity.

Purpose:

- test whether anisotropy in the raw representation causes false merges.

Normalization statistics must be fit only on training data.

## H3 — Last-\(k\)-token hidden-state pooling

A single token hidden state may be too local.

Represent the state as:

\[
z_t
=
\frac{1}{k}
\sum_{r=t-k+1}^{t}h_r
\]

and compare \(z_i,z_j\) by cosine similarity.

Support:

```text
k ∈ {1, 2, 4, 8}
```

`k=1` reduces to the token-state representation.

## H4 — Confidence-gated latent similarity

Require:

1. latent similarity above threshold; and
2. both states to exceed a minimum recent-confidence level, or their confidence difference to remain below a configured tolerance.

This tests whether uncertain latent states cause unstable merges.

## H5 — Token-distribution-aware similarity

Combine hidden-state similarity with similarity between the models' next-token distributions.

For example:

\[
S =
\alpha\,\cos(h_i,h_j)
-
(1-\alpha)\operatorname{JSD}(P_i,P_j)
\]

where \(P_i,P_j\) are top-k next-token distributions.

This is closer to a behavioral definition:

> similar internal state **and** similar immediate future predictions.

## H6 — Multi-layer hidden similarity

Compare selected layers instead of only the last layer.

\[
S(i,j)
=
\frac{1}{|L|}
\sum_{\ell\in L}
\cos(h_i^{(\ell)},h_j^{(\ell)})
\]

The first POC does not need a large layer sweep, but the software architecture must support it.

---

# 12. Threshold sweeps

No single hidden-similarity threshold should be hard-coded into the scientific result.

For cosine-based metrics, support an initial sweep such as:

```text
0.90
0.93
0.95
0.97
0.98
0.99
0.995
```

This is only an initial grid.

## 12.1 Absolute thresholds

```yaml
thresholds: [0.90, 0.93, 0.95, 0.97, 0.99]
```

## 12.2 Quantile-calibrated thresholds

Because hidden-state cosine values may be concentrated, allow thresholds based on empirical training distributions.

```yaml
threshold_quantiles: [0.95, 0.975, 0.99, 0.995]
```

The report should show performance as a function of threshold rather than selecting one threshold without justification.

---

# 13. Offline consolidation algorithm for the first graph POC

For the initial graph-construction experiment, generation happens first and consolidation happens offline.

This isolates the merge heuristic from changes in generation.

Pseudo-procedure:

```text
INPUT:
  N completed raw CoTs for one question
  token hidden states
  merge heuristic H
  threshold τ

1. Create one raw token path per CoT.
2. Process token nodes in increasing generation depth.
3. For each node v:
      a. obtain candidate nodes from other chains;
      b. apply depth/acyclicity candidate restrictions;
      c. compute H(v, candidate);
      d. find the highest-similarity admissible candidate.
4. If similarity >= τ:
      consolidate v with the candidate cluster.
5. Otherwise:
      create a new graph state.
6. Add incoming/outgoing edges.
7. Verify DAG invariants.
8. Calculate graph diagnostics.
9. Generate HTML visualization.
```

This gives a scientific view of how different latent heuristics reconstruct reasoning topology from the **same underlying samples**.

---

# 14. Online merging rule for later test-time search

After the offline POC is validated, the same interface is used during generation.

Assume a maximum of \(K\) active chains.

After a newly developed chain produces a token/chunk:

1. compute the new hidden representation;
2. score SBC risk using the future detector;
3. if SBC risk exceeds the pruning threshold:
   - stop the branch;
   - remove its KV cache from the active frontier;
4. otherwise compare it against admissible active/visited frontier states;
5. if it matches another state:
   - create/record a graph join;
   - compare recent mean log-probabilities;
   - retain only the higher-confidence physical chain;
   - release the losing KV cache;
6. if it does not merge:
   - retain it as a distinct frontier state;
7. the search controller later chooses which surviving state to develop.

The crucial invariant is:

> **A merge produces one future continuation, not two equivalent continuations.**

---

# 15. Graph statistics required for debugging

Every graph build must produce both machine-readable statistics and a human-readable summary.

## 15.1 Raw generation statistics

Per question:

- number of sampled chains;
- total generated tokens;
- min/mean/median/max chain length;
- correct-chain count;
- incorrect-chain count;
- invalid-format count;
- average sampled-token log-probability per chain;
- terminal answer distribution;
- generation time;
- peak GPU memory.

Across dataset:

- same statistics aggregated;
- answer accuracy;
- chain-level success rate;
- fraction of questions with at least one correct sampled chain.

## 15.2 Consolidated graph statistics

For every heuristic × threshold:

- raw token nodes;
- consolidated nodes;
- raw edges;
- consolidated edges;
- node reduction:

\[
1-\frac{|V_{\text{consolidated}}|}{|V_{\text{raw}}|}
\]

- number of merge events;
- merge acceptance rate;
- number of clusters;
- cluster-size distribution;
- largest cluster size;
- number of branch nodes:

\[
|\{v:d_{out}(v)>1\}|
\]

- number of join nodes:

\[
|\{v:d_{in}(v)>1\}|
\]

- maximum \(d_{in}\);
- maximum \(d_{out}\);
- mean/median \(d_{in}\);
- mean/median \(d_{out}\);
- graph depth;
- number of terminal nodes;
- weakly connected component count;
- DAG validity;
- topological-sort success;
- number of chains terminated by merge during online mode;
- tokens avoided by merge during online mode.

## 15.3 Merge diagnostics

For every proposed merge:

- similarity value;
- threshold;
- heuristic;
- node depths;
- token texts;
- last \(w\) generated tokens from both chains;
- recent confidence of both chains;
- winning representative;
- whether final outcomes of the original offline chains agree.

If supposedly equivalent nodes belong to traces with systematically incompatible futures, the heuristic may be over-merging.

---

# 16. Required histograms and plots

Every run should generate:

1. histogram of node \(d_{in}\);
2. histogram of node \(d_{out}\);
3. histogram of cluster sizes;
4. distribution of accepted merge similarities;
5. distribution of rejected candidate similarities;
6. chain-length histogram;
7. graph compression ratio versus merge threshold;
8. join-node count versus threshold;
9. branch-node count versus threshold;
10. correct/incorrect future agreement of merged states versus threshold.

Later, once SBC labels exist:

11. SBC-node count;
12. SBC-adjacent-node count;
13. SBC score distribution by label;
14. distance in tokens between SBC-adjacent and terminal failure.

---

# 17. HTML graph visualization

The visualization is part of the scientific debugging artifact, not cosmetic presentation.

Each question should produce a standalone HTML report.

## 17.1 View A — CoT lanes

Show every sampled CoT **one above the other**.

Conceptually:

```text
CoT 0:  ●──●──●──●────────────●──✓
                 ╲
CoT 1:  ●──●──●──●──●─────────╯
                        ...
CoT 2:  ●──●──●──●──●──●──✗
```

Requirements:

- one horizontal lane per original CoT;
- token order left-to-right;
- common prompt/root visually shared;
- branch points visible;
- joins/merges visible across lanes;
- after an **online** merge:
  - losing chain becomes gray/dashed;
  - winner visibly continues;
  - label the reason: `merged → representative chain X`;
- eventual SBC nodes red;
- SBC-adjacent nodes orange;
- correct terminal nodes marked clearly;
- incorrect terminal nodes marked clearly.

## 17.2 Node hover/click information

Selecting a node should show:

- visible token;
- token ID;
- chain ID;
- token depth;
- hidden layer;
- cluster ID;
- similarity that caused merge;
- recent mean log-probability;
- generated-token probability;
- \(d_{in}\);
- \(d_{out}\);
- SBC label/score when available;
- representative-chain status.

## 17.3 View B — Consolidated graph

A second panel should display the resulting DAG independently of the original lane layout.

This allows us to see:

- branching structure;
- convergence;
- hubs;
- terminal states;
- SBC regions.

## 17.4 View C — Graph statistics

The same HTML report should include:

- \(d_{in}\) histogram;
- \(d_{out}\) histogram;
- cluster-size histogram;
- summary-statistics table.

## 17.5 Interactive heuristic/threshold comparison

The preferred debugging interface should eventually allow selection of:

- merge heuristic;
- threshold;
- last-\(k\) pooling value;
- depth window.

Changing these settings should display the corresponding **precomputed graph**, not rerun the LLM.

This makes threshold sensitivity immediately visible.

---

# 18. SBC labeling stage after graph construction

Once graph generation and consolidation are trustworthy, reuse the existing backward-traversal SBC procedure.

The scientific intention is:

1. mark correct and incorrect terminal outcomes;
2. traverse backward from failed regions;
3. use observed alternative descendants to determine whether earlier states were still recoverable;
4. identify the transition into the irrecoverable region;
5. label:
   - safe/recoverable nodes;
   - SBC-adjacent nodes;
   - SBC nodes.

A natural graph definition would be:

\[
R(v)=1
\]

if the graph contains at least one acceptable successful continuation from \(v\), and:

\[
R(v)=0
\]

if all admissible observed continuations fail.

However, **the exact SBC boundary definition from the existing SBC work must be preserved rather than silently replaced by this proposed definition.** This is one of the blocking questions at the end.

The label-generation code should be a separate module from graph construction:

```text
raw generation
      ↓
graph consolidation
      ↓
SBC backward labeling
      ↓
detector dataset
```

This lets us rerun labels under different graph heuristics and study label stability.

---

# 19. Training the SBC-level detector

After graph labeling is validated, build a detector dataset.

One row corresponds to one graph/token state:

```text
question_id
chain_id
token_index
hidden_state
graph_cluster_id
label
recoverability metadata
```

## 19.1 First detector

Keep the LLM frozen.

Start with:

1. multinomial logistic/linear probe;
2. two-layer MLP.

Possible target:

```text
0 = safe
1 = SBC-adjacent
2 = SBC
```

or an ordinal/continuous SBC-risk target.

The exact target is still an open design choice.

## 19.2 Split correctly

**Split by question, never by node.**

All chains and graph nodes from one question must belong to the same train/validation/test partition.

Otherwise states from the same reasoning graph leak across splits.

## 19.3 Detector baselines

At minimum compare hidden-state SBC prediction against:

- sampled-token confidence;
- recent mean log-probability;
- next-token entropy;
- raw chain depth;
- token identity/simple text baseline if practical.

The scientific claim should require the hidden-state detector to outperform simple uncertainty/confidence heuristics.

---

# 20. Detector end goal for the POC

The detector POC is successful if:

1. SBC/SBC-adjacent labels can be generated consistently from held-out reasoning graphs;
2. a lightweight detector predicts those labels on **held-out questions**;
3. performance is better than confidence/entropy-only baselines;
4. the result is not specific to exactly one arbitrary graph-merge threshold;
5. the detector can run online cheaply enough to be used during generation.

This is the point at which we proceed to SBC-guided test-time graph search.

---

# 21. Test-time POC after detector validation

The next stage uses at most \(K\) active reasoning chains.

Initial intended value based on the current design:

\[
K=10
\]

Process:

```text
prompt
  ↓
generate initial diverse branches
  ↓
score new token/chunk with SBC detector
  ↓
SBC? ── yes → prune
  │
  no
  ↓
equivalent to another state?
  │
  yes → merge → retain higher-confidence representative only
  │
  no  → new graph node
  ↓
search controller chooses the next state to develop
  ↓
repeat
  ↓
verified terminal answer
```

For the first online POC, keep the search controller simple.

Possible sequence:

1. uniform/random frontier selection;
2. confidence-based frontier selection;
3. UCB/MAB selection;
4. full MCTS-style policy if justified.

This permits ablation of the SBC/merge contribution from the controller.

---

# 22. Scientific experiment sequence

## Step 1 — Infrastructure sanity check

Use 10–20 multiple-choice questions.

Goal:

- verify prompt formatting;
- verify final-letter parser;
- verify token-level hidden states;
- verify per-token log-probabilities;
- verify T4 memory usage;
- produce raw trace artifacts.

**Exit criterion:** raw traces are deterministic under fixed seeds and inspectable.

## Step 2 — Raw graph generation

Use approximately 50 questions and multiple chains per question.

Goal:

- construct one token path per CoT;
- verify raw graph schema;
- generate CoT-lane HTML;
- generate raw graph statistics.

**Exit criterion:** graph exactly reconstructs every sampled text trace.

## Step 3 — Offline latent consolidation sweep

On the same saved traces, sweep:

- heuristic;
- similarity threshold;
- pooling \(k\);
- depth restriction.

Goal:

- characterize how graph topology changes;
- identify over-merging regimes;
- identify thresholds that yield nontrivial but not destructive consolidation.

**Exit criterion:** a small set of candidate merge policies is selected using predeclared criteria, not visual preference alone.

## Step 4 — Merge-quality analysis

For every accepted merge, compare the original downstream futures.

Questions:

- Do merged chains tend to reach the same final answer?
- Do they have similar correctness probability?
- Does aggressive merging collapse successful and failed futures?
- Does recent-confidence winner selection preserve the better representative?

**Exit criterion:** evidence that the chosen latent merge policy captures something stronger than arbitrary geometric proximity.

## Step 5 — Backward SBC labeling

Run the existing SBC backward procedure over consolidated graphs.

Goal:

- generate safe/SBC-adjacent/SBC node labels;
- study label stability across reasonable merge policies.

**Exit criterion:** SBC labels are not completely unstable under small threshold changes.

## Step 6 — Train the SBC detector

Freeze the LLM.

Train:

- linear probe;
- 2-layer MLP.

Evaluate on held-out questions.

**Exit criterion:** hidden-state detector beats uncertainty/confidence baselines and is calibrated well enough for online pruning.

## Step 7 — Online merge-only search

Before SBC pruning, implement online merging with the 10-chain budget.

Goal:

- validate virtual merges;
- verify that only one chain continues after a join;
- measure tokens saved by removing redundant continuations;
- verify winner selection using recent confidence.

**Exit criterion:** merging reduces tokens without unacceptable accuracy loss.

## Step 8 — Online SBC pruning

Add detector-based pruning.

Goal:

- stop branches predicted to be irrecoverable;
- measure early-pruning false positives;
- measure tokens avoided after predicted SBC.

**Exit criterion:** saved tokens are not obtained by simply destroying successful branches.

## Step 9 — Adaptive frontier allocation

Only after the previous stages work, add MAB/UCB/MCTS-style state selection.

Goal:

- reallocate freed compute toward safe, distinct states.

Compare under **equal total generation-token budget**.

**Paper-relevant end criterion:**

\[
\text{full method}
>
\text{independent CoT / self-consistency}
\]

in accuracy at matched tokens,

or comparable accuracy with meaningfully fewer generated tokens.

---

# 23. POC experiment matrix

The experiment code should make the following dimensions declarative:

```yaml
model:
  name: meta-llama/Llama-3.2-1B-Instruct
  dtype: float16
  hidden_layers: [-1]

generation:
  num_chains: 10
  max_new_tokens: 256
  temperature: TBD
  top_p: TBD
  seeds: TBD

merge:
  heuristic:
    - token_identity
    - hidden_cosine
    - centered_hidden_cosine
    - pooled_hidden_cosine
    - confidence_gated_cosine
    - hidden_plus_next_distribution
    - multi_layer_cosine
  thresholds:
    - 0.90
    - 0.93
    - 0.95
    - 0.97
    - 0.98
    - 0.99
  pooling_k: [1, 2, 4, 8]
  depth_policy: TBD
  confidence_window: TBD

sbc:
  labeling_definition: TBD
  adjacent_window: TBD

detector:
  architecture:
    - linear
    - mlp
  target: TBD
```

Every plot/table should include the complete configuration or a run ID resolving to it.

---

# 24. Proposed research-code organization

```text
sbc-reasoning/
├── README.md
├── pyproject.toml
├── configs/
│   ├── poc_llama1b.yaml
│   └── sweeps/
├── src/sbc/
│   ├── models/
│   │   ├── loader.py
│   │   └── hidden_state_hook.py
│   ├── generation/
│   │   ├── prompt.py
│   │   ├── sampler.py
│   │   └── answer_parser.py
│   ├── traces/
│   │   ├── schema.py
│   │   └── storage.py
│   ├── graph/
│   │   ├── schema.py
│   │   ├── raw_graph.py
│   │   ├── consolidate.py
│   │   ├── merge_registry.py
│   │   ├── candidate_filter.py
│   │   ├── representative.py
│   │   ├── statistics.py
│   │   └── sbc_backward.py
│   ├── detector/
│   │   ├── dataset.py
│   │   ├── linear_probe.py
│   │   ├── mlp.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── search/
│   │   ├── frontier.py
│   │   ├── online_merge.py
│   │   ├── sbc_prune.py
│   │   └── controller.py
│   └── viz/
│       ├── html_report.py
│       ├── cot_lanes.py
│       └── histograms.py
├── tests/
│   ├── test_answer_parser.py
│   ├── test_graph_reconstruction.py
│   ├── test_dag_invariant.py
│   ├── test_merge_policy.py
│   ├── test_representative_selection.py
│   └── test_sbc_backward.py
└── outputs/
```

---

# 25. Reproducibility requirements

Every run must record:

- git commit;
- Python version;
- PyTorch version;
- Transformers version;
- CUDA version;
- GPU type;
- model ID and revision;
- dataset name/version;
- complete generation parameters;
- complete merge parameters;
- random seeds;
- graph schema version;
- SBC label schema version.

No paper result should depend on a notebook-only manual operation.

Notebooks can be used for analysis, but all generation, graph building, labeling, training, and evaluation must have CLI/script entry points.

---

# 26. Unit/invariant tests that are scientifically important

## Trace reconstruction

Concatenating stored generated tokens must exactly reconstruct the original model completion.

## Graph preservation

Before online merge termination is introduced, every raw path must be recoverable from the graph representation.

## DAG invariant

Consolidation must never introduce a cycle.

## Merge determinism

Given identical saved traces, heuristic, and threshold, graph consolidation must be deterministic.

## Winner determinism

Given two merge candidates and a confidence window, the representative chain must be selected deterministically.

## Answer parser

The final answer must be exactly the last allowed letter according to the documented parser.

## Question split isolation

No graph node from the same question can appear in multiple detector splits.

---

# 27. End goal of the T4 POC

The POC is **not** merely “produce a nice graph.”

Its end goal is to establish the minimum scientific chain of evidence:

### A. Representation

Token-level hidden states yield nontrivial recurring reasoning-state structure across independently sampled CoTs.

### B. Consolidation

A measurable merge criterion can remove redundant reasoning states without arbitrarily collapsing incompatible futures.

### C. SBC supervision

The consolidated graph supports the existing backward procedure for identifying SBC and SBC-adjacent states.

### D. Prediction

A cheap latent detector can learn those labels on held-out questions.

### E. Inference utility

At test time:

- equivalent states merge;
- only the higher-confidence representative continues;
- SBC states are pruned;
- freed compute can later be reallocated to another safe reasoning state.

If A–D fail, we should **not** spend time building a complicated MCTS system.

If A–D succeed, E becomes the paper's main inference-time validation.

---

# 28. Proposed later model progression

Do not start with all models.

Suggested order:

1. **Llama 3.2 1B Instruct** — POC and debugging.
2. **Qwen3 1.5B/1.7B** — architecture/model-family replication.
3. **SmolLM2 1.7B Instruct** — very-small-model replication.
4. **Gemma 2 2B-it** — different architecture family.
5. **Llama 3.2 3B / Qwen3 3B** — only after memory/runtime profiling.

The exact final benchmark suite can be decided after the POC.

---

# 29. Reference directly relevant to this plan

**Learning Rewrite-Invariant Reasoning with Targeted Alternation Training**  
Mousa Arraf, Ido Guy, Kira Radinsky. ICML 2026 poster.  
OpenReview: https://openreview.net/pdf?id=Z2qLXu0Yf9

The work samples multiple solution traces under semantically equivalent rewrites and aggregates recurring intermediate reasoning steps into a graph in order to identify where incorrect traces diverge from correct ones. The present POC reuses the graph/backward-localization direction as the training-time foundation and investigates token-level latent consolidation and learned SBC prediction.

---

# 30. BLOCKING / OPEN QUESTIONS FOR YOU

The following details are not fully specified in our discussion. I do **not** want the implementation to silently choose them.

Please answer these before the research-code specification is considered final.

## Q1. Dataset for the first POC

Which multiple-choice dataset should the first experiment use?

Possible choices:

- ARC-Challenge;
- MMLU;
- MMLU-Pro;
- another dataset from the existing SBC work;
- a mixture.

**Proposed default if you do not have a preference:** ARC-Challenge for software/graph validation, followed by the paper's target benchmark.

## Q2. Exact SBC definition from the existing work

When the backward traversal identifies the boundary, does the **SBC node** mean:

**A.** the **first irrecoverable node**, whose parent was still recoverable;

or

**B.** the **last recoverable node immediately before** the chain became irrecoverable;

or another definition already used in the SBC implementation?

I need the exact existing definition preserved.

## Q3. What exactly is “SBC-adjacent”?

Should SBC-adjacent mean:

- exactly the parent immediately preceding the SBC;
- ±1 graph hop around the SBC;
- the previous \(k\) token states;
- a continuous recoverability range;
- something already defined in your current work?

## Q4. How many chains per training question?

We have discussed a maximum of **10 active chains at test time**.

Should offline training-graph generation also use exactly 10 CoTs per question, or more (for example 20–50) to obtain better recoverability evidence?

## Q5. Reasoning-length limit

What maximum number of newly generated tokens should the POC allow?

Proposed starting point:

```text
256 tokens / chain
```

## Q6. Sampling configuration

What sampling policy should define the different CoTs?

We need fixed values for at least:

- temperature;
- top-p;
- optional top-k.

The diversity of the graph is directly controlled by this choice.

## Q7. Merge candidate depth constraint

When two token states from different chains are similar, should we allow a merge when they are at different token depths?

Choose one initial rule:

**A.** same depth only;  
**B.** depth within ±4 tokens;  
**C.** configurable window and sweep it;  
**D.** unrestricted depth except ancestor/descendant merges.

My recommendation is **C**.

## Q8. Confidence window for selecting the surviving chain

After two chains merge, how many recent tokens should determine which physical chain survives?

Potential values:

```text
w ∈ {4, 8, 16}
```

My recommendation is to support the sweep but use **8** as the initial default.

## Q9. Definition of model confidence

Should the primary definition be:

\[
\text{mean log-probability of the sampled tokens}
\]

as proposed here?

Alternative possibilities:

- arithmetic mean of sampled-token probabilities;
- geometric mean probability;
- negative entropy;
- combination of confidence and SBC risk.

My recommendation is mean log-probability.

## Q10. What happens when two merged chains have almost identical confidence?

Do we:

- deterministically choose lower chain ID;
- choose shorter chain;
- choose lower SBC risk;
- keep both when confidence difference is below a margin?

For a clean graph abstraction I recommend **always retaining one**, with SBC risk as first tie-breaker and chain ID as final deterministic tie-breaker.

## Q11. Are merges allowed only among the current frontier, or against all previously visited graph states?

**Frontier-only merging** is simpler and avoids “jumping back” to old states.

**Global visited-state merging** gives a stronger transposition-table interpretation but requires a policy for what happens when a new chain reaches an old state whose representative is no longer active.

Which one is intended?

My recommendation for the first online POC: **frontier-only**.

## Q12. Raw token state or short chunk state as the primary unit?

You explicitly want token-level hidden representations.

Should the *primary graph node* always be every generated token, while last-\(k\) pooling is only a similarity heuristic?

That is how this document currently specifies it.

Please confirm.

## Q13. Which hidden layer is the primary representation?

Proposed POC:

- last layer as the primary graph state;
- selected intermediate layers only as an ablation.

Do you want a specific layer or layer combination from the beginning?

## Q14. Is the graph consolidation used to create SBC labels allowed to use the same hidden-state similarity later used by the detector?

This is scientifically important.

If SBC labels depend strongly on hidden-state clustering and the detector is then trained on those same hidden states, reviewers may ask whether the target is partly circular.

Options:

**A.** use the existing SBC graph-construction method as the labeling oracle and use hidden states only for the new detector/test-time merge;

**B.** use latent merging for label graphs as well, then explicitly measure label stability across heuristics and thresholds;

**C.** maintain both and compare them.

My recommendation is **C** if computationally feasible.

## Q15. What is the correctness oracle?

For the multiple-choice POC I assume:

```text
last generated allowed option letter == gold answer
```

Is that sufficient, or does the existing project have a separate evaluator?

## Q16. Should an invalid final-answer format be treated as:

- wrong but retained in the graph;
- discarded entirely;
- a separate terminal failure class?

This plan currently recommends: **wrong + retained + separately flagged**.

Please confirm.

## Q17. Online stopping rule

At test time, if any active chain reaches a terminal answer letter, do we:

- stop immediately;
- verify it somehow first;
- wait for multiple agreeing terminal branches;
- use a fixed terminal score?

Earlier we discussed stopping at a **verified** terminal state, but the multiple-choice verification policy has not yet been specified.

## Q18. Search controller scope for this first implementation

Do you want the first POC codebase to already include the MAB/UCB controller, or should the milestones be:

1. graph generation;
2. merging;
3. SBC labeling/detector;
4. pruning;
5. only then MAB?

My recommendation is the staged version so every component has an independent ablation.

## Q19. Dataset scale for POC

What scale is acceptable for the first T4 experiment?

A reasonable progression could be:

```text
20 questions   → engineering validation
100 questions  → graph/threshold analysis
500+ questions → detector POC
```

## Q20. Should graph equivalence be question-local only?

This plan assumes nodes are merged **only within the same question**.

Do you ever intend to cluster states across different questions?

For the POC I strongly recommend **question-local only**.

---

# 31. Immediate implementation milestone after these questions are answered

The first code milestone should contain only:

1. Llama 3.2 1B loader for T4/FP16;
2. reproducible multiple-choice prompt + answer parser;
3. generation of \(N\) CoTs with per-token:
   - hidden state,
   - log-probability,
   - metadata;
4. raw graph creation;
5. merge-policy registry;
6. threshold sweep;
7. confidence-based representative selection;
8. DAG validation;
9. graph statistics;
10. standalone HTML containing:
    - stacked CoT lanes,
    - branch/join visualization,
    - consolidated graph,
    - \(d_{in}\) histogram,
    - \(d_{out}\) histogram,
    - merge/debug table.

**Do not train the SBC detector until this artifact is trustworthy.**

That produces a clean first scientific checkpoint:

> **Given identical sampled CoTs, how do different latent-state equivalence assumptions change the topology of the reasoning graph?**

Only after answering that question should the backward SBC labels become training targets for the detector.
