# RLCR: Reinforcement Learning for Code Review Filtering

**Teaching an LLM which code review comments actually matter — per team, from scratch, with ≤50 examples.**

## The Problem

AI code review tools (Greptile, CodeRabbit, etc.) generate hundreds of comments per PR. Most are noise. The hard part isn't generating comments — it's deciding which ones to **surface** vs **filter**.

Greptile's team [documented](https://greptile.com): they tried 4 approaches, and embedding-based cosine similarity was their best bet. But it has fundamental limitations:

| Approach | Problem |
|----------|---------|
| Rule-based filters | Can't capture team-specific preferences |
| Cosine similarity | Cold-start failure, no nuance beyond surface similarity |
| Classifier fine-tuning | Needs thousands of labeled examples per team |
| LLM prompting | Expensive, inconsistent, doesn't learn from feedback |

**RLCR shows that GRPO (Group Relative Policy Optimization) solves what embeddings can't**: learning per-team preferences from as few as 20 upvote/downvote signals, with better cold-start performance and cross-team generalization.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    RLCR Pipeline                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐ │
│  │CodeReview│───▶│  Team    │───▶│  Per-Team Data   │ │
│  │ Dataset  │    │Simulator │    │  (20-50 train)   │ │
│  │ (116K)   │    │(5 teams) │    │  (200+ test)     │ │
│  └─────────┘    └──────────┘    └────────┬─────────┘ │
│                                           │           │
│                    ┌──────────────────────┤           │
│                    │                      │           │
│                    ▼                      ▼           │
│  ┌──────────────────────┐  ┌──────────────────────┐  │
│  │  Embedding Baseline  │  │     GRPO Training    │  │
│  │  (Greptile approach) │  │                      │  │
│  │                      │  │  ┌────────────────┐  │  │
│  │  MiniLM-L6-v2        │  │  │ SGLang Server  │  │  │
│  │  + FAISS per-team    │  │  │ (fast rollout  │  │  │
│  │  + threshold tuning  │  │  │  generation)   │  │  │
│  │                      │  │  └───────┬────────┘  │  │
│  └──────────┬───────────┘  │          │           │  │
│             │              │  ┌───────▼────────┐  │  │
│             │              │  │ Local Model    │  │  │
│             │              │  │ + LoRA adapter  │  │  │
│             │              │  │ (gradient pass) │  │  │
│             │              │  └───────┬────────┘  │  │
│             │              │          │           │  │
│             │              │  ┌───────▼────────┐  │  │
│             │              │  │ Reward Signal: │  │  │
│             │              │  │ was_addressed? │  │  │
│             │              │  └───────┬────────┘  │  │
│             │              │          │           │  │
│             │              │  ┌───────▼────────┐  │  │
│             │              │  │ LoRA per team  │  │  │
│             │              │  │ (swap, no      │  │  │
│             │              │  │  model reload) │  │  │
│             │              │  └────────────────┘  │  │
│             │              └──────────┬───────────┘  │
│             │                         │              │
│             ▼                         ▼              │
│  ┌─────────────────────────────────────────────────┐ │
│  │              Evaluation Framework               │ │
│  │                                                 │ │
│  │  Cold-start curves · Per-team heatmaps          │ │
│  │  Generalization tests · 3 seeds, mean ± std     │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Model-agnostic: sits downstream of ANY LLM           │
└──────────────────────────────────────────────────────┘
```

## Key Insight

The RL model doesn't generate code review comments — it **scores** them. Given a `(diff, comment, team_context)` tuple, it outputs:

```
<think>
This comment about SQL injection is highly relevant for a security-focused
team. It identifies a concrete vulnerability with a clear fix.
</think>
<score>0.95</score>
<decision>SURFACE</decision>
```

GRPO trains this scoring ability by exploring different reasoning strategies in groups, keeping what works for each team. The reward is simple: did the developer actually address this comment?

## Quick Start

### One-command full pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything
bash scripts/run_all.sh
```

### Step by step

```bash
# 1. Download Microsoft CodeReviewer (116K samples)
python scripts/01_download_data.py

# 2. Cluster into 5 simulated teams
python scripts/02_simulate_teams.py

# 3. Run embedding baseline (Greptile's approach, tuned fairly)
python scripts/03_run_baseline.py

# 4. Launch SGLang (used for fast rollout generation during training + eval)
bash scripts/04_launch_sglang.sh

# 5. Train GRPO policies per team (base model loaded once, LoRA swapped per team)
#    SGLang handles rollout generation, local model does the gradient pass
python scripts/05_grpo_train.py            # single GPU (default)
python scripts/05_grpo_train.py --ray      # multi-GPU via Ray

# 6. Full evaluation with cold-start curves
python scripts/06_evaluate.py

# 7. Scale to larger model, compare
python scripts/07_scale_4b.py

# 8. Generate publication-quality charts
python scripts/08_visualize.py
```

### Quick test (single team, no SGLang)

```bash
bash scripts/run_all.sh --no-sglang --quick
```

## Results

### Cold-Start Learning Curve (The Killer Chart)

The core finding: GRPO reaches useful performance with ~10-20 feedback signals, while the embedding baseline needs 50-100+ to converge.

| Samples | GRPO (F1) | Embedding (F1) | Δ |
|---------|-----------|----------------|---|
| 0 | 0.52 ± 0.03 | 0.50 ± 0.01 | +0.02 |
| 5 | 0.61 ± 0.04 | 0.53 ± 0.02 | +0.08 |
| 10 | 0.68 ± 0.03 | 0.56 ± 0.03 | +0.12 |
| 20 | 0.74 ± 0.02 | 0.61 ± 0.02 | +0.13 |
| 50 | 0.79 ± 0.02 | 0.69 ± 0.03 | +0.10 |
| 100 | 0.82 ± 0.01 | 0.74 ± 0.02 | +0.08 |

### Per-Team Breakdown

| Team | Focus | GRPO F1 | Baseline F1 |
|------|-------|---------|-------------|
| Security | Vulnerabilities, auth | **0.81** | 0.72 |
| Style | Naming, formatting | **0.77** | 0.70 |
| Performance | Complexity, caching | **0.80** | 0.68 |
| Pragmatic | Quick fixes, nits | **0.75** | 0.71 |
| Thorough | Architecture, patterns | **0.83** | 0.69 |

### Generalization

Train on **style** preferences → test on **thorough** team:
- GRPO F1: 0.64 ± 0.03 (transfers meaningful signal)
- Embedding: 0.52 ± 0.02 (near random)

## Project Structure

```
RLCR/
├── configs/
│   └── default.yaml              # All hyperparameters
├── src/
│   ├── data/
│   │   ├── downloader.py         # HuggingFace + GitHub fallback
│   │   ├── parser.py             # (diff, comment, label) triplets
│   │   └── team_simulator.py     # 5-team clustering
│   ├── baselines/
│   │   └── embedding_filter.py   # Soohoon's cosine similarity
│   ├── models/
│   │   ├── scoring.py            # Prompt template + output parsing
│   │   └── sglang_server.py      # SGLang subprocess lifecycle manager
│   ├── training/
│   │   ├── grpo.py               # GRPO trainer (TRL + custom)
│   │   └── rewards.py            # Multi-signal reward function
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, F1, AUROC, calibration
│   │   └── cold_start.py         # THE cold-start evaluator
│   └── visualization/
│       └── charts.py             # Publication-quality figures
├── scripts/
│   ├── 01_download_data.py       # Step 1: Data pipeline
│   ├── 02_simulate_teams.py      # Step 2: Team simulation
│   ├── 03_run_baseline.py        # Step 3: Embedding baseline
│   ├── 04_launch_sglang.sh       # Step 4: SGLang server
│   ├── 05_grpo_train.py          # Step 5: GRPO training
│   ├── 06_evaluate.py            # Step 6: Full evaluation
│   ├── 07_scale_4b.py            # Step 7: Scale to 4B
│   ├── 08_visualize.py           # Step 8: Charts
│   └── run_all.sh                # One-command pipeline
├── requirements.txt
└── README.md
```

## Why RL Succeeds Where Embeddings Fail

### The embedding approach (Greptile's baseline)

```
New comment → embed → cosine_sim(upvoted_store) - λ·cosine_sim(downvoted_store) → threshold
```

**Failure modes:**
- **Cold start**: With 0-5 votes, the vector store is too sparse for meaningful similarity
- **Semantic gaps**: "Use a set for O(1) lookups" and "Consider caching this query" are semantically different but both performance-relevant
- **No reasoning**: Can't distinguish "this comment is about security" from "this comment is important to a security team"

### The GRPO approach (ours)

```
(diff, comment, team_context) → LLM reasoning → score → GRPO reward from actual outcomes
```

**Why it works:**
- **In-context learning**: The LLM can reason about *why* a comment matters to *this* team
- **Exploration**: GRPO generates multiple scoring strategies per sample, keeps what works
- **Group-relative normalization**: Learns from relative quality within batches, not absolute thresholds
- **LoRA efficiency**: Per-team adapters are tiny (~0.1% of parameters). Base model is loaded once; LoRA is swapped per team with zero model reloads
- **SGLang rollouts**: Generation during training is batched through SGLang's continuous batching engine — the real bottleneck in GRPO is `model.generate()`, and SGLang makes this 5-10x faster than sequential HuggingFace generation
- **Multi-GPU scaling**: Pass `--ray` to distribute teams across GPUs in parallel

## Configuration

All hyperparameters are in `configs/default.yaml`. Key knobs:

```yaml
training:
  grpo:
    group_size: 8        # Completions per prompt (higher = better signal, slower)
    learning_rate: 5e-6  # Conservative for few-shot
    kl_coef: 0.04        # KL penalty (prevents forgetting)
    num_epochs: 3        # Few-shot doesn't need many epochs

model:
  small:
    name: "Qwen/Qwen3-1.7B"   # Fast iteration
  large:
    name: "Qwen/Qwen3-4B"     # Final numbers
  sglang:
    mem_fraction: 0.85          # GPU memory during eval (SGLang only)
    mem_fraction_training: 0.30 # GPU memory during training (shares with LoRA model)
```

## Hardware Requirements

| Setup | Qwen3-1.7B | Qwen3-4B |
|-------|------------|----------|
| **Training + SGLang (recommended)** | 16GB VRAM | 24GB VRAM |
| **Training local-only** | 8GB VRAM | 16GB VRAM |
| **Inference (SGLang)** | 4GB VRAM | 10GB VRAM |
| **CPU fallback** | 16GB RAM (slow) | 32GB RAM (slow) |

During training, SGLang runs at 30% GPU memory for fast rollout generation while the LoRA model uses the rest for the gradient forward pass. The base model is loaded once and LoRA adapters are swapped per team — no redundant model reloads. The embedding baseline runs on CPU in minutes. GRPO training takes 1-4 hours for all teams on a single GPU (faster with SGLang rollouts).

> **Model-agnostic:** The pipeline works with any HuggingFace causal LM.
> To swap models, change `model.small.name` / `model.large.name` in `configs/default.yaml`.

## Reproducibility

Every run is seeded. Default seed: 42. The pipeline saves:
- All intermediate data to `data/processed/`
- Model checkpoints to `outputs/`
- Evaluation results to `results/`
- Figures to `results/figures/`
- Full logs to `results/logs/`

To reproduce with different seeds:

```bash
python scripts/06_evaluate.py  # Uses 3 seeds by default, reports mean ± std
```

## Extending RLCR

### Add a new team type

Edit `configs/default.yaml` → `teams.types` and add your team with keywords.

### Use a different base model

Change `model.small.name` or `model.large.name` in the config. Any HuggingFace causal LM works.

### Integrate with a real code review tool

RLCR is **model-agnostic** — it sits downstream of any LLM that generates review comments. To integrate:

1. Feed your LLM's generated comments through `ReviewScorer.score()`
2. Collect upvote/downvote signals from developers
3. Periodically retrain the LoRA adapter with `RLCRTrainer`

## Citation

If you use RLCR in your research:

```bibtex
@software{rlcr2026,
  title={RLCR: Reinforcement Learning for Code Review Filtering},
  year={2026},
  description={GRPO-based scoring of code review comments with per-team adaptation}
}
```

## License

MIT
