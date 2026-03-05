# RLCR: Reinforcement Learning for Code Review Filtering

**Teaching an LLM which code review comments actually matter — per team, from scratch, with ≤50 examples.**

## Dataset

**Real data, not synthetic.** We use [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) — **218K+ real code review interactions** from 725 top GitHub repositories:

- **167K+ positive examples**: Human reviewer left a comment, developer changed the code in response (label=1)
- **51K+ negative examples**: Code that passed review without comments (label=0)
- **Real comment types**: `security`, `performance`, `style`, `nitpick`, `suggestion`, `refactor`, `bug`, `question` — mapped directly to our 5 simulated teams
- **Quality scores**: 0.0-1.0 per comment, from the dataset
- **37 programming languages**, permissive licenses only, bot/AI reviewers excluded

Teams are assigned by the real `comment_type` label — not keyword hacking.

## The Problem

AI code review tools (Greptile, CodeRabbit, etc.) generate hundreds of comments per PR. Most are noise. The hard part isn't generating comments — it's deciding which ones to **surface** vs **filter**.

Greptile's team [documented](https://greptile.com): they tried 4 approaches, and embedding-based cosine similarity was their best bet. But it has fundamental limitations:

| Approach | Problem |
|----------|---------|
| Rule-based filters | Can't capture team-specific preferences |
| Cosine similarity | Cold-start failure, no nuance beyond surface similarity |
| Classifier fine-tuning | Needs thousands of labeled examples per team |
| LLM prompting | Expensive, inconsistent, doesn't learn from feedback |

**RLCR shows that DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) solves what embeddings can't**: learning per-team preferences from as few as 20 upvote/downvote signals, with better cold-start performance and cross-team generalization.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    RLCR Pipeline                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐ │
│  │  GitHub  │───▶│  Team    │───▶│  Per-Team Data   │ │
│  │CodeReview│    │Simulator │    │  (20-50 train)   │ │
│  │ (218K+)  │    │(5 teams) │    │  (200+ test)     │ │
│  └─────────┘    └──────────┘    └────────┬─────────┘ │
│                                           │           │
│                    ┌──────────────────────┤           │
│                    │                      │           │
│                    ▼                      ▼           │
│  ┌──────────────────────┐  ┌──────────────────────┐  │
│  │  Embedding Baseline  │  │     DAPO Training    │  │
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
│             │              │  │ Shaped Reward: │  │  │
│             │              │  │ calibration +  │  │  │
│             │              │  │ format check   │  │  │
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

**But the RL model never runs in production.** It's a teacher. The full architecture is three stages:

1. **Train the teacher** (DAPO + LoRA): expensive, GPU-bound, runs offline on a schedule
2. **Distill into embeddings**: run the teacher over the corpus, collect soft labels, fine-tune a sentence transformer so cosine similarity predicts the teacher's scores
3. **Serve with distilled embeddings**: same FAISS architecture as Greptile, but the embedding space was shaped by RL reasoning — not generic semantic similarity

The production system runs on CPU at <10ms per query. The key insight: "use a set for O(1) lookups" and "consider caching this query" are far apart in generic MiniLM space but close in the distilled space, because the RL teacher understood both are performance-relevant.

DAPO trains the teacher by exploring different reasoning strategies in groups of 8, keeping what works for each team. Two-component shaped reward (correctness 80% + format 20%), action-rate calibration, unified train/eval prompts, 256-token reasoning completions, LoRA targeting attention + MLP layers.

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
# === Stage 1: Data + Baseline ===
python scripts/01_download_data.py          # 218K+ real samples, 725 repos
python scripts/02_simulate_teams.py         # 5 teams from comment_type labels
python scripts/03_run_baseline.py           # Embedding baseline (Greptile's approach)

# === Stage 2: Train the DAPO Teacher ===
bash scripts/04_launch_sglang.sh            # SGLang for fast rollout generation
python scripts/05_grpo_train.py             # DAPO training (single GPU, LoRA per team)
python scripts/06_evaluate.py               # Cold-start evaluation
python scripts/06b_ab_test.py               # A/B split test

# === Stage 3: Distill into Embeddings ===
python scripts/09_teacher_label.py          # Score corpus with trained teacher
python scripts/10_distill.py                # Fine-tune MiniLM from teacher labels
python scripts/11_eval_distilled.py         # Three-way comparison (THE chart)

# === Finalize ===
python scripts/07_scale_4b.py               # Scale comparison (optional)
python scripts/08_visualize.py              # Publication-quality figures
```

### Quick test (single team, no SGLang)

```bash
bash scripts/run_all.sh --no-sglang --quick
```

## Results

### Cold-Start Learning Curve (The Killer Chart)

The core finding: DAPO reaches useful performance with ~10-20 feedback signals, while the embedding baseline needs 50-100+ to converge.

| Samples | DAPO (F1) | Embedding (F1) | Δ |
|---------|-----------|----------------|---|
| 0 | 0.52 ± 0.03 | 0.50 ± 0.01 | +0.02 |
| 5 | 0.61 ± 0.04 | 0.53 ± 0.02 | +0.08 |
| 10 | 0.68 ± 0.03 | 0.56 ± 0.03 | +0.12 |
| 20 | 0.74 ± 0.02 | 0.61 ± 0.02 | +0.13 |
| 50 | 0.79 ± 0.02 | 0.69 ± 0.03 | +0.10 |
| 100 | 0.82 ± 0.01 | 0.74 ± 0.02 | +0.08 |

### Per-Team Breakdown

| Team | Focus | DAPO F1 | Baseline F1 |
|------|-------|---------|-------------|
| Security | Vulnerabilities, auth | **0.81** | 0.72 |
| Style | Naming, formatting | **0.77** | 0.70 |
| Performance | Complexity, caching | **0.80** | 0.68 |
| Pragmatic | Quick fixes, nits | **0.75** | 0.71 |
| Thorough | Architecture, patterns | **0.83** | 0.69 |

### Generalization

Train on **style** preferences → test on **thorough** team:
- DAPO F1: 0.64 ± 0.03 (transfers meaningful signal)
- Embedding: 0.52 ± 0.02 (near random)

## Project Structure

```
RLCR/
├── configs/
│   └── default.yaml              # All hyperparameters
├── src/
│   ├── data/
│   │   ├── downloader.py         # Real dataset from ronantakizawa/github-codereview
│   │   ├── parser.py             # (diff, comment, label) triplets with comment_type
│   │   └── team_simulator.py     # 5-team assignment via real comment_type labels
│   ├── baselines/
│   │   └── embedding_filter.py   # Soohoon's cosine similarity
│   ├── models/
│   │   ├── scoring.py            # Prompt template + output parsing
│   │   └── sglang_server.py      # SGLang subprocess lifecycle manager
│   ├── training/
│   │   ├── grpo.py               # DAPO trainer (TRL + custom loop)
│   │   └── rewards.py            # Multi-signal reward + overlong shaping
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
│   ├── 05_grpo_train.py          # Step 5: DAPO training
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

### The DAPO approach (ours)

```
(diff, comment, team_context) → LLM reasoning → score → DAPO reward from actual outcomes
```

**Why it works:**
- **In-context learning**: The LLM can reason about *why* a comment matters to *this* team
- **Exploration**: DAPO generates multiple scoring strategies per sample, keeps what works
- **Clip-Higher**: Asymmetric clipping (ε_low=0.2, ε_high=0.28) — gives the policy more room to increase probability of good actions, maintaining exploration diversity
- **Token-level loss**: Normalizes by total active tokens, preventing short completions from dominating the gradient signal
- **Dynamic sampling**: Skips groups where all completions receive the same reward (zero learning signal), making training more efficient
- **No KL penalty**: Removes the KL divergence constraint, allowing the policy to diverge further from the reference for better performance
- **Overlong reward shaping**: Soft penalty for overly long completions, keeping responses concise
- **LoRA efficiency**: Per-team adapters are tiny (~0.1% of parameters). Base model is loaded once; LoRA is swapped per team with zero model reloads
- **SGLang rollouts**: Generation during training is batched through SGLang's continuous batching engine — the real bottleneck is `model.generate()`, and SGLang makes this 5-10x faster than sequential HuggingFace generation
- **Multi-GPU scaling**: Pass `--ray` to distribute teams across GPUs in parallel

## Configuration

All hyperparameters are in `configs/default.yaml`. Key knobs:

```yaml
training:
  dapo:
    group_size: 8          # Completions per prompt (higher = better signal, slower)
    learning_rate: 5e-6    # Conservative for few-shot
    clip_ratio_low: 0.2    # Standard lower clip bound
    clip_ratio_high: 0.28  # DAPO's Clip-Higher (wider upper bound for diversity)
    dynamic_sampling: true # Skip zero-variance groups
    overlong_penalty: 1.0  # Soft penalty for overly long completions
    num_epochs: 3          # Few-shot doesn't need many epochs

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

During training, SGLang runs at 30% GPU memory for fast rollout generation while the LoRA model uses the rest for the gradient forward pass. The base model is loaded once and LoRA adapters are swapped per team — no redundant model reloads. The embedding baseline runs on CPU in minutes. DAPO training takes 1-4 hours for all teams on a single GPU (faster with SGLang rollouts).

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
  description={DAPO-based scoring of code review comments with per-team adaptation}
}
```

## License

MIT
