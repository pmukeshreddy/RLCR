"""Prompt construction and output parsing for the RL scoring model.

The model receives a structured prompt containing:
  - The code diff
  - The review comment
  - Team context (recent vote history, team description)

And generates:
  - Chain-of-thought reasoning (inside <think> tags)
  - A relevance score from 0.0 to 1.0
  - A binary decision: SURFACE or FILTER
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger


SYSTEM_PROMPT = """You are a code review filter for a {team_type} development team. \
Your job is to decide whether a review comment should be surfaced to the developer \
or filtered out based on this team's preferences.

Team profile: {team_description}

Recent team feedback:
{vote_context}

Analyze the given code diff and review comment. Consider:
1. Is this comment relevant to what this team cares about?
2. Is it actionable — would it lead to a concrete code change?
3. Does it match the patterns of comments this team has valued before?

Respond with your analysis in this exact format:
<think>
[Your reasoning about why this comment is or isn't relevant for this team]
</think>
<score>[0.0 to 1.0]</score>
<decision>[SURFACE or FILTER]</decision>"""


USER_PROMPT = """Code diff:
```
{diff}
```

Review comment: "{comment}"

Score this comment for the {team_type} team."""


@dataclass
class ModelOutput:
    """Parsed output from the scoring model."""

    reasoning: str
    score: float
    decision: str  # "SURFACE" or "FILTER"
    raw_text: str

    @property
    def is_surface(self) -> bool:
        return self.decision == "SURFACE"

    @property
    def binary_label(self) -> int:
        return 1 if self.is_surface else 0


def format_scoring_prompt(
    diff: str,
    comment: str,
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    max_context_votes: int = 6,
) -> list[dict[str, str]]:
    """Build the chat-formatted prompt for the scoring model.

    Returns a list of message dicts compatible with HuggingFace chat templates.
    """
    recent_votes = vote_history[-max_context_votes:] if vote_history else []
    vote_lines = []
    for v in recent_votes:
        icon = "LIKED" if v["vote"] == "upvote" else "DISLIKED"
        vote_lines.append(f'- {icon}: "{v["comment"][:120]}"')
    vote_context = "\n".join(vote_lines) if vote_lines else "(No feedback history yet)"

    system_msg = SYSTEM_PROMPT.format(
        team_type=team_name,
        team_description=team_description,
        vote_context=vote_context,
    )
    user_msg = USER_PROMPT.format(
        diff=diff[:1500],
        comment=comment[:500],
        team_type=team_name,
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def format_prompt_text(
    diff: str,
    comment: str,
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    tokenizer=None,
    max_context_votes: int = 6,
) -> str:
    """Build a single text prompt string for models without chat template support."""
    messages = format_scoring_prompt(
        diff, comment, team_name, team_description, vote_history, max_context_votes
    )
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ImportError, Exception):
            pass
    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"### {role}\n{msg['content']}")
    parts.append("### ASSISTANT\n")
    return "\n\n".join(parts)


def parse_model_output(text: str) -> ModelOutput:
    """Parse the model's structured output into a ModelOutput object.

    Handles various formatting inconsistencies gracefully.
    """
    reasoning = ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()

    score = 0.5
    score_match = re.search(r"<score>\s*([\d.]+)\s*</score>", text)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5
    else:
        num_match = re.search(r"(?:score|rating)[:\s]*([\d.]+)", text, re.IGNORECASE)
        if num_match:
            try:
                score = float(num_match.group(1))
                if score > 1.0:
                    score = score / 10.0
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

    decision = "SURFACE" if score > 0.5 else "FILTER"
    dec_match = re.search(r"<decision>\s*(SURFACE|FILTER)\s*</decision>", text, re.IGNORECASE)
    if dec_match:
        decision = dec_match.group(1).upper()
    else:
        if re.search(r"\b(surface|show|display|relevant|important)\b", text, re.IGNORECASE):
            decision = "SURFACE"
        elif re.search(r"\b(filter|hide|skip|irrelevant|ignore)\b", text, re.IGNORECASE):
            decision = "FILTER"

    return ModelOutput(
        reasoning=reasoning,
        score=score,
        decision=decision,
        raw_text=text,
    )


class ReviewScorer:
    """End-to-end scoring pipeline for code review comments.

    Used for evaluation and as the policy model during GRPO training.
    Default path: SGLang server for fast batched inference.
    Fallback: direct HuggingFace transformers inference (slower, no server needed).

    Initialization order when use_sglang=True (the default):
      1. Probe the SGLang health endpoint
      2. If reachable → use SGLang for all inference
      3. If unreachable → log warning, fall back to local HF model
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_sglang: bool = True,
        sglang_url: str = "http://127.0.0.1:30000",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.sglang_url = sglang_url
        self.use_sglang = False
        self.model = None
        self.tokenizer = None
        self.device = device

        if use_sglang:
            if self._probe_sglang():
                self.use_sglang = True
                logger.success(f"Using SGLang server at {self.sglang_url}")
            else:
                logger.warning(
                    f"SGLang not reachable at {self.sglang_url} — "
                    "falling back to local HuggingFace inference"
                )
                self._init_local_model(device)
        else:
            self._init_local_model(device)

    def _probe_sglang(self) -> bool:
        """Check if SGLang server is running and healthy."""
        try:
            import requests
            resp = requests.get(f"{self.sglang_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _init_local_model(self, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model locally: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        logger.success(f"Model loaded on {device}")

    def score(
        self,
        diff: str,
        comment: str,
        team_name: str,
        team_description: str,
        vote_history: list[dict],
    ) -> ModelOutput:
        """Score a single comment."""
        if self.use_sglang:
            return self._score_sglang(diff, comment, team_name, team_description, vote_history)
        return self._score_local(diff, comment, team_name, team_description, vote_history)

    def _score_local(
        self, diff: str, comment: str, team_name: str,
        team_description: str, vote_history: list[dict],
    ) -> ModelOutput:
        messages = format_scoring_prompt(
            diff, comment, team_name, team_description, vote_history
        )
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ImportError, Exception):
            text = format_prompt_text(
                diff, comment, team_name, team_description, vote_history
            )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return parse_model_output(response)

    def _score_sglang(
        self, diff: str, comment: str, team_name: str,
        team_description: str, vote_history: list[dict],
    ) -> ModelOutput:
        import requests

        messages = format_scoring_prompt(
            diff, comment, team_name, team_description, vote_history
        )
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
        }
        resp = requests.post(
            f"{self.sglang_url}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_model_output(content)

    def batch_score(
        self,
        samples: list[dict[str, Any]],
        team_name: str,
        team_description: str,
        vote_history: list[dict],
        max_workers: int = 64,
    ) -> list[ModelOutput]:
        """Score multiple samples concurrently via SGLang (or sequentially for local)."""
        if not self.use_sglang or len(samples) <= 1:
            return [
                self.score(s["diff"], s["comment"], team_name, team_description, vote_history)
                for s in samples
            ]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import requests

        payloads = []
        for s in samples:
            messages = format_scoring_prompt(
                s["diff"], s["comment"], team_name, team_description, vote_history
            )
            payloads.append({
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
            })

        results: list[ModelOutput] = [None] * len(payloads)

        def _request(idx_payload):
            idx, payload = idx_payload
            try:
                resp = requests.post(
                    f"{self.sglang_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return idx, parse_model_output(content)
            except Exception as e:
                logger.warning(f"Batch score request {idx} failed: {e}")
                return idx, ModelOutput(reasoning="", score=0.5, decision="FILTER", raw_text="")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_request, (i, p)) for i, p in enumerate(payloads)]
            for future in as_completed(futures):
                idx, output = future.result()
                results[idx] = output

        return results
