"""Result records shared by every lgeval module (one JSON object per line in `<name>.jsonl`)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(UTC)


class Question(BaseModel):
    id: str
    question: str  # the research goal given to the system
    dataset: str
    gold_answer: str | None = None  # short answer, when the dataset has one (FRAMES)
    gold_facts: list[str] = Field(default_factory=list)  # facts a good brief should contain (optional)
    sources: list[str] = Field(default_factory=list)  # reference URLs (optional)
    meta: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(tokens_in=self.tokens_in + other.tokens_in, tokens_out=self.tokens_out + other.tokens_out)


class Manifest(BaseModel):
    experiment: str  # quality | baseline | import | recompute | score | pairwise
    created_at: datetime = Field(default_factory=now)
    git_sha: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)  # CLI args as given
    configs: dict[str, dict[str, str]] = Field(default_factory=dict)  # config name -> env overrides
    models: dict[str, str] = Field(default_factory=dict)  # role -> model id (smart, fast, judge, ...)
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)  # model -> {in_per_m, out_per_m}


class Citation(BaseModel):
    sentence: str
    claim_keys: list[str] = Field(default_factory=list)  # LiveGraph: (C1) refs in the sentence
    chunk_ids: list[str] = Field(default_factory=list)  # evidence chunks behind those claims
    urls: list[str] = Field(default_factory=list)  # external baselines cite URLs


class ClaimOut(BaseModel):
    key: str
    text: str
    confidence: float | None = None
    status: Literal["active", "struck"] = "active"
    support_chunk_ids: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """A chunk (or fetched page excerpt) the system saw; what citation scoring checks sentences against."""

    chunk_id: str
    title: str = ""
    url: str | None = None
    text: str


class RunRecord(BaseModel):
    question_id: str
    dataset: str
    system: str  # config name ("full", "no_debate", ...) or external system ("chatgpt_dr", "single_agent")
    repeat: int = 0
    run_id: str | None = None  # LiveGraph run id (None for external/baseline)
    status: str  # done | partial (side node failed, brief complete) | failed | timeout | cancelled | imported
    error: str | None = None
    started_at: datetime = Field(default_factory=now)
    wall_s: float | None = None
    usage: dict[str, TokenUsage] = Field(default_factory=dict)  # model -> tokens
    cost_usd: float | None = None
    n_nodes: int | None = None
    brief: str = ""
    claims: list[ClaimOut] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    snapshot_path: str | None = None


Mutation = Literal["distrust", "steer", "edit", "identical", "patch_question"]


class RecomputeRecord(BaseModel):
    question_id: str
    run_id: str
    mutation: Mutation
    target: str  # node key or document id the mutation was applied to
    detail: str = ""  # steer text / edit summary / document title
    wall_s: float
    usage: dict[str, TokenUsage] = Field(default_factory=dict)  # tokens of versions created by the update
    cost_usd: float
    nodes_total: int  # active nodes in the run
    nodes_impacted: int  # GET /impact upper bound
    nodes_recomputed: int  # new versions with an LLM call
    nodes_reused: int  # node.version reused=true (hash hit or early cutoff)
    baseline: Literal["fresh_run", "original_run"]  # what "start over" was measured as
    baseline_wall_s: float
    baseline_cost_usd: float
    brief_before: str = ""
    brief_after: str = ""
    baseline_brief: str = ""  # fresh run's brief, when baseline == fresh_run (for quality parity)
    meta: dict[str, Any] = Field(default_factory=dict)


class ScoreRecord(BaseModel):
    question_id: str
    system: str
    repeat: int = 0
    metric: str  # factual_precision | citation_precision | citation_recall | answer_accuracy | brier | ece | ...
    value: float
    n: int | None = None  # items the metric was computed over (atomic claims, sentences, ...)
    judge_model: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PairwiseRecord(BaseModel):
    question_id: str
    system_a: str
    system_b: str
    repeat: int = 0
    winner: Literal["a", "b", "tie"]  # after reconciling both orderings
    first_order: Literal["a", "b", "tie"]  # verdict with A shown first
    swapped_order: Literal["a", "b", "tie"]  # verdict with B shown first (already mapped back to a/b)
    judge_model: str
    rationale: str = ""


class ClaimLabel(BaseModel):
    """Truth label of one LiveGraph claim, for calibration (Claim B)."""

    question_id: str
    system: str
    repeat: int = 0
    claim_key: str
    claim_text: str
    confidence: float | None
    struck: bool = False
    label: Literal["true", "false", "unverifiable"]
    labeler: str  # "human:<name>" or "llm:<model>"
    rationale: str = ""


class JudgeCost(BaseModel):
    """Spend of one `score`/`pairwise` invocation (judge calls are shared across runs, so it is logged per call)."""

    command: Literal["score", "pairwise", "label"]
    judge_model: str
    calls: int
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float
    created_at: datetime = Field(default_factory=now)
