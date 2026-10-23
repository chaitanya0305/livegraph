# LiveGraph evaluation protocol

This is the pre-registered protocol for evaluating LiveGraph. It is written **before** the full study is
run: hypotheses, systems, metrics, analysis and stopping rules are fixed here, and any deviation is
recorded in the [Deviations log](#deviations-log) at the end. The harness is `apps/api/lgeval`
(contract: [`apps/api/lgeval/SPEC.md`](../apps/api/lgeval/SPEC.md)); generated results land in
[`docs/results/`](results/results.md).

## 1. Hypotheses

| | Claim | Hypothesis | Primary measure | Test |
|---|---|---|---|---|
| **A** | Quality | Living Brief (`full`) produces briefs with higher factual precision and better citation precision/recall than its ablations and a single-agent baseline using the same models and search. | Factual precision; citation precision/recall | Paired bootstrap over questions (95% CI of the difference excludes 0); pairwise judge win rate with Wilson CI excludes 0.5 |
| **B** | Calibration | The judge's per-claim confidence predicts whether the claim is true. | Brier score, ECE, AUROC | AUROC 95% bootstrap CI excludes 0.5; ECE reported with its reliability diagram |
| **C** | Incremental recompute | After a change (distrust a source, steer a node, edit an output, identical rerun), LiveGraph updates the brief at a fraction of the cost and wall time of starting over, with no loss of quality. | Update cost and time as % of baseline; quality of the updated brief vs. a fresh run | Mean % with 95% bootstrap CI; quality parity: CI of (updated − fresh) factual precision contains 0 or is positive |
| **D** | Usability (user study, §9) | Users correct a planted error faster and more often with LiveGraph than by re-prompting a chat assistant. | Time-to-fix, success rate | Within-subject comparison (Wilcoxon signed-rank on time; McNemar on success) |

A–C are measured automatically by `lgeval`; D is a small human study. We expect C to hold strongly
(identical reruns should cost ≈0 thanks to input-hash reuse and early cutoff), A to hold against the
single-agent baseline and `single_shot`, and to be smaller or null against single ablations; a null
result for an ablation is reported as such.

## 2. Systems and ablations

All LiveGraph configurations run the same code, the same models and the same search provider. Ablations
are environment overrides on the API process (`lgeval/runners/configs.py`), default-off in production.

| System | What changes | Isolates |
|---|---|---|
| `full` | Living Brief as shipped (planner, researchers + critics, claims, prosecutor vs. defender, judge, editor) | — |
| `no_debate` | Prosecutor and defender removed; the judge scores claims from evidence alone | adversarial debate |
| `no_critic` | No critic nodes, so no follow-up research | self-critique / research expansion |
| `single_shot` | `AGENT_TOOLS` emptied: every node is one model call without tools | agentic tool use |
| `no_context_tools` | `CONTEXT_TOOLS=false` | cross-agent context reading |
| `single_agent` | One tool-using agent with the same models and the same `web_search`/`search_documents` tools writes a cited brief in one go | the whole multi-agent graph |
| external (`chatgpt_dr`, `gemini_dr`, `perplexity_dr`, ...) | Commercial deep-research products, collected manually and imported | state of practice |

**Collecting external products fairly.** External systems are not part of the hypothesis tests (their
models, tools and budgets differ and cannot be controlled); they are reported descriptively.
1. Use the **exact question text** given to LiveGraph, pasted verbatim with no system prompt, custom
   instructions, memory, or attached files (disable memory/personalization; use a fresh chat).
2. Collect all questions for one product **within the same 7-day window** as the LiveGraph runs, and
   record the date, product name, plan tier and any visible model/mode name per brief.
3. **No follow-ups.** If the product asks a clarifying question, reply with the fixed text
   "Please proceed with your best judgement." once, and nothing else. Do not regenerate or cherry-pick:
   the first completed answer is the one imported. Failures (refusals, timeouts) are imported as failures.
4. Export the answer with its citations (Markdown or copied text with URLs) to
   `briefs/<system>/<question_id>.md` and import with `python -m lgeval import --system <system> --dir briefs/<system>`.
   Imported briefs cite URLs; citation scoring fetches nothing new and scores against the cited pages'
   text recorded at import time (`RunRecord.evidence`); pages that cannot be fetched count as unsupported.
5. Cost is recorded as unknown (subscription products); wall time is recorded if the product shows it.

## 3. Datasets

- **Custom set** (`lgeval/datasets/custom.jsonl`): 20–40 open research questions of the kind LiveGraph is
  built for (multi-source, contested or time-sensitive), written before any system was run on them, with
  optional `gold_facts` (facts a good brief should contain) and reference `sources`. Used for A, B, C.
- **FRAMES** subset (Krishna et al., 2024; `google/frames-benchmark`): multi-hop questions with a short
  gold answer and Wikipedia sources. A seeded random sample (`--limit N --seed 0`) is used for answer
  accuracy in A. Questions are given as research goals ("Research and answer: …").
- **BrowseComp** (Wei et al., 2025) is noted as a harder browsing benchmark; it is out of scope for the
  pilot (short-answer, needle-finding tasks fit a brief-writing system poorly) and may be added later.

## 4. Metrics

| Metric | Definition | Reference |
|---|---|---|
| Factual precision | Brief → atomic facts (LLM decomposition); each fact judged supported/unsupported against the evidence the system retrieved (and, for externals, the cited pages); precision = supported / all | FActScore (Min et al., 2023); SAFE/LongFact (Wei et al., 2024) |
| Citation precision | Share of citations whose evidence supports the sentence they are attached to (NLI-style judge) | ALCE (Gao et al., 2023) |
| Citation recall | Share of factual sentences fully supported by the union of their citations | ALCE (Gao et al., 2023) |
| Answer accuracy | FRAMES: the brief's answer graded correct / incorrect / not attempted against the gold answer by a grader prompt; accuracy = correct / all | SimpleQA-style grading (Wei et al., 2024) |
| Pairwise preference | Judge compares two briefs on factuality, sourcing and usefulness; run in both orders, a win counts only if both orders agree, else tie; win rate = (W + T/2)/n with Wilson 95% CI; Bradley–Terry strengths when ≥3 systems are compared | Zheng et al. (2023); Bradley & Terry (1952); Wilson (1927) |
| Calibration | Each LiveGraph claim labelled true/false/unverifiable against evidence + web (LLM labeler, human-audited); Brier score, ECE (10 equal-width bins), AUROC of confidence vs. truth, reliability diagram; struck-false rate = share of struck claims that are false | Brier (1950); Naeini et al. (2015); Guo et al. (2017) |
| Recompute efficiency | Per update: cost and wall time as % of the baseline ("start over" = a fresh run of the mutated goal, or the original run), nodes recomputed / reused / impacted (`GET /impact` upper bound) | — |
| Cost, time | Token usage per model × OpenRouter prices pinned at study start; wall time from run creation to settled | — |

**Uncertainty.** Repeats are averaged per question first; the question is the resampling unit. 95%
percentile bootstrap CIs (Efron, 1979) with 2,000 resamples and a fixed seed. System differences use a
paired bootstrap over questions. No correction for multiple comparisons is applied to secondary metrics;
primary comparisons are `full` vs. `single_agent` and `full` vs. each ablation on factual precision
(Holm-corrected).

## 5. Judge model and bias mitigation

- **Different model family.** In the study every LiveGraph agent runs on `deepseek/deepseek-v4.1-flash`. The
  judge/scorer is from another family — default `xiaomi/mimo-v2.6-pro` via `deepinfra/fp8`, low reasoning effort (the
  strongest model under ~$1/M on the Artificial Analysis Intelligence Index, Oct 2026) — pinned by exact model id in the
  manifest, to limit self-preference bias (Panickssery et al., 2024; Zheng et al., 2023). Tradeoff: a cheap
  judge is a weaker verifier than a frontier model, which is why the human spot-check below gates every metric.
- **Groundedness, not world truth.** Factual precision and claim labels check a brief against the evidence
  its own run retrieved (plus `gold_answer`/`gold_facts` when the dataset has them). For systems that report
  no evidence (external deep-research products, imported briefs) the scorer has only their cited URLs, so
  always score them with `--fetch-urls`; otherwise their facts come out unsupported by construction.
- **Batched judgments.** Citation support is judged a few sentences per call by the LLM judge rather than
  ALCE's per-pair NLI model, to bound cost; it is somewhat less strict, and the spot-check covers it.
- **Position swap.** Every pairwise comparison runs twice with the order swapped; disagreement → tie.
  The swap-disagreement rate is reported.
- **Blinding.** System names, node names and LiveGraph-specific markup (claim keys `(C1)`) are stripped
  or normalized to the same citation format before judging.
- **Temperature 0** for all judge calls; prompts are versioned in `lgeval/metrics/`.
- **Human spot-check.** A random, stratified ≥10% sample of judge decisions (atomic-fact support, citation
  support, claim labels, pairwise verdicts) is labelled independently by two humans blind to system and to
  the judge's verdict. We report Cohen's κ (Cohen, 1960) between humans and between each human and the
  judge. If judge–human κ < 0.6 on a metric, that metric is reported as exploratory only.

## 6. Controls

- **Fixed models and pricing.** Model ids are pinned in the manifest; OpenRouter prices are fetched once at
  study start and cached (`costs.py`), so costs are comparable across the study.
- **Web drift.** `web_cache` makes recomputes within a run reproducible; across systems, all runs for a
  question are executed within the same 48-hour window, interleaved by system (not system-by-system).
- **Repeats = 3** per (question, system); results average repeats per question.
- **Temperature**: the provider default used in production for LiveGraph and the single-agent baseline
  (the system is evaluated as shipped); 0 for all judges.
- **Budget guard**: every command takes `--max-usd` and `--dry-run`; a run stops starting new work once
  measured spend reaches the cap. Failed runs are kept and counted (as zero-quality briefs in the
  intention-to-treat table, and excluded in the per-protocol table); the failure rate is reported.
- **Order and caching**: runs are shuffled with a fixed seed; no prompt caching across systems.

## 7. Budget

Measured tokens of one LiveGraph run (8 real runs, Oct 2026): ~72K in / 21K out on the fast model and
~28K in / 3K out on the smart model. The study runs **every role on `deepseek/deepseek-v4.1-flash`**
(`lgeval/runners/configs.py: STUDY_MODELS`, independent of `apps/api/.env`) through the cheapest provider,
`baidu/fp8` ($0.054 / $0.215 per M), **≈ $0.011/run**. Costs are priced at the pinned provider's endpoint
(`costs.PROVIDER_PINS`), not the list price. The judge (`xiaomi/mimo-v2.6-pro`) costs ≈ $0.03 per scored
brief, so judging, not running the systems, is most of the budget. Orders of magnitude:

| Suite | Contents | Est. cost |
|---|---|---|
| `pilot` | 10 custom questions × 5 configs × 1 repeat, single-agent baseline, recompute on 5 runs × 4 mutations, scoring | ≈ $4 |
| `full` | 30 custom questions × 6 systems × 3 repeats, recompute on 20 runs × 4 mutations, full scoring + pairwise + claim labels | ≈ $26 |

OpenRouter falls back to another provider when the pinned one is down, which costs more than the estimate;
check the real spend on the OpenRouter activity page after each step. Exa search is billed separately by Exa. Since all systems share the same models, the ablation comparisons
are unaffected by the model choice; the absolute numbers describe LiveGraph on DeepSeek v4.1 Flash.

The authoritative estimate is printed by `python -m lgeval plan --suite pilot|full` (dry run: runs ×
configs × repeats and estimated USD, using current pricing). Run the pilot first and re-plan with the
pilot's measured per-run cost. The actual spend of the study is reported in Table 5 of the results.

## 8. Reproducing end to end

```bash
cd apps/api
uv run pytest lgeval/tests -q                       # offline harness tests (make eval-test)
make -C ../.. db                                    # Postgres for the API under test
uv run python -m lgeval plan --suite pilot          # dry-run estimate

# Claim A
uv run python -m lgeval quality  --dataset custom --configs full,no_debate,no_critic,single_shot,no_context_tools \
                                 --repeats 3 --limit 10 --max-usd 20
uv run python -m lgeval quality  --dataset frames --configs full --repeats 3 --limit 30 --max-usd 20
uv run python -m lgeval baseline --dataset custom --limit 10 --max-usd 5
uv run python -m lgeval import   --system chatgpt_dr --dir briefs/chatgpt_dr     # optional, see §2
# Claim C
uv run python -m lgeval recompute --from ../../eval-results/quality/<tag> \
                                  --mutations distrust,steer,edit,identical --max-usd 10
# Scoring (Claims A, B)
uv run python -m lgeval score    --results ../../eval-results --metrics factual,citation,accuracy,calibration \
                                 --max-usd 20 --fetch-urls
uv run python -m lgeval pairwise --results ../../eval-results --a full --b no_debate
# Report
uv run python -m lgeval report   --results ../../eval-results --out ../../docs/results
```

`report` writes `docs/results/results.md` (Tables 1–5), `docs/results/tables/*.tex` (input by the paper),
`docs/results/figures/*.svg` and `docs/results/summary.json`. Every result directory carries a
`manifest.json` (git SHA, args, configs, model ids, pricing). Raw results (`eval-results/`) are
gitignored; we publish them as a release artifact alongside the paper.

## 9. Human user study (Claim D)

- **Participants**: 8–15 adults who use AI chat assistants at least weekly; recruited from colleagues and
  university mailing lists; no compensation contingent on performance.
- **Task**: each participant receives two research briefs, each with **one planted error** (a wrong number
  traced to a distrusted/incorrect source, or a claim with no support). They must find and correct it so
  the final brief is consistent. Conditions: **LiveGraph** (canvas: inspect claims, distrust a source,
  steer/edit a node, watch the downstream recompute) vs. **chat re-prompting** (the same brief in a chat
  assistant with the same models; the participant can only send messages).
- **Design**: within-subject, two different brief topics; order of condition and topic **counterbalanced**
  (Latin square). 5-minute tutorial per condition; 10-minute cap per task.
- **Measures**: time-to-fix (s, capped), success (error corrected and no new error introduced; judged by
  two raters, κ reported), number of interactions, SUS (Brooke, 1996) and NASA-TLX raw (Hart & Staveland,
  1988) after each condition, and a short semi-structured interview.
- **Analysis**: Wilcoxon signed-rank on time and SUS/TLX; McNemar on success; effect sizes with bootstrap
  CIs. With n ≤ 15 the study is reported as formative.
- **Ethics and consent**: written informed consent (purpose, data kept, right to withdraw at any time
  without giving a reason), screen and interaction logs only (no audio/video unless separately consented),
  pseudonymous IDs, data deleted after 12 months. Check whether institutional review (IRB/ethics board)
  is required at your institution before recruiting; obtain it if so.

## 10. Threats to validity

- **Web drift.** Search results change over time, so runs of different systems may see different
  sources. Mitigation: interleaved execution within a 48-hour window, `web_cache` within runs, evidence
  stored per run so scoring is against what each system actually saw.
- **LLM-judge bias.** Judges prefer longer answers, their own family's outputs and the first position.
  Mitigation: cross-family judge, position swap, blinding, human spot-check with κ; any metric with low
  agreement is demoted to exploratory.
- **Scoring against retrieved evidence.** Factual precision checks support in the evidence a system
  retrieved, which rewards faithful citation but cannot catch errors present in the sources themselves.
  The calibration labels use independent web verification to partially address this.
- **Small n.** 10–40 questions per dataset give wide CIs; we report CIs everywhere and avoid claims the
  CIs do not support.
- **External products change** silently and cannot be pinned; they are reported descriptively with the
  collection date and are not part of hypothesis tests.
- **Custom set authored by the system's builders.** Questions were written before any run, but may still
  suit LiveGraph's design; FRAMES is the external check.
- **Cost estimates** depend on OpenRouter pricing at study time; token counts are reported so costs can
  be recomputed at other prices.

## Deviations log

| Date | Change | Reason |
|---|---|---|
| — | — | — |
